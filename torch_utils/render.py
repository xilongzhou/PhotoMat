

import torch 
import numpy as np

from PIL import Image

import torch.distributions as tdist

eps = 1e-6

# light pos to crop
def pos_to_crop(pos, fov = 4):

	if len(pos.shape)==2:

		c_h = 128 + pos[:,1]*256/fov
		c_w = 128 - pos[:,0]*256/fov

		c_h = [int(item) for item in c_h]
		c_w = [int(item) for item in c_w]

		return c_h, c_w

	elif len(pos.shape)==1:

		c_h = 128 + pos[1]*256/fov
		c_w = 128 - pos[0]*256/fov

		c_h = int(c_h)
		c_w = int(c_w)

		return c_h, c_w




# set up size, light pos, light intensity
def set_param(device='cuda'):

	size = 4.0

	light_pos = torch.tensor([0.0, 0.0, 4], dtype=torch.float32).view(1, 3, 1, 1)
	light = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).view(1, 3, 1, 1) * 16 * np.pi

	light_pos = light_pos.to(device)
	light = light.to(device)

	return light, light_pos, size


def AdotB(a, b):
	return (a*b).sum(1, keepdim=True).clamp(min=0).expand(-1,3,-1,-1)


def norm(vec): #[B,C,W,H]
	vec = vec.div(vec.norm(2.0, 1, keepdim=True)+eps)
	return vec


def GGX(cos_h, alpha):
	c2 = cos_h**2
	a2 = alpha**2
	den = c2 * a2 + (1 - c2)
	return a2 / (np.pi * den**2 + 1e-6)

def Beckmann( cos_h, alpha):
	c2 = cos_h ** 2
	t2 = (1 - c2) / c2
	a2 = alpha ** 2
	return torch.exp(-t2 / a2) / (np.pi * a2 * c2 ** 2)

def Fresnel(cos, f0):
	return f0 + (1 - f0) * (1 - cos)**5

def Fresnel_S(cos, specular):
	sphg = torch.pow(2.0, ((-5.55473 * cos) - 6.98316) * cos)
	return specular + (1.0 - specular) * sphg

def Smith(n_dot_v, n_dot_l, alpha):
	def _G1(cos, k):
		return cos / (cos * (1.0 - k) + k)
	k = (alpha * 0.5).clamp(min=1e-6)
	return _G1(n_dot_v, k) * _G1(n_dot_l, k)

# def norm(vec): #[B,C,W,H]
# 	vec = vec.div(vec.norm(2.0, 1, keepdim=True))
# 	return vec

def getDir(pos, tex_pos):
	vec = pos - tex_pos
	return norm(vec), (vec**2).sum(1, keepdim=True)

# def AdotB(a, b):
# 	return (a*b).sum(1, keepdim=True).clamp(min=0).expand(-1,3,-1,-1)
def getTexPos(res, size, device='cpu'):
	x = torch.arange(res, dtype=torch.float32)
	x = ((x + 0.5) / res - 0.5) * size

	# surface positions,
	y, x = torch.meshgrid((x, x))
	z = torch.zeros_like(x)
	pos = torch.stack((x, -y, z), 0).to(device)

	return pos

def compute_cos(x,y):
	return (x*y).sum(1,keepdim=True)[:,0:1,0:1,0:1]

# point light
def render(maps, tex_pos, li_color, li_pos, post='gamma', device='cuda', isMetallic=False, no_decay=False, amb_li=False, cam_pos=None, dir_flag=False):

	if len(li_pos.shape)!=4:
		li_pos = li_pos.unsqueeze(-1).unsqueeze(-1)

	assert len(li_color.shape)==4, "dim of the shape of li_color pos should be 4"
	assert len(li_pos.shape)==4, f"dim of the shape of camlight pos {li_pos.shape} should be 4"
	assert len(tex_pos.shape)==4, "dim of the shape of position map should be 4"
	assert len(maps.shape)==4, "dim of the shape of feature map should be 4"
	assert li_pos.shape[1]==3, "the 1 channel of position map should be 3"

	# print(" maps: ",maps.shape)
	if maps.shape[1]==12:
		use_spec = True
		spec = maps[:,9:12,:,:]
	else:
		use_spec = False

	# print('use_spec: ', use_spec)

	if cam_pos is None:
		cam_pos = li_pos

	normal = maps[:,0:3,:,:]
	albedo = maps[:,3:6,:,:]
	rough = maps[:,6:9,:,:]
	if isMetallic:
		metallic = maps[:,9:12,:,:]
		f0 = 0.04
		# update albedo using metallic
		f0 = f0 + metallic * (albedo - f0)
		albedo = albedo * (1.0 - metallic) 
	else:
		f0 = 0.04

	if dir_flag:
		l = norm(li_pos)

		# v_dir = torch.zeros_like(l).to(l.device)
		# v_dir[:,-1,:,:]=1
		# cos = compute_cos(l, v_dir)
		# dist_l_sq = (4/cos)**2

		dist_l_sq = 16

		v = l
	else:
		l, dist_l_sq = getDir(li_pos, tex_pos)
		v, _ = getDir(cam_pos, tex_pos)


	h = norm(l + v)
	normal = norm(normal)

	n_dot_v = AdotB(normal, v)
	n_dot_l = AdotB(normal, l)
	n_dot_h = AdotB(normal, h)
	v_dot_h = AdotB(v, h)

	# print('dist_l_sq:',dist_l_sq)
	if no_decay:
		geom = n_dot_l
	else:
		geom = n_dot_l / (dist_l_sq + eps)

	D = GGX(n_dot_h, rough**2)
	if use_spec:
		F = Fresnel_S(v_dot_h, spec)
	else:
		F = Fresnel(v_dot_h, f0)
	G = Smith(n_dot_v, n_dot_l, rough**2)

	## lambert brdf
	f1 = albedo / np.pi
	if use_spec:
		f1 *= (1-spec)

	## cook-torrance brdf
	f2 = D * F * G / (4 * n_dot_v * n_dot_l + eps)
	f = f1 + f2
	img = f * geom * li_color

	if amb_li:
		amb_intensity = 0.05
		amb_light = torch.rand([img.shape[0],img.shape[1],1,1], device=device)*amb_intensity
		# amb_light = albedo*amb_intensity

		img = img + amb_light

	if post=='gamma':
		# print('gamma')
		return img.clamp(eps, 1.0)**(1/2.2)
	elif post=='reinhard':
		# print('reinhard')
		img = img.clamp(min=eps)
		return img/(img+1)
	elif post=='hdr':
		# print('hdr')
		return img.clamp(min=eps)



# car paint render
def render_carpaint(maps, tex_pos, li_color, li_pos, post='gamma', device='cuda', isMetallic=False, no_decay=False, amb_li=False, cam_pos=None, dir_flag=False):

	if len(li_pos.shape)!=4:
		li_pos = li_pos.unsqueeze(-1).unsqueeze(-1)

	assert len(li_color.shape)==4, "dim of the shape of li_color pos should be 4"
	assert len(li_pos.shape)==4, f"dim of the shape of camlight pos {li_pos.shape} should be 4"
	assert len(tex_pos.shape)==4, "dim of the shape of position map should be 4"
	assert len(maps.shape)==4, "dim of the shape of feature map should be 4"
	assert li_pos.shape[1]==3, "the 1 channel of position map should be 3"

	assert maps.shape[1]==11, f"channel num of car paint renderer should be 12, but got {maps.shape[1]}"


	color = maps[:,0:3,:,:]
	top_spec_r = maps[:,3:4,:,:]
	top_spec = maps[:,4:7,:,:]
	bot_spec_r = maps[:,7:8,:,:]
	bot_spec = maps[:,8:11,:,:]

	# print('use_spec: ', use_spec)

	if cam_pos is None:
		cam_pos = li_pos

	normal = torch.tensor([0,0,1.]).to(maps.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

	l, dist_l_sq = getDir(li_pos, tex_pos)

	v, _ = getDir(cam_pos, tex_pos)
	h = norm(l + v)
	normal = norm(normal)

	n_dot_v = AdotB(normal, v)
	n_dot_l = AdotB(normal, l)
	n_dot_h = AdotB(normal, h)
	v_dot_h = AdotB(v, h)

	geom = n_dot_l / (dist_l_sq + eps)

	# f_mf1
	D1 = GGX(n_dot_h, top_spec_r**2)
	# G1 = Smith(n_dot_v, n_dot_l, top_spec_r**2)
	f_mf1 = D1 / (4 * n_dot_v * n_dot_l + eps)

	# f_mf2
	D2 = GGX(n_dot_h, bot_spec_r**2)
	# G2 = Smith(n_dot_v, n_dot_l, bot_spec_r**2)
	f_mf2 = D2 / (4 * n_dot_v * n_dot_l + eps)


	# final
	img = top_spec * f_mf1 + bot_spec * f_mf2 + color * (1 / np.pi) 
	img = img * geom * li_color

	if post=='gamma':
		# print('gamma')
		return img.clamp(eps, 1.0)**(1/2.2)
	elif post=='reinhard':
		# print('reinhard')
		img = img.clamp(min=eps)
		return img/(img+1)
	elif post=='hdr':
		# print('hdr')
		return img.clamp(min=eps)

#[B,c,H,W]
def height_to_normal(img_in, size, intensity=0.2): # 0.02 for debugging, 0.2 is regular setting

	"""Atomic function: Normal (https://docs.substance3d.com/sddoc/normal-172825289.html)

	Args:
		img_in (tensor): Input image.
		mode (str, optional): 'tangent space' or 'object space'. Defaults to 'tangent_space'.
		normal_format (str, optional): 'gl' or 'dx'. Defaults to 'dx'.
		use_input_alpha (bool, optional): Use input alpha. Defaults to False.
		use_alpha (bool, optional): Enable the alpha channel. Defaults to False.
		intensity (float, optional): Normalized height map multiplier on dx, dy. Defaults to 1.0/3.0.
		max_intensity (float, optional): Maximum height map multiplier. Defaults to 3.0.

	Returns:
		Tensor: Normal image.
	"""
	# grayscale_input_check(img_in, "input height field")
	assert img_in.shape[1]==1, 'should be grayscale image'

	def roll_row(img_in, n):
		return img_in.roll(n, 2)

	def roll_col(img_in, n):
		return img_in.roll(n, 3)

	def norm(vec): #[B,C,W,H]
		vec = vec.div(vec.norm(2.0, 1, keepdim=True))
		return vec

	img_size = img_in.shape[2]
	
	img_in = img_in*intensity

	dx = (roll_col(img_in, 1) - roll_col(img_in, -1))
	dy = (roll_row(img_in, 1) - roll_row(img_in, -1))
	
	pixSize = size / img_in.shape[-1]
	dx /= 2 * pixSize
	dy /= 2 * pixSize

	img_out = torch.cat((dx, -dy, torch.ones_like(dx)), 1)
	img_out = norm(img_out)
	# img_out = img_out / 2.0 + 0.5 #[-1,1]->[0,1]
	
	return img_out


# xy to normal
def xy_to_normal(img_in):

	assert img_in.shape[1]==2, 'should be 2 channel normal'

	img_in = img_in*2-1
	img_out = torch.cat((img_in, torch.ones_like(img_in[:,0:1,...])),dim=1) #[-1,1] (times 3 to make normal stronger)

	img_out = norm(img_out)

	return img_out

# gaussian regularization
def gaussian_reg(img_in, size, filter, intensity=0.2): # 0.02 for debugging, 0.2 is regular setting

	"""Atomic function: Normal (https://docs.substance3d.com/sddoc/normal-172825289.html)

	Args:
		img_in (tensor): Input image.
		mode (str, optional): 'tangent space' or 'object space'. Defaults to 'tangent_space'.
		normal_format (str, optional): 'gl' or 'dx'. Defaults to 'dx'.
		use_input_alpha (bool, optional): Use input alpha. Defaults to False.
		use_alpha (bool, optional): Enable the alpha channel. Defaults to False.
		intensity (float, optional): Normalized height map multiplier on dx, dy. Defaults to 1.0/3.0.
		max_intensity (float, optional): Maximum height map multiplier. Defaults to 3.0.

	Returns:
		Tensor: Normal image.
	"""
	# grayscale_input_check(img_in, "input height field")
	assert img_in.shape[1]==1, 'should be grayscale image'

	def roll_row(img_in, n):
		return img_in.roll(n, 2)

	def roll_col(img_in, n):
		return img_in.roll(n, 3)

	def norm(vec): #[B,C,W,H]
		vec = vec.div(vec.norm(2.0, 1, keepdim=True))
		return vec

	img_size = img_in.shape[2]
	
	img_in = img_in*intensity

	dx = (roll_col(img_in, 1) - roll_col(img_in, -1))
	dy = (roll_row(img_in, 1) - roll_row(img_in, -1))
	
	pixSize = size / img_in.shape[-1]
	dx /= 2 * pixSize
	dy /= 2 * pixSize

	l1_metric = torch.nn.L1Loss()

	xy_normal = torch.cat((dx, -dy), 1)
	# xy_normal = norm(xy_normal)

	# print("xy_normal: ", xy_normal.shape)

	gau_xynormal = filter(xy_normal)
	gt = torch.zeros_like(gau_xynormal)
	reg_loss = l1_metric(gau_xynormal, gt)

	img_out = torch.cat((gau_xynormal, torch.ones_like(dx)), 1)
	img_out = norm(img_out)
	
	return reg_loss, img_out


# def debug():
# 	l = torch.tensor([1.,0.,1]).cuda().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
# 	l = norm(l)
# 	v_dir = torch.zeros_like(l).to(l.device)

# 	v_dir[:,-1,:,:]=1
# 	print("l: ", l)
# 	print("v_dir: ", v_dir)


# 	cos = compute_cos(l, v_dir)
# 	print(cos)
# 	dist_l_sq = (4/cos)**2

# 	return dist_l_sq


if __name__ == '__main__':

	import argparse
	import os

	from PIL import Image
	import torchvision.transforms as transforms
	from torchvision import  utils

	parser = argparse.ArgumentParser()
	parser.add_argument("--out_path", type=str, help='output path') 
	parser.add_argument("--car_paint", type=bool, help='car paint or not') 
	parser.add_argument("--video", type=bool, help='car paint or not') 
	args = parser.parse_args()


	if not os.path.exists(args.out_path):
		os.makedirs(args.out_path)

	# args.in_path = 'D:/XilongZhou/Research/Research_2021S/Dataset/Stone/StoneDataset_1'
	# args.in_path = "D:/XilongZhou/Research/Research_2021S/EGSR2022/LoganProject/data/CGF23_rebuttal/Fea"
	args.in_path = 'D:/XilongZhou/Research/Research_2021S/ConStyleGAN2/output/Leather3_cir_stylereg/sample/520k_5/0'

	device = "cuda"
	# load feature maps
	allfiles=os.listdir(args.in_path)

	# rough_idx = 1

	#----------------------------------------------------------------------------

	def rand_light_sample(b,r=0.3):
		# this setting is for non-colocated cam and light
		# u_1 = np.abs(np.random.normal(0,r,(b,1))).clip(0,0.9)

		u_1 = np.random.uniform(0,0.2,(b,1))
		# u_1 = np.random.uniform(0,0.4,(b,1))
		u_2 = np.random.uniform(0,1,(b,1))

		theta = 2*np.pi*u_2

		r = np.sqrt(u_1)
		z = np.sqrt(1-r*r)
		x = r*np.cos(theta)
		y = r*np.sin(theta)

		light_pos = np.concatenate((x,y,z),axis=-1) * 4.

		return light_pos


	def rand_light_v(num, li_range=0.45):

	    # u_1 = np.abs(np.random.normal(0,0.2,(1))).clip(0,0.9)
	    u_1 = np.array([li_range])
	    u_2 = 0.0
	    light_list = []

	    # circle 1
	    crement = 1./num
	    for i in range(num):
	        theta = 2*np.pi*u_2

	        x = u_1*np.cos(theta)
	        y = u_1*np.sin(theta)

	        light_pos = np.concatenate((x,y,np.array([1])),axis=0) * 4.

	        light_list.append(light_pos)
	        u_2 += crement

	    return light_list


	# rand light dir # [(-1,1),(-1,1),1 ]
	def rand_light_dir_tmp(num, device, li_range=1.0):
		x = np.random.rand(num, 1)*li_range*2 - li_range
		y = np.random.rand(num, 1)*li_range*2 - li_range
		light_dir = np.concatenate((x,y,np.ones_like(y)), axis=-1)
		light_dir = torch.from_numpy(light_dir).float().to(device).unsqueeze(-1).unsqueeze(-1)
		return light_dir

	dir_li=False

	if dir_li:
		rand_light = rand_light_dir_tmp(1, device, li_range=0.0)
	else:
		rand_light = rand_light_sample(1,r=0)
		rand_light = torch.from_numpy(rand_light).unsqueeze(-1).unsqueeze(-1).to(device)

	# video light
	# if args.video:
	# 	light_v = rand_light_v(43)

	# torch.save( rand_light, "D:/XilongZhou/Research/Research_2021S/EGSR2022/LoganProject/data/SynData/10_light.pt")

	# rand_light = torch.load("D:/XilongZhou/Research/Research_2021S/EGSR2022/LoganProject/data/SynData/10_light.pt").to(device)
	print("rand_light: ", rand_light.shape)
	light, cam_pos, size = set_param(device)
	toTensor = transforms.ToTensor()

	for index, file in enumerate(allfiles):

		name = file.split('.')[0]

		if name!= "aspere_cliff_mossy_32":
			continue

		# print(index, rough_idx)
		path = os.path.join(args.in_path, file)
		pat_pil = Image.open(path)

		full_img = toTensor(pat_pil).cuda()

		c,h,w = full_img.shape

		if args.car_paint:
			# H = full_img[0:1,:,0:h]
			# color = full_img[:,:,h:2*h]
			# R1 = full_img[0:1,:,2*h:3*h]
			# R2 = full_img[0:1,:,2*h:3*h]*1.3
			# f = full_img[0:1,:,2*h:3*h]*0.+0.9

			color = torch.tensor([0.2, 0.1, 0.1]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(1,1,512,512).cuda()
			R1 = full_img[0:1,:,2*h:3*h].unsqueeze(0)
			R2 = full_img[0:1,:,2*h:3*h].unsqueeze(0)*1.3
			R2 = torch.roll(R2, shifts=(10, 10), dims=(-1, -1))

			S1 = torch.tensor([0.04, 0.04, 0.04]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(1,1,512,512).cuda()
			S2 = torch.tensor([0.2, 0.2, 0.8]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(1,1,512,512).cuda()

			feas = torch.cat([color, R1, S1, R2, S2], dim=1)
			print(feas.shape)
		else:
			H = full_img[0:1,:,0:h]
			D = full_img[:,:,h:2*h]
			R = full_img[0:1,:,2*h:3*h]
			feas = torch.cat([H, D, R], dim=0).unsqueeze(0)
			fake_N = height_to_normal(feas[:,0:1,:,:], size=size)
			feas = torch.cat((fake_N,feas[:,1:4,:,:],feas[:,4:5,:,:].repeat(1,3,1,1)),dim=1)

		tex_pos = getTexPos(feas.shape[2], size, device).unsqueeze(0)
		# print("light_pos: ", light_pos.shape)


		# rens = env_render2(feas, tex_pos, light_pos, isMetallic=False) #[0,1]


		for idx in range(rand_light.shape[0]):

			light_pos = rand_light[idx:idx+1,...]
			print("light_pos: ", light_pos.shape)

			if args.car_paint:
				rens_p = render_carpaint(feas, tex_pos, light, light_pos, isMetallic=False, cam_pos=None if not dir_li else cam_pos, dir_flag=dir_li ) #[0,1]

			else:
				rens_p = render(feas, tex_pos, light, light_pos, isMetallic=False, cam_pos=None if not dir_li else cam_pos, dir_flag=dir_li ) #[0,1]

			# rens = rens_p

			# save_path = os.path.join(args.out_path, str(index)+'_Ren.png')
			# utils.save_image( rens, save_path, nrow=1, normalize=False)
			save_path = os.path.join(args.out_path, f'{name}_{idx}.png')
			utils.save_image( rens_p, save_path, nrow=1, normalize=False)

