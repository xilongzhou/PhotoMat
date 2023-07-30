# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import os
from time import perf_counter
import random

import click
import imageio
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import dnnlib
import legacy

from training.networks import Generator, MatUnet, weights_init

from torch_utils.render import set_param, getTexPos, render, height_to_normal, gaussian_reg, norm, render_carpaint, xy_to_normal
from torch_utils import misc

from training.loss import VGGLoss, TDLoss
from torch_utils.misc import tile_shift

#----------------------------------------------------------------------------

# this is for filtering out data
# def rand_light_fix(num=4, li_range=0.5):

#     u_1 = np.array([li_range])
#     u_2 = 1/8.
#     light_list = []

#     # circle 1
#     crement = 1./num
#     for i in range(num):
#         theta = 2*np.pi*u_2

#         x = u_1*np.cos(theta)
#         y = u_1*np.sin(theta)

#         light_pos = np.concatenate((x,y,np.array([1])),axis=0) * 4.

#         light_list.append(light_pos)
#         u_2 += crement

#     # center light
#     center = np.array([0,0,4])
#     light_list.append(center)

#     return light_list


#----------------------------------------------------------------------------

def project(
	G,
	mat,
	c,
	outdir,
	*,
	target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
	num_steps                  = 1000,
	w_avg_samples              = 10000,
	initial_learning_rate      = 0.1,
	initial_noise_factor       = 0.05,
	lr_rampdown_length         = 0.25,
	lr_rampup_length           = 0.05,
	noise_ramp_length          = 0.75,
	regularize_noise_weight    = 1e5,
	verbose                    = True,
	w_plus                     = True,
	shift                     = True,
	device: torch.device
):
	 
	# print(G.img_channels, G.img_resolution)    
	
	# assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

	def logprint(*args):
		if verbose:
			print(*args)


	# Compute w stats.
	logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
	z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
	w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
	w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
	w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
	w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

	# Setup noise inputs.
	noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

	# Features for target image.

	target_images = target.unsqueeze(0).to(device).to(torch.float32).permute(0,3,1,2)/255.
 
	if w_plus:
		w_avg = np.repeat(w_avg, G.mapping.num_ws, axis=1)
	print('w_avg: ',w_avg.shape)

	w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
	print('w_opt: ',w_opt.shape)

	optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

	# w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)

	# Init noise.
	for buf in noise_bufs.values():
		buf[:] = torch.randn_like(buf)
		buf.requires_grad = True

	light, _, size = set_param(device)
	tex_pos = getTexPos(256, size, device).unsqueeze(0)

	loss_TD = TDLoss(device, 2)
	loss_L1 = torch.nn.L1Loss()

	for step in range(num_steps+1):
		# Learning rate schedule.
		t = step / num_steps
		w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
		lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
		lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
		lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
		lr = initial_learning_rate * lr_ramp
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr

		# Synth images from opt_w.
		w_noise = torch.randn_like(w_opt) * w_noise_scale
		if not w_plus:
			ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
			# ws = w_opt.repeat([1, G.mapping.num_ws, 1])
		else:
			ws = w_opt + w_noise

		# forward passing
		_, fea = G.synthesis(ws, c, out_fea=True, noise_mode='const',test_mode=True, no_shift=True)
		maps = mat(fea) * 0.5 + 0.5 # (-1,1) --> (0,1)

		N = height_to_normal(maps[:,0:1,:,:], size=size)
		D = maps[:,1:4,:,:].clamp(min=0, max=1)
		R = maps[:,4:5,:,:].repeat(1,3,1,1).clamp(min=0.2, max=0.9)
		S = maps[:,5:8,:,:].clamp(min=0, max=1)
		ren_fea = torch.cat((N, D, R, S), dim=1)

		if shift:
			# if random.choice([True, False]):
			ren_fea = tile_shift(ren_fea, ren_fea.shape[-1], not_batch = True)

		rens = render(ren_fea, tex_pos, light, c, isMetallic=False, no_decay=False, cam_pos=None, dir_flag=False).float() #[0,1] [1,C,H,W]
		# rens = (rens.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)

		td_loss = loss_TD(rens, target_images)
		l1_loss = loss_L1(rens, target_images)*0.01

		# Noise regularization.
		reg_loss = 0.0
		# for v in noise_bufs.values():
		# 	noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
		# 	while True:
		# 		reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
		# 		reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
		# 		if noise.shape[2] <= 8:
		# 			break
		# 		noise = F.avg_pool2d(noise, kernel_size=2)

		loss = td_loss + l1_loss + reg_loss * regularize_noise_weight

		# Step
		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		optimizer.step()

		if step%500==0:
			logprint(f'step {step+1:>4d}/{num_steps}: td_loss {td_loss}, l1_loss {l1_loss} loss {float(loss)}')

			rens = (rens.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
			Image.fromarray(rens.cpu().numpy(), 'RGB').save(f'{outdir}/{step}_rens.png') 

			N = (N.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).squeeze(0)
			D = (D.permute(0, 2, 3, 1)**(1/2.2) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
			R = (R.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
			S = (S.permute(0, 2, 3, 1)**(1/2.2) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)

			fea_save = torch.cat([D,N,R,S], dim=1)
			Image.fromarray(fea_save.cpu().numpy(), 'RGB').save(os.path.join(outdir, "fea.png"))


		# Save projected W for each optimization step.
		# w_out[step] = w_opt.detach()[0]

		# Normalize noise.
		with torch.no_grad():
			for buf in noise_bufs.values():
				buf -= buf.mean()
				buf *= buf.square().mean().rsqrt()

	# if w_plus:
	#     print('w_out: ', w_out.shape)
	#     return w_out
	# else:
	#     return w_out.repeat([1, G.mapping.num_ws, 1])

#----------------------------------------------------------------------------

def set_args(

	network: str,
	matunet: str,
	out_nc: int,
	layer_n: int,
	shift: bool,

	):

	args = dnnlib.EasyDict()

	args.network = network
	args.matunet = matunet
	args.out_nc = out_nc
	args.layer_n = layer_n
	args.shift = shift

	return args


@click.command()
@click.option('--network', help='Network pickle filename', required=True)
@click.option('--matunet', help='matunet pickle filename', required=True)
@click.option('--seed', help='Random seed', type=int, default=303, show_default=True)
@click.option('--save_name', help='saved name', type=str, default='256_old', show_default=True)
@click.option('--out_nc', help='output channel of MatUnet: 5 | 8', type=int, default=8, metavar='INT')
@click.option('--layer_n', help='number of MatUnet', type=int, default=5, metavar='INT')
@click.option('--shift', help='shift', type=bool, required=False, metavar='BOOL', default=False, show_default=True)

def run_projection(
	seed: int,
	save_name: str,
	**config_kwargs
):
	"""Project given image to the latent space of pretrained network pickle.

	Examples:

	\b
	python projector.py --outdir=out --target=~/mytargetimg.png \\
		--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
	"""
	np.random.seed(seed)
	torch.manual_seed(seed)


	args = set_args(**config_kwargs)

	device = torch.device('cuda')


	cond_li = torch.tensor([0,0,4], device=device,dtype=torch.float32).unsqueeze(0)


	# cond_li = cam_pos
	tar_path = './data'
	for tar in os.listdir(tar_path):
	# while True:
		print(f"processing {tar}....")
		name = tar.split('.')[0]


		# load network
		print('Loading networks from "%s"...' % args.network)
		with dnnlib.util.open_url(args.network) as f:
			G_tmp = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

		init_kwargs_tmp = G_tmp.init_kwargs
		res = init_kwargs_tmp['img_resolution']
		circular = init_kwargs_tmp['synthesis_kwargs']['circular']
		print(f"res: {res}, circular: {circular}")

		try:
			if 'mlp_fea' not in init_kwargs_tmp:
				init_kwargs_tmp['mlp_fea']=32
			if 'mlp_hidden' not in init_kwargs_tmp:
				init_kwargs_tmp['mlp_hidden']=64
			init_kwargs_tmp["synthesis_kwargs"].pop('high_res', None)
		except KeyError:
			print("""""""""""""""""""")
			pass

		print("Reloading Modules!")
		G = Generator(*G_tmp.init_args, **init_kwargs_tmp).eval().requires_grad_(False).to(device)
		misc.copy_params_and_buffers(G_tmp, G, require_all=True)
		
		# set up material Unet
		net = MatUnet(out_c = args.out_nc, batch_norm=False, layer_n=args.layer_n).to(device)
		net.apply(weights_init)
		net.load_state_dict(torch.load(args.matunet)['MatUnet'])

		G_ema = copy.deepcopy(G).eval()
		net = copy.deepcopy(net).eval().requires_grad_(False).to(device) # type: ignore

		print("finish loading modules...")

		# load each image
		outdir = os.path.join(f'./projector/{save_name}/{name}')
		os.makedirs(outdir, exist_ok=True)

		target_fpath = os.path.join(tar_path,tar)
		target_pil = Image.open(target_fpath).convert('RGB')
		w, h = target_pil.size
		target_pil = target_pil.resize((G.img_resolution, G.img_resolution), Image.LANCZOS)
		target_pil.save(f'{outdir}/target.png') 
		target_uint8 = torch.from_numpy(np.array(target_pil, dtype=np.uint8)).to(device)


		# # Optimize projection.
		# start_time = perf_counter()
		project(
			G_ema,
			net,
			cond_li,
			outdir,
			target=target_uint8, # pylint: disable=not-callable
			device=device,
			w_plus=True,
			shift=args.shift,
		)
		# print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

		# # Render debug output: optional video and projected image and W vector.
		# os.makedirs(outdir, exist_ok=True)
		# if save_video:
		#     video = imageio.get_writer(f'{outdir}/{name}_proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
		#     print (f'Saving optimization progress video "{outdir}/proj.mp4"')
		#     for projected_w in projected_w_steps:
		#         synth_image = G.synthesis(projected_w.unsqueeze(0), cond_li,noise_mode='const')
		#         synth_image = (synth_image + 1) * (255/2)
		#         synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
		#         video.append_data(np.concatenate([target_uint8.permute(1,2,0).cpu().numpy(), synth_image], axis=1))
		#     video.close()

		# # Save final projected frame and W vector.
		# Image.fromarray(target_uint8.permute(1,2,0).cpu().numpy(), 'RGB').save(f'{outdir}/{name}_target.png')
		# projected_w = projected_w_steps[-1]
		# synth_image = G.synthesis(projected_w.unsqueeze(0), cond_li, noise_mode='const')
		# synth_image = (synth_image + 1) * (255/2)
		# synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
		# Image.fromarray(synth_image, 'RGB').save(f'{outdir}/{name}_proj.png')
		# np.savez(f'{outdir}/{name}_projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())

		# # test under different lighting
		# num=100
		# all_li = rand_light_video(num)
		# video = imageio.get_writer(f'{outdir}/{name}_relight.mp4', mode='I', fps=24, codec='libx264', bitrate='16M')
		# for frame_idx in range(num):
		#     cond_li = torch.from_numpy(all_li[frame_idx]).unsqueeze(0).to(device)
		#     img = G.synthesis(projected_w.unsqueeze(0), cond_li, noise_mode='const')
		#     img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).squeeze(0)
		#     img = img.cpu().numpy()
		#     video.append_data(img)
		# video.close()    

#----------------------------------------------------------------------------

if __name__ == "__main__":
	run_projection() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
