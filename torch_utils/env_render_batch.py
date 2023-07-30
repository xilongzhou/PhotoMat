from PIL import Image

from sys import argv
from numpy import *
import torch as th
from torch_utils.common_batch import *

import os

# device = 'cpu'
# device = 'cuda'




def schlick(cos, f0):
  return f0 + (1 - f0) * (1 - cos)**5


def sample_cosine(r1, r2, device):
    theta = 2 * pi * r1
    r = th.sqrt(r2)
    x = th.cos(theta) * r # [b,k,n,n]
    y = th.sin(theta) * r
    z = th.sqrt(1 - r2) # note: x^2 + y^2 = r^2 = r2
    final = vec(x, y, z).to(device) 
    return final



def sample_ggx_ndf(r1, r2, alpha):
    theta = th.atan(alpha * th.sqrt(r1) / (th.sqrt(1 - r1)+eps))
    phi = 2 * pi * r2
    ct, st = th.cos(theta), th.sin(theta)
    cp, sp = th.cos(phi), th.sin(phi)
    return vec(st*cp, st*sp, ct)


def smith_g(n_dot_v, n_dot_l, alpha):
    def g1(cos, alpha):
        c2 = cos**2
        a2 = alpha**2
        t2 = (1 - c2) / (c2+eps)
        tmp = th.sqrt(1 + a2 * t2)
        return 2.0 / (tmp + 1)
    return g1(n_dot_v, alpha) * g1(n_dot_l, alpha)


def get_envcolor(env_map, ray_dir):

    assert ray_dir.dim()==5, f"dimension is {ray_dir.dim()}, should be 5"

    b,k,_,n,_ = ray_dir.shape

    # theta = th.acos(getz(ray_dir))[:,:,0,:,:] # [b, k*k, w, h]
    # phi = th.atan2(gety(ray_dir), getx(ray_dir))[:,:,0,:,:]

    # temp_z = getz(ray_dir)[:,:,0,:,:].clamp(-1+eps, 1-eps)
    # theta = th.acos(temp_z) # [b, k*k, w, h]

    # temp_x = getx(ray_dir)[:,:,0,:,:]
    # temp_y = gety(ray_dir)[:,:,0,:,:]
    # phi = th.atan2(temp_y+eps, temp_x+eps)

    # # in [-1, 1]
    # x = (phi + pi) / (2 * pi) * 2 - 1
    # y = theta / (pi / 2) * 2 - 1


    x = ray_dir[:,:,0,:,:]
    y = ray_dir[:,:,1,:,:]

    xy = th.stack((x, y), dim=-1) # b x k*k x n x n x 2
    xy = xy.view(-1,n,n,2) # b*k*k x n x n x 2

    # xy.register_hook(lambda x: print('xy ',x.mean()))

    env_map = env_map.repeat(k,1,1,1) # b*k*k x n x n x 3
    # env_map.register_hook(lambda x: print('env_map ',x.mean()))

    # sampled_env_map = env_map[:,x,y,:].view(b, k, 3, n, n)
    sampled_env_map = grid_sample(env_map, xy).view(b, k, 3, n, n)  # b*k*k x 3 x n x n

    assert sampled_env_map.isnan().any()==False, f'nan in sampled_env_map'
    assert sampled_env_map.isinf().any()==False, f'inf in self.real_img'


    # print('sampled_env_map: ',sampled_env_map.shape)
    return sampled_env_map



# sample brdf, lookup envmap
def brdf_sample_batch(k, v, frame, albedo, normal, rough, env, device='cuda'):
    b,_,n,_ = albedo.shape

    inv_k = 1.0 / k

    grid_x, grid_y = th.meshgrid(th.arange(k, device=device),th.arange(k, device=device)) # k x k
    temp_grid = th.cat([grid_y.unsqueeze(0), grid_x.unsqueeze(0)], dim=0).reshape(2, k*k,1,1) # 2 x k*k x n x n

    # diffuse term
    rand_r = (th.rand(b, 2, k*k,n,n, device=device) + temp_grid) * inv_k # b x 2 x k*k x n x n
    l = frame.to_world(sample_cosine(rand_r[:,0,...], rand_r[:,1,...], device)) # b x k*k x 3 x n x n
    diff = get_envcolor(env, l)  # b x k*k x 3 x n x n
    diff = th.mean(diff, dim=1) # b x 3 x n x n

    # specular term
    rand_r = (th.rand(2, k*k,n,n, device=device) + temp_grid) * inv_k
    alpha = rough ** 2
    h_local = sample_ggx_ndf(rand_r[0,...], rand_r[1,...], alpha)
    h = frame.to_world(h_local)
    v = v.unsqueeze(0).repeat(b,k*k,1,1,1)
    normal = frame.normal.unsqueeze(1).repeat(1,k*k,1,1,1)
    l = normalize(-v + 2 * dot(v, h) * h) # reflect v around h

    # l.register_hook(lambda x: print('l2 ',x.mean()))


    n_dot_l = dot(normal, l).clamp(eps, 1)
    n_dot_v = dot(normal, v).clamp(eps, 1)
    n_dot_h = dot(normal, h).clamp(eps, 1)
    v_dot_h = dot(v, h).clamp(eps, 1)

    F = schlick(v_dot_h, 0.04)
    # with th.no_grad():
    m1 = th.logical_and(n_dot_v > eps, n_dot_h > eps)
    m2 = th.logical_and(n_dot_l > eps, v_dot_h > eps)
    mask = th.logical_and(m1, m2).to(th.float32)
    G = smith_g(n_dot_v, n_dot_l, alpha.unsqueeze(1).repeat(1,k*k,1,1,1))

    # F.register_hook(lambda x: print('F ',x.mean()))

    E = get_envcolor(env, l)
    # E.register_hook(lambda x: print('E ',x.mean()))

    spec = E * (mask * F * G * v_dot_h / (n_dot_v * n_dot_h+eps))    
    # spec = mask * F * G * v_dot_h / (n_dot_v * n_dot_h+eps)    
    spec = th.mean(spec, dim=1)#.repeat(1,3,1,1)
    # spec.register_hook(lambda x: print('spec ',x.mean()))

    
    final = diff * albedo + spec
    # final.register_hook(lambda x: print('final 1 ',x.mean()))
    final = final.clamp(eps, 1.0)**(1/2.2)
    # final.register_hook(lambda x: print('final 2 ',x.mean()))

    # return final, diff, spec
    tmp = diff * albedo
    return tmp.clamp(eps, 1.0)**(1/2.2), diff, spec
    # return spec, diff, spec


def env_render(k, camera_pos, surface_pos, albedo, normal, rough, env, device='cuda'):
    view_dir, _ = get_cam_dir(camera_pos, surface_pos)
    frame = Frame(normal, device)
    return brdf_sample_batch(k, view_dir, frame, albedo, normal, rough, env, device)


def main():
    with th.no_grad():
        # load envmap
        if argv[1].endswith('.exr'):
            env = exr.read(argv[1])
        else:
            env = asarray(Image.open(argv[1])).astype(float32) / 255
            env **= 2.2
        env = img2th(env).unsqueeze(0).to(device)

        # load maps
        fname = argv[2]
        img = Image.open(fname)
        img = asarray(img).astype(float32)
        img /= 255

        scale = float32(argv[3]) # height scaling
        size = float32(argv[4])
        cam_dist = float32(argv[5])
        k = int32(argv[6]) # use k^2 samples per pixel
        n = img.shape[0]

        bump = img[:, :n, 0] * scale
        albedo = img[:, n:2*n, :3]
        rough = img[:, 2*n:3*n, 0]

        # compute normal
        print('bump: ',bump.shape)
        normal1 = bump_to_normal(bump, scale, size)
        normal2 = height_to_normal(th.from_numpy(img[:, :n, 0]).unsqueeze(0).unsqueeze(0), size, intensity=scale)

        # render
        albedo = img2th(albedo).to(device).unsqueeze(0).repeat(2,1,1,1)
        rough = img2th(rough).to(device).unsqueeze(0).repeat(2,1,1,1)
        normal1 = img2th(normal1).to(device).unsqueeze(0).repeat(2,1,1,1)
        normal2 = img2th(normal2).to(device).repeat(2,1,1,1)
        camera_pos = th.tensor((0, 0, cam_dist), device=device)

        env = env.repeat(2,1,1,1)

        # texture pos
        surface_pos = getTexPos(albedo.shape[-1], size, device).unsqueeze(0)


        result1, diff1, spec1 = env_render(k, camera_pos, surface_pos, albedo, normal1, rough, env)
        result2, diff2, spec2 = env_render(k, camera_pos, surface_pos, albedo, normal2, rough, env)

        result1 = th2img(result1[0,...])
        diff1 = th2img(diff1[0,...])
        spec1 = th2img(spec1[0,...])
        normal1 = th2img(normal1[0,...])

        result2 = th2img(result2[0,...])
        diff2 = th2img(diff2[0,...])
        spec2 = th2img(spec2[0,...])
        normal2 = th2img(normal2[0,...])


        save_path = 'debug'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # exr.write16(result, os.path.join(save_path,'out.exr'))
        # exr.write16(diff, os.path.join(save_path,'diff.exr'))
        # exr.write16(spec, os.path.join(save_path,'spec.exr'))


        save_img(result1, os.path.join(save_path,'out1.png'))
        save_img(diff1, os.path.join(save_path,'diff1.png'))
        save_img(spec1, os.path.join(save_path,'spec1.png'))

        save_img(result2, os.path.join(save_path,'out2.png'))
        save_img(diff2, os.path.join(save_path,'diff2.png'))
        save_img(spec2, os.path.join(save_path,'spec2.png'))

        save_img(0.5*normal1+0.5, os.path.join(save_path,'normal1.png'), gamma=False)
        save_img(0.5*normal2+0.5, os.path.join(save_path,'normal2.png'), gamma=False)
# 

if __name__ == '__main__':
    main()
