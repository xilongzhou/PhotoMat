

import os
import click
import re
import json
import tempfile
import torch
import dnnlib
import copy
import pickle
import torch.nn.functional as F

from training import training_loop
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops

import numpy as np

import legacy

from torch_utils.render import set_param, getTexPos, render, height_to_normal, gaussian_reg, norm, xy_to_normal, render_carpaint
from training.networks import Generator, MatUnet, weights_init
from torch_utils import misc

from training.loss import VGGLoss, TDLoss

import PIL.Image 

from torch_utils.misc import tile_shift

import torchvision.transforms as T

from typing import List, Optional, Tuple, Union

import imageio
from tqdm import tqdm

import warnings
import random
warnings.filterwarnings("ignore")

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size, res=None, car_paint=False):
    gw, gh = grid_size

    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)

    # this is for car paint
    if img.shape[1]==11:

        print(".....save to car paint....")
        # img = (img - lo) / (hi - lo) #[0,1]

        # save features
        color = img[:,0:3,:,:] #[0,1]
        r1 = np.repeat(img[:,3:4,:,:], 3, axis=1) #[0,1]
        s1 = img[:,4:7,:,:] #[0,1]
        r2 = np.repeat(img[:,7:8,:,:], 3, axis=1) #[0,1]
        s2 = img[:,8:11,:,:] #[0,1]

        fea = np.concatenate((color**(1/2.2), r1, s1**(1/2.2), r2, s2**(1/2.2) ), axis=-1)
        fea = (fea - lo) * (255 / (hi - lo)) #[-1,1]
        fea = np.rint(fea).clip(0, 255).astype(np.uint8)

        _N, C, H, W = fea.shape
        fea = fea.reshape(gh, gw, C, H, W)
        fea = fea.transpose(0, 3, 1, 4, 2)
        fea = fea.reshape(gh * H, gw * W, C)
        PIL.Image.fromarray(fea, 'RGB').save(fname)

    elif img.shape[1]==12:

        # print("img.shape: ",img.shape)
        # save features
        N = img[:,0:3,:,:] #[0,1]
        D = img[:,3:6,:,:] #[0,1]
        R = img[:,6:9,:,:] #[0,1]
        S = img[:,9:12,:,:] #[0,1]

        fea = np.concatenate((0.5*N+0.5, D**(1/2.2), R, S**(1/2.2)), axis=-1)
        fea = (fea - lo) * (255 / (hi - lo)) #[-1,1]
        fea = np.rint(fea).clip(0, 255).astype(np.uint8)

        _N, C, H, W = fea.shape
        fea = fea.reshape(gh, gw, C, H, W)
        # fea_t = np.tile(fea, (1, 1, 1, 2, 2))    

        fea = fea.transpose(0, 3, 1, 4, 2)
        fea = fea.reshape(gh * H, gw * W, C)
        # print("img 2.shape: ",img.shape)


        PIL.Image.fromarray(fea, 'RGB').save(fname)

        # fea_t = fea_t.transpose(0, 3, 1, 4, 2)
        # fea_t = fea_t.reshape(gh * H * 2, gw * W * 2, C)
        # PIL.Image.fromarray(fea_t, 'RGB').save(fname.split('.')[0]+'_tiled.png')

    elif img.shape[1]==3:
        # lo, hi = drange
        # img = np.asarray(img, dtype=np.float32)
        img = (img - lo) * (255 / (hi - lo)) #[-1,1]
        img = np.rint(img).clip(0, 255).astype(np.uint8) #[0,255]

        _N, C, H, W = img.shape
        img = img.reshape(gh, gw, C, H, W)
        img = img.transpose(0, 3, 1, 4, 2)
        img = img.reshape(gh * H, gw * W, C)

        assert C in [1, 3]
        if C == 1:
            PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
        if C == 3:
            PIL.Image.fromarray(img, 'RGB').save(fname)


# syn ----------------------------------------------------------------------------

# randlight for generating video
def rand_light_video(num,li_range=0.45):

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

# randlight for generating video
def rand_light_dir_video(num,li_range=0.45):

    # u_1 = np.abs(np.random.normal(0,0.2,(1))).clip(0,0.9)
    theta = 0

    light_list = []

    # circle 1
    crement = np.pi/num
    for i in range(num):

        x = np.cos(theta)
        z = np.sin(theta)

        light_pos = np.array([x,0,z])
        # print("light_pos: ", light_pos)
        light_list.append(light_pos)
        theta += crement

    return light_list

# rand point light pos
def rand_light(num, device, li_range=0.5):
    x = np.random.rand(num, 1)*li_range*2 - li_range
    y = np.random.rand(num, 1)*li_range*2 - li_range
    light_pos = np.concatenate((x,y,np.ones_like(y)), axis=-1) * 4.
    light_pos = torch.from_numpy(light_pos).float().to(device)
    return light_pos

# rand light dir # [(-1,1),(-1,1),1 ]
def rand_light_dir(num, device, li_range=1.0, fix_theta=False):
    if not fix_theta:
        x = np.random.rand(num, 1)*li_range*2 - li_range
        y = np.random.rand(num, 1)*li_range*2 - li_range
        light_dir = np.concatenate((x,y,np.ones_like(y)), axis=-1)
        light_dir = torch.from_numpy(light_dir).float().to(device).unsqueeze(-1).unsqueeze(-1)
        # print("light_dir ", light_dir.shape)

    else:
        phi = np.pi/4

        z = np.array([np.cos(phi)])
        z = z[:, np.newaxis]

        z = np.repeat(z, num, axis=0)

        theta = 2*np.pi*np.random.rand(num,1)

        x = z*np.cos(theta)
        y = z*np.sin(theta)

        light_dir = np.concatenate([x,y,z], axis=-1)
        light_dir = torch.from_numpy(light_dir).float().to(device).unsqueeze(-1).unsqueeze(-1)

    return norm(light_dir)

# this is for filtering out data
def rand_light_fix(num=4, li_range=0.5):

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

def compute_sim(img_dict):

    # 0-2, 1-3, 0-1, 2-3
    L2 = torch.nn.MSELoss()
    total = 0

    total = total + L2(img_dict[0], img_dict[2])
    total = total + L2(img_dict[1], img_dict[3])
    total = total + L2(img_dict[0], img_dict[1])
    total = total + L2(img_dict[2], img_dict[3])

    total /= 4

    return total


# --------------------------------------------------------------------------

def gaussian_fn(M, std, c, device):
    n = torch.arange(0, M, dtype=torch.float32) - (M - 1.0) / 2.0
    n = n.unsqueeze(0).repeat(c.shape[0],1).to(device)
    w = torch.exp(-(n-c) ** 2 / (2 * std * std))
    return w

def gkern(device, size=512, std=256, c=0):
    """Returns a 2D Gaussian kernel array."""
    gkern1d_x = gaussian_fn(size, std=std, c=c[0], device=device) 
    gkern1d_y = gaussian_fn(size, std=std, c=c[1], device=device) 
    gkern2d = torch.bmm(gkern1d_x.unsqueeze(2), gkern1d_y.unsqueeze(1)).unsqueeze(1)
    return gkern2d

def li_to_offset(li, size):
    li_y = size*li[:,0:1]/2.0
    li_x = -size*li[:,1:2]/2.0
    return (li_x, li_y)
    
#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(args.run_dir, 'log.txt'), file_mode='a', should_flush=True)
    print('Logger')

    # Init torch.distributed.
    if args.num_gpus > 1:
        print('init_file: ', args.num_gpus)
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    # training_loop.training_loop(rank=rank, **args)

    maps_trainer(rank=rank, args=args)

#----------------------------------------------------------------------------

class shift_invariant_loss(torch.nn.Module):
    def __init__(self):
        super(shift_invariant_loss, self).__init__()        
        
        print("setting up shift shift_invariant_loss")

        self.L1 = torch.nn.L1Loss()

    def forward(self, x, y):  # x is output, y is GT 

        # print("x: ", x.shape)
        # preprocess
        m_x = torch.mean(x, (1,2,3), keepdim=True)
        m_y = torch.mean(y, (1,2,3), keepdim=True)

        # print("m_x: ", m_x.shape)
        # print("m_y: ", m_y.shape)

        loss = self.L1(x/m_x, y/m_y)

        return loss

#----------------------------------------------------------------------------

def maps_trainer(rank, args):

    # loading network
    print('Loading networks from "%s"...' % args.network)
    device = torch.device('cuda', rank)
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

    if args.reload_modules:
        print("Reloading Modules!")
        G = Generator(*G_tmp.init_args, **init_kwargs_tmp).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G_tmp, G, require_all=True)

    G_ema = copy.deepcopy(G).eval()

    # set up material Unet
    net = MatUnet(out_c = args.out_nc, batch_norm=args.batch_norm, layer_n=args.layer_n).to(device)

    net.apply(weights_init)
    if args.matunet:
        print(f"resume from {args.matunet}")
        net.load_state_dict(torch.load(args.matunet)['MatUnet'])
    print(f"number of layer is {args.layer_n}")

    # optimizer
    opt_sigma = 0
    # if args.falloff:
    #     opt_sigma = torch.tensor(0.5, device=device, requires_grad=True, dtype=torch.float32)
    #     optimizer = torch.optim.Adam(list(net.parameters())+[opt_sigma], lr=args.lr)
    # else:
    #     optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    if args.falloff > 0:
        opt_sigma = args.falloff
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # loss
    loss_L1 = torch.nn.L1Loss() if not args.shift_invariant else shift_invariant_loss()
    loss_VGG = VGGLoss(device)
    loss_TD = TDLoss(device, 2)

    # render
    light, cam_pos, _ = set_param(device)
    size = args.size

    print(f"size of tex sample is {size}")
    tex_pos_temp = getTexPos(res, size, device).unsqueeze(0)
    # tex_pos_te2 = getTexPos(res, 2, device).unsqueeze(0)
    # tex_pos_te1 = getTexPos(res, 1, device).unsqueeze(0)
    tex_pos = tex_pos_temp.repeat(args.batch,1,1,1) 

    # for testing
    gd_h, gd_w = 5, 5
    grid_size = (gd_h, gd_w)
    tex_pos_te = tex_pos_temp


    # Initialize logs.
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        print('Initializing logs...')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(args.run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # training loop
    step = 0
    offset = None
    L1loss_value = 0.
    VGGloss_value = 0.
    TDloss_value =0.

    # add regularization
    if args.w_reg > 0:
        k = int(res/3)
        sigma = (k-1)/6
        Gaussianblur = T.GaussianBlur(kernel_size=(k, k), sigma=(sigma, sigma))


    # this is for 256 model
    if res==256:
        # old
        test_seeds = [305, 306, 310, 313,315, 316, 324, 333, 343, 348, 363, 398, 402, 403, 411, 422, 425, 426, 433, 434, 436, 439, 462, 475, 603]

        # new


    # this is for new 512 model
    if res==512:
        test_seeds = [0,2,5,7,12, 16, 34, 50, 51, 53, 54, 65, 71, 73, 132, 133, 213,311, 351, 453, 488, 515, 625, 629, 678]

    if res==1024:
        test_seeds = [45, 54,612,59,64,71,73,83, 86, 90, 99, 101, 102, 439, 166, 167,207, 225, 232, 250, 275, 338, 349, 426, 430]

    use_dir = True if args.light_type=="dir" else False
    no_shift = not circular 
    print(f"use_dir: {use_dir}, no_shift: {no_shift}, strategy {args.strategy}")
    all_li = rand_light_fix()

    while True:

        net.train()

        if args.light_type=="ptdir":
            use_dir = True if random.randint(0,1)==0 else False

        z = torch.from_numpy(np.random.randn(args.batch, G.z_dim)).to(device)

        if use_dir:
            cond_li = rand_light_dir(args.batch, device, li_range=args.li_range, fix_theta=args.fix_theta)
            cond_li = cond_li.repeat(1,1,res,res)
            # cond_li = rand_light(args.batch, device, li_range=args.li_range).unsqueeze(-1).unsqueeze(-1)
        else:
            cond_li = rand_light(args.batch, device, li_range=args.li_range)

        w = G.mapping(z, cond_li)
        img, fea = G.synthesis(w, cond_li, out_fea=True, test_mode=True, no_shift=no_shift) # noise_mode='const'
        # img, fea = G_ema(z=z, c=cond_li, out_fea=True, noise_mode='const', test_mode=True)

        maps = net(fea) * 0.5 + 0.5 # (-1,1) --> (0,1)
        img = img.float()*0.5 + 0.5
        img  = img.clamp(0,1) # clamp img but may not necessary

        if args.car_paint:
            rgb = maps[:,0:3,:,:].clamp(min=0, max=1)
            top_r = maps[:,3:4,:,:].clamp(min=0.2, max=0.9)
            top_spec = maps[:,4:7,:,:].clamp(min=0, max=1)
            bot_r = maps[:,7:8,:,:].clamp(min=0.2, max=0.9)
            bot_spec = maps[:,8:11,:,:].clamp(min=0, max=1)
            ren_fea = torch.cat((rgb, top_r, top_spec, bot_r, bot_spec), dim=1)

        else:

            # render maps
            if args.n_c==1:
                N = height_to_normal(maps[:,0:1,:,:], size=size)
                D = maps[:,1:4,:,:].clamp(min=0, max=1)
                R = maps[:,4:5,:,:].repeat(1,3,1,1).clamp(min=0.2, max=0.9)

                if args.out_nc==8:
                    # print('...............use_spec....................')
                    S = maps[:,5:8,:,:].clamp(min=0, max=1)
                    ren_fea = torch.cat((N, D, R, S), dim=1)
                else:
                    ren_fea = torch.cat((N, D, R), dim=1)

            elif args.n_c==2:
                N = xy_to_normal(maps[:,0:2,:,:])
                D = maps[:,2:5,:,:].clamp(min=0, max=1)
                R = maps[:,5:6,:,:].repeat(1,3,1,1).clamp(min=0.2, max=0.9)

                if args.out_nc==9:
                    # print('...............use_spec....................')
                    S = maps[:,6:9,:,:].clamp(min=0, max=1)
                    ren_fea = torch.cat((N, D, R, S), dim=1)
                else:
                    ren_fea = torch.cat((N, D, R), dim=1)

        # regular no interleave 
        if args.strategy==0:

            if args.force_shift:
                ren_fea = tile_shift(ren_fea, ren_fea.shape[-1], not_batch = True)

            if args.car_paint:
                rens = render_carpaint(ren_fea, tex_pos, light, cond_li, isMetallic=False, no_decay=False, cam_pos=None, dir_flag=use_dir).float() #[0,1] [1,C,H,W]
            else:
                rens = render(ren_fea, tex_pos, light, cond_li, isMetallic=False, no_decay=False, cam_pos=None, dir_flag=use_dir).float() #[0,1] [1,C,H,W]

            if args.falloff > 0:
                c = li_to_offset(cond_li, 0.5*res)
                falloff = gkern(size=res, std=opt_sigma*res, c=c, device=device)
                rens = rens*falloff

            # compute loss
            if args.force_shift:
                rens_d = F.interpolate(rens, size=(args.d_size,args.d_size), mode='bilinear', align_corners=True)                
                img_d = F.interpolate(img, size=(args.d_size,args.d_size), mode='bilinear', align_corners=True)   
                L1loss_value = loss_L1(rens_d, img_d)*args.w_dl1
            else:
                L1loss_value = loss_L1(rens, img)*args.w_dl1

            if args.w_vgg!=0:
                VGGloss_value = loss_VGG(rens, img)*args.w_vgg
            if args.w_td!=0:
                TDloss_value = loss_TD(rens, img)*args.w_td

            # add reg
            reg_loss = 0
            if args.w_reg > 0:
                reg_loss, _ = gaussian_reg(maps[:,0:1,:,:], size=size, filter=Gaussianblur)
                reg_loss = reg_loss*args.w_reg

            full_loss = L1loss_value + VGGloss_value + TDloss_value + reg_loss

        # shifting TD loss + non-shifting L1 full loss
        elif args.strategy==1:

            # print("shifting TD loss + non-shifting L1 full loss")

            rens = render(ren_fea, tex_pos, light, cond_li, isMetallic=False, no_decay=False, cam_pos=None, dir_flag=use_dir).float() #[0,1] [1,C,H,W]
            L1loss_value = loss_L1(rens, img)*args.w_dl1

            ren_fea = tile_shift(ren_fea, ren_fea.shape[-1], not_batch = True)
            rens = render(ren_fea, tex_pos, light, cond_li, isMetallic=False, no_decay=False, cam_pos=None, dir_flag=use_dir).float() #[0,1] [1,C,H,W]
            TDloss_value = loss_TD(rens, img)*args.w_td

            full_loss = L1loss_value + TDloss_value 

        # shifting downsampled L1 loss + non-shifting TD loss
        elif args.strategy==2:

            # print("shifting downsampled L1 loss + non-shifting TD loss")

            rens = render(ren_fea, tex_pos, light, cond_li, isMetallic=False, no_decay=False, cam_pos=None, dir_flag=use_dir).float() #[0,1] [1,C,H,W]
            TDloss_value = loss_TD(rens, img)*args.w_td

            ren_fea = tile_shift(ren_fea, ren_fea.shape[-1], not_batch = True)
            rens = render(ren_fea, tex_pos, light, cond_li, isMetallic=False, no_decay=False, cam_pos=None, dir_flag=use_dir).float() #[0,1] [1,C,H,W]
            rens_d = F.interpolate(rens, size=(args.d_size,args.d_size), mode='bilinear', align_corners=True)                
            img_d = F.interpolate(img, size=(args.d_size,args.d_size), mode='bilinear', align_corners=True)   
            L1loss_value = loss_L1(rens_d, img_d)*args.w_dl1

            full_loss = L1loss_value + TDloss_value 

        optimizer.zero_grad()
        full_loss.backward()
        optimizer.step()

        # saving images
        if rank==0 and step%args.log_freq==0:

            val_l1 = 0
            val_td = 0
            val_reg = 0

            with torch.no_grad():

                net.eval()

                allimg_te = []
                allrens_te = []
                # allrens_te1 = []
                if args.w_reg > 0:
                    allfea_gaussN_te = []

                allren_fea_te = []
                for tmp_batch in range(gd_h*gd_w):

                    z_te = torch.from_numpy(np.random.RandomState(test_seeds[tmp_batch]).randn(1, G_ema.z_dim)).to(device)
                    if use_dir:
                        cond_li_te = rand_light_dir(1, device, li_range=args.li_range, fix_theta=args.fix_theta)
                        cond_li_te = cond_li_te.repeat(1,1,res,res)
                    else:
                        cond_li_te = rand_light(1, device, li_range=args.li_range)
                    img_te, fea_te = G_ema(z=z_te, c=cond_li_te, out_fea=True, test_mode=True, no_shift = no_shift)

                    maps_te = net(fea_te) * 0.5 + 0.5 # (-1,1) --> (0,1)
                    img_te = img_te * 0.5 + 0.5 # (0,1)

                    if args.car_paint:
                        rgb_te = maps_te[:,0:3,:,:].clamp(min=0, max=1)
                        top_r_te = maps_te[:,3:4,:,:].clamp(min=0.2, max=0.9)
                        top_spec_te = maps_te[:,4:7,:,:].clamp(min=0, max=1)
                        bot_r_te = maps_te[:,7:8,:,:].clamp(min=0.2, max=0.9)
                        bot_spec_te = maps_te[:,8:11,:,:].clamp(min=0, max=1)
                        ren_fea_te = torch.cat((rgb_te, top_r_te, top_spec_te, bot_r_te, bot_spec_te), dim=1)
                    else:
                        # render maps
                        if args.n_c==1:
                            N = height_to_normal(maps_te[:,0:1,:,:], size=size)
                            D = maps_te[:,1:4,:,:].clamp(min=0, max=1)
                            R = maps_te[:,4:5,:,:].repeat(1,3,1,1).clamp(min=0.2, max=0.9)
                            if args.out_nc==8:
                                S = maps_te[:,5:8,:,:].clamp(min=0, max=1)
                                ren_fea_te = torch.cat((N, D, R, S), dim=1)
                                # del S
                            else:
                                ren_fea_te = torch.cat((N, D, R), dim=1)

                        elif args.n_c==2:
                            N = xy_to_normal(maps_te[:,0:2,:,:])
                            D = maps_te[:,2:5,:,:].clamp(min=0, max=1)
                            R = maps_te[:,5:6,:,:].repeat(1,3,1,1).clamp(min=0.2, max=0.9)
                            if args.out_nc==9:
                                # print('...............use_spec....................')
                                S = maps_te[:,6:9,:,:].clamp(min=0, max=1)
                                ren_fea_te = torch.cat((N, D, R, S), dim=1)
                            else:
                                ren_fea_te = torch.cat((N, D, R), dim=1)

                    if args.force_shift:
                        ren_fea_te = tile_shift(ren_fea_te, ren_fea_te.shape[-1], not_batch = True)

                    if args.car_paint:
                        rens_te = render_carpaint(ren_fea_te, tex_pos_te, light, cond_li_te, isMetallic=False, no_decay=False, cam_pos=None, dir_flag=use_dir).float() #[0,1] [1,C,H,W]
                    else:
                        rens_te = render(ren_fea_te, tex_pos_te, light, cond_li_te, isMetallic=False, no_decay=False, cam_pos=None, dir_flag=use_dir).float() #[0,1] [1,C,H,W]
                    # rens_te2 = render(ren_fea_te, tex_pos_te2, light, cond_li_te, isMetallic=False, no_decay=False, cam_pos=None, dir_flag=use_dir).float() #[0,1] [1,C,H,W]
                    # rens_te1 = render(ren_fea_te, tex_pos_te1, light, cond_li_te, isMetallic=False, no_decay=False, cam_pos=None, dir_flag=use_dir).float() #[0,1] [1,C,H,W]

                    if args.falloff > 0:
                        c_te = li_to_offset(cond_li_te, 0.5*res)
                        falloff_te = gkern(size=res, std=opt_sigma*res, c=c_te, device=device)
                        rens_te = rens_te*falloff_te

                    if args.w_reg > 0:
                        _, N_gaussian = gaussian_reg(maps_te[:,0:1,:,:], size=size, filter=Gaussianblur)
                        ren_fea_gaussN_te = torch.cat((N_gaussian, D, R, S), dim=1)
                        allfea_gaussN_te.append(ren_fea_gaussN_te)


                    l1_tmp = loss_L1(rens_te, img_te)*args.w_dl1
                    td_tmp = loss_TD(rens_te, img_te)*args.w_td

                    val_l1 += l1_tmp.data
                    val_td += td_tmp.data

                    allimg_te.append(img_te)
                    allrens_te.append(rens_te)
                    # allrens_te1.append(rens_te1)
                    allren_fea_te.append(ren_fea_te)

                val_l1 /= gd_h*gd_w
                val_td /= gd_h*gd_w

                allimg_te = torch.cat(allimg_te)
                allrens_te = torch.cat(allrens_te)
                # allrens_te2 = torch.cat(allrens_te2)
                allren_fea_te = torch.cat(allren_fea_te)

                save_image_grid(allimg_te.cpu().numpy(), os.path.join(args.run_dir, f'{step}_GT.png'), drange=[0,1], grid_size=grid_size)
                save_image_grid(allrens_te.cpu().numpy(), os.path.join(args.run_dir, f'{step}_ren.png'), drange=[0,1], grid_size=grid_size)
                # save_image_grid(allrens_te2.cpu().numpy(), os.path.join(args.run_dir, f'{step}_ren2.png'), drange=[0,1], grid_size=grid_size)
                # save_image_grid(allrens_te1.cpu().numpy(), os.path.join(args.run_dir, f'{step}_ren1.png'), drange=[0,1], grid_size=grid_size)
                save_image_grid(allren_fea_te.cpu().numpy(), os.path.join(args.run_dir, f'{step}_maps.png'), drange=[0,1], grid_size=grid_size)

                if args.w_reg > 0:
                    allfea_gaussN_te = torch.cat(allfea_gaussN_te)
                    save_image_grid(allfea_gaussN_te.cpu().numpy(), os.path.join(args.run_dir, f'{step}_gaussmaps.png'), drange=[0,1], grid_size=grid_size)


                # vis tileability
                # if circular:
                # fea_te_t = np.tile(ren_fea_te.cpu().numpy(), (1,1,2,2))
                # save_image_grid(fea_te_t, os.path.join(args.run_dir, f'{step}_maps_tile.png'), drange=[0,1], grid_size=grid_size)

                # fea_te = np.tile(fea_te[:,0:3,...].cpu().numpy(), (1,1,2,2))
                # save_image_grid(fea_te, os.path.join(args.run_dir, f'{step}_fea_tile.png'), drange=[0,1], grid_size=grid_size)


                del z_te
                del cond_li_te
                del img_te
                del fea_te
                # del N, D, R
                del maps_te
                del ren_fea_te
                del rens_te



            print(f"step: {step}, loss: {val_l1+val_td+val_reg}, L1: {val_l1:.4f}, TD: {val_td:.4f}, reg: {val_reg:.4f}")

            # recording loss
            if stats_tfevents is not None:
                stats_tfevents.add_scalar('TD:', TDloss_value, global_step=step)
                stats_tfevents.add_scalar('dL1: ', L1loss_value, global_step=step)
                stats_tfevents.flush()


        # Save network.
        # if step%args.save_ckpt_freq==0:
        if step%args.log_freq==0:
            torch.save({
                'MatUnet':net.state_dict(),
                # 'optimizer': optimizer.state_dict(),
                }, os.path.join(args.run_dir,f'MatUnet_{step}.pt'))
            print("saving done!!")

        step += 1

#----------------------------------------------------------------------------

def maps_opt(args, gt_img, name):

    # some settings
    num_steps                  = 2000
    initial_noise_factor       = 0.05
    lr_rampdown_length         = 0.25
    lr_rampup_length           = 0.05
    noise_ramp_length          = 0.75
    regularize_noise_weight    = 1e-2
    initial_learning_rate   = 0.1


    # loading network
    print('Loading networks from "%s"...' % args.network)
    device = torch.device('cuda')
    with dnnlib.util.open_url(args.network) as f:
        G_tmp = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    if args.reload_modules:
        print("Reloading Modules!")
        G = Generator(*G_tmp.init_args, **G_tmp.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G_tmp, G, require_all=True)
    
    G_ema = copy.deepcopy(G).eval()

    # set up material Unet
    net = MatUnet(out_c = args.out_nc, batch_norm=args.batch_norm).to(device)
    net.load_state_dict(torch.load(args.matunet)['MatUnet'])
    print("Finish loading")

    gt_img = gt_img.to(device).float()

    h = 256 * args.args.superres_scale


    light, cam_pos, size = set_param(device)
    tex_pos = getTexPos(h, size, device).unsqueeze(0)
    c = rand_light(1, device=device, li_range=0.0)

    # Compute w stats.
    w_avg_samples = 50000
    print(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.randn(w_avg_samples, G_ema.z_dim)
    w_samples = G_ema.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G_ema.synthesis.named_buffers() if 'noise_const' in name }

    w_avg = np.repeat(w_avg, G.mapping.num_ws, axis=1)
    print('w_avg: ',w_avg.shape)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    print('w_opt: ',w_opt.shape)


    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # loss
    loss_L1 = torch.nn.L1Loss()
    loss_VGG = VGGLoss(device)
    loss_TD = TDLoss(device, 2)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True


    # sample GT
    # z = torch.from_numpy(np.random.randn(1, G_ema.z_dim)).to(device)    
    # ws = G_ema.mapping(z, c)
    # gt_img = G_ema.synthesis(ws, c)*0.5 + 0.5
    # gt_img_save = (gt_img.permute(0,2,3,1) * 255).clamp(0, 255).to(torch.uint8)
    # print(gt_img_save.shape)
    # PIL.Image.fromarray(gt_img_save[0].cpu().numpy(), 'RGB').save(f'{args.outdir}/{name}_gt.png')


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
        ws = w_opt + w_noise

        img_out1, fea = G_ema.synthesis(ws, c, out_fea=True, noise_mode='const', test_mode=True, no_shift=True)
        img_out1 = img_out1*0.5 + 0.5

        if args.shift:
            fea = tile_shift(fea, h, not_batch = False)


        maps = net(fea) * 0.5 + 0.5 # (-1,1) --> (0,1)
        N = height_to_normal(maps[:,0:1,:,:], size=size)
        D = maps[:,1:4,:,:].clamp(min=0, max=1)
        R = maps[:,4:5,:,:].repeat(1,3,1,1).clamp(min=0.2, max=0.9)

        if args.out_nc==8:
            S = maps[:,5:8,:,:].clamp(min=0, max=1)
            ren_fea = torch.cat((N, D, R, S), dim=1)
        else:
            ren_fea = torch.cat((N, D, R), dim=1)

        img = render(ren_fea, tex_pos, light, c, isMetallic=False, no_decay=False, cam_pos=None, dir_flag=False).float() #[0,1] [1,C,H,W]

        if not args.use_mlp:
            img_out = img
        else:
            img_out = img_out1

        loss = 0
        if args.w_vgg>0:
            loss += loss_VGG(img_out, gt_img)*args.w_vgg

        if args.w_dl1>0:
            if img_out.shape[-1]!=gt_img.shape[-1]:
                img_out = F.interpolate(img_out, size=(256,256), mode='bilinear', align_corners=True)                
            loss += loss_L1(img_out, gt_img)*args.w_dl1

        if args.w_td>0:
            loss += loss_TD(img_out, gt_img)*args.w_td

        # Noise regularization.
        if regularize_noise_weight>0:
            reg_loss = 0.0
            for v in noise_bufs.values():
                noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                    reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)

            loss += reg_loss * regularize_noise_weight


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        if step%1000==0:
            print("loss: ", loss, "reg loss: ", reg_loss * regularize_noise_weight)
            save_image_grid(ren_fea.detach().cpu().numpy(), os.path.join(args.outdir, f'{name}_maps_step{step}.png'), drange=[0,1], grid_size=(1,1))

            img = (img.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{args.outdir}/{name}_step{step}.png')

            img_out1 = (img_out1.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img_out1[0].cpu().numpy(), 'RGB').save(f'{args.outdir}/{name}_step{step}_mlp.png')

    # render
    light, _, size = set_param(device)
    tex_pos = getTexPos(h, size, device).unsqueeze(0)

    net.eval()
    time = 3
    fps = 24
    num = time*fps
    all_li = rand_light_video(num)
 

    # Render video for sampled results
    # ----------------------------------------------------------------------------------------
    mp4 = os.path.join(args.outdir, f'{name}_sampled.mp4') 
    video_out = imageio.get_writer(mp4, mode='I', fps=fps, codec='libx264', bitrate='10M')
    for frame_idx in tqdm(range(num)):
        cond_li = torch.from_numpy(all_li[frame_idx]).unsqueeze(0).to(device)

        img, fea = G_ema.synthesis(ws, cond_li, out_fea=True, noise_mode='const', test_mode=True, no_shift=True)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).squeeze(0)
        img = img.cpu().numpy()
        video_out.append_data(img)
    video_out.close()
    # ----------------------------------------------------------------------------------------

    # maps = net(fea) * 0.5 + 0.5 # (-1,1) --> (0,1)
    # N = height_to_normal(maps[:,0:1,:,:], size=size)
    # D = maps[:,1:4,:,:].clamp(min=0, max=1)
    # R = maps[:,4:5,:,:].repeat(1,3,1,1).clamp(min=0.2, max=0.9)

    # if args.out_nc==8:
    #     S = maps[:,5:8,:,:].clamp(min=0, max=1)
    #     ren_fea = torch.cat((N, D, R, S), dim=1)
    # else:
    #     ren_fea = torch.cat((N, D, R), dim=1)

    # save_image_grid(ren_fea.cpu().numpy(), os.path.join(args.outdir, f'{seed}_maps.png'), drange=[0,1], grid_size=(1,1))


    mp4 = os.path.join(args.outdir, f'{name}_rendered.mp4') 
    video_out = imageio.get_writer(mp4, mode='I', fps=fps, codec='libx264', bitrate='10M')
    for frame_idx in tqdm(range(num)):
        cond_li = torch.from_numpy(all_li[frame_idx]).unsqueeze(0).to(device)        
        rens = render(ren_fea, tex_pos, light, cond_li, isMetallic=False, no_decay=False, cam_pos=None, dir_flag=False).float() #[0,1] [1,C,H,W]
        rens = (rens.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
        rens = rens.cpu().numpy()
        video_out.append_data(rens)
    video_out.close()

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


def set_args(

    # general setting
    network: str,
    gpus: int,
    trunc: float,
    trunc_cutoff: int,
    reload_modules: bool,

    # training setting
    batch: int,
    lr: float,
    save_ckpt_freq: int,
    log_freq: int,
    out_nc: int,
    w_vgg: float,
    w_td: float,
    w_dl1: float,
    w_reg: float,
    falloff: float,
    li_range: float,
    light_type: str,
    shift: bool,
    use_mlp: bool,
    force_shift: bool,
    shift_invariant: bool,
    fix_theta: bool,
    size: float,
    d_size: int,
    strategy: int,
    layer_n: int,
    n_c: int,
    filter_data: bool,

    # MatUnet 
    batch_norm: bool,
    car_paint: bool,

    # for testing,
    matunet: str,
    dataset: str,


    ):

    args = dnnlib.EasyDict()


    args.network = network
    args.trunc = trunc
    args.trunc_cutoff = trunc_cutoff
    args.reload_modules = reload_modules
    args.num_gpus = gpus

    args.save_ckpt_freq = save_ckpt_freq
    args.log_freq = log_freq
    args.batch = batch
    args.lr = lr

    args.w_vgg = w_vgg
    args.w_td = w_td
    args.w_reg = w_reg
    args.w_dl1 = w_dl1
    args.out_nc = out_nc
    args.falloff = falloff
    args.shift = shift
    args.use_mlp = use_mlp
    args.batch_norm = batch_norm
    args.li_range = li_range
    args.force_shift = force_shift
    args.shift_invariant = shift_invariant
    args.fix_theta = fix_theta
    args.size = size
    args.d_size = d_size
    args.filter_data = filter_data
    args.strategy = strategy
    args.layer_n = layer_n
    args.n_c = n_c
    args.car_paint = car_paint

    # for testing
    args.matunet = matunet
    args.light_type = light_type
    args.dataset = dataset

    if args.shift_invariant:
        args.d_size = 256


    if args.n_c==2:
        args.out_nc=9

    if args.car_paint:
        args.out_nc=11

    return args

#----------------------------------------------------------------------------
@click.command()
@click.option('--network', help='Network pickle filename', required=True)
@click.option('--gpus', help='Number of GPUs to use [default: 1]', type=int, metavar='INT')
@click.option('--trunc', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--outdir', help='Output directory', type=str, required=True, metavar='DIR')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=True, show_default=True)

@click.option('--batch', help=' batch size per gpu', type=int, default=4, metavar='INT')
@click.option('--lr', type=float, help='learning rate', default=1e-4, show_default=True)
@click.option('--log_freq', help='save images every # steps', type=int, default=5000, metavar='INT')
@click.option('--save_ckpt_freq', help=' save ckpt every # steps', type=int, default=20000, metavar='INT')
@click.option('--w_vgg', type=float, help='weight of vgg', default=0.0, show_default=True)
@click.option('--w_td', type=float, help='weight of TD', default=0.0, show_default=True)
@click.option('--w_dl1', type=float, help='weight of downsampled L1', default=0.0, show_default=True)
@click.option('--w_reg', type=float, help='weight of regularization of normal', default=0.0, show_default=True)
@click.option('--out_nc', help='output channel of MatUnet: 5 | 8 | 9', type=int, default=5, metavar='INT')
@click.option('--li_range', type=float, help='range of light', default=0.5, show_default=True)

@click.option('--use_mlp', help='use MLP', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--falloff', help='add falloff', type=float, required=False, default=False, show_default=True)
@click.option('--batch_norm', help='use batch normalizatin in the Unet', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--force_shift', help='force to shift feature during training', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--size', help=' size of sample', type=float, required=False, default=4, show_default=True)
@click.option('--d_size', help='downsampled size', type=int, default=16, metavar='INT')
@click.option('--shift_invariant', help='use shift_invariant loss', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--fix_theta', help='fix theta', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--filter_data', help='filter data or not', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--strategy', help='training strategy: 0: regular | 1: shifting TD loss + non-shifting L1 full loss | 2: shifting downsampled L1 loss + non-shifting TD loss ', type=int, default=0, metavar='INT')
@click.option('--layer_n', help='num of layer for matUnet', type=int, default=5, metavar='INT')
@click.option('--n_c', help='number of channel for normal maps: 1 | 2', type=int, default=1, metavar='INT')
@click.option('--car_paint', help='use car paint model', type=bool, required=False, metavar='BOOL', default=False, show_default=True)

# ------------- for testing---------------------
@click.option('--opt_mode', help='this is optimization mode', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--shift', help='tile shift or not', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--matunet', help='Output directory', type=str, required=False, metavar='DIR')
@click.option('--seeds', type=num_range, help='List of random seeds', required=False)
@click.option('--light_type', help='using directional light for all texel or point light or mixing: pt | dir | ptdir', type=str, default="pt", required=False)
@click.option('--dataset', default='../Dataset/TestData2', type=str, required=False)

def main(outdir, opt_mode, seeds, **config_kwargs):

    args = set_args(**config_kwargs)

    if opt_mode:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        args.outdir = outdir

        # loop over test data

        data_path = args.dataset
        for img in os.listdir(data_path):
            name = img.split(".png")[0]

            test_img = PIL.Image.open(os.path.join(data_path, img)).convert('RGB')

            if test_img.size[-1] > 256 and args.superres_scale == 1:
                test_img = test_img.resize((256, 256), resample=PIL.Image.LANCZOS)

            # if args.use_syn:
            test_img.save(f'{args.outdir}/{name}_gt.png')

            test_img = np.array(test_img)/255.
            test_img = torch.from_numpy(test_img).permute(2,0,1).unsqueeze(0)

            maps_opt(args, test_img, name)

    else:

        dnnlib.util.Logger(should_flush=True)

        # Pick output directory.
        prev_run_dirs = []
        if os.path.isdir(outdir):
            prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}')
        assert not os.path.exists(args.run_dir)

        # Create output directory.
        print('Creating output directory...')
        os.makedirs(args.run_dir)
        with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(args, f, indent=2)

        # Launch processes.
        print('Launching processes...')
        torch.multiprocessing.set_start_method('spawn')
        with tempfile.TemporaryDirectory() as temp_dir:
            if args.num_gpus == 1:
                subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
            else:
                torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
