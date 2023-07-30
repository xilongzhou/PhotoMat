# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Training Generative Adversarial Networks with Limited Data"."""

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

from torch_utils.render import set_param, getTexPos, render, height_to_normal, gaussian_reg, norm, render_carpaint, xy_to_normal
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

def save_image_grid(img, fname, drange, grid_size, res=None):
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
def rand_light_filter(num=4, li_range=0.5):

    u_1 = np.array([li_range])
    u_2 = 0.
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

# this is for filtering out data
def rand_light_fix(num=4, li_range=0.5):

    u_1 = np.array([li_range])
    u_2 = 1/8.
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

    # center light
    center = np.array([0,0,4])
    light_list.append(center)

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

def tile_maps(input):
    tmp_row = torch.cat([input, input], dim=-1)
    full = torch.cat([tmp_row, tmp_row], dim=-2)

    return full

#----------------------------------------------------------------------------

def maps_test(args, seeds):

    """
    here are seeds selected
    """

    # old 256: 48k+155k matunet
    old_256_list = [1,2,4,5,7,11,12,13,15,16,18,19,20,21,23,26,31,33,36,40,42,44,47,49,50,51,52,56,57,59,60,67,69,71,78,80,85,92,93,\
    95,96,98,100,102,103,104,107,109,112,116,117,119,128,129,130,131,133,139,141,143,146,154,156,164,165,168,167,170,184,185,189,187,\
    190,193,196,197,200,205,210,214,216, 218, 219, 220, 228,230,235, 243,245,248,256,257,259, 260,267, 270, 273, 277, 280, 281,290,293,\
    298,305,308,310,313,315,316,319,321,324,327,328,329,333,334,339,342,345,350,354,352,363,365,369,370,372,373,374,378,391,395,398,400,401,\
    402, 405,411,419,422,424,425,426,428,433,434,435,439,444,449,450,451,454,461,462,464,467,471,475,476,479,480,482, 484,489,493,494,495,496,499,\
    502, 504,505,508,511,512,522,530,531,532,543,545,549,544,558,560,561,562,581,587,588,591,593,601,604,605,606,609,610,612,613,614,618,620,621,\
    623,631,644,645,647,648,650,654,659,660,684,690,695,720,723,727,731,736,745,747,756,763,764,766,770,780,781,782,789,794,795,854,848,844,838,\
    830, 815,827,950,951,952,957,962,963,968,973,974,978,986,989,990,1002,1015,1025,1032,1031,1034,1038,1039,1041,1055,1070,1074,1075,1077,1079,\
    1083,1096,1098,1099,1105,1112,1115,1117,1121,1122,1133,1132,1135,1138,1154,1157,1158,1166,1175,1178,1193,1197,1200,1207,1216,1218,1222,1224,1235,\
    1240, 1241,1259, 1258, 1260, 1269,1279,1284,1297,1299,1347,1353,1354, 1368
    ]


    new_512_list = [1,5,7,8,10,11,12,14,17,19,22,30,33,34,37,39,45,47,50,51,53,54,56,64,65, 66, 70,71,75,77,80,81,86,87,89,93,96,97,98,101,\
    105,110,112,114,118,119,124,131,132,133,134,135,136,140,143,144,156,157,160,170,172,174,175,178,179,192,194,195,196,198,202,203,209,213,\
    212,216,218,221,222,223,236,238,242,258,274,275,284,286,290,299,303,305,308,309,311,314,317,319,323,333,326,337,339,349,351,362,363,366,\
    371,372,376,378, 384, 385,386,388,389,390,393,397,402,403,405,406,408,409,416,417,424,421,425,428,431,439,443,444,445,453,454,459,463,472,\
    476,492,495,497,500,502,503,506,515,520,528,537,543,544,546,552,554,566,567,585,586,587,589,591,602,604,615,616,619,621,625,640,647,658,680,\
    701,703,706,707,708,726,729,730,734,737,758,784,794,797,799,800,806,812,814,828,832,838,842,845,852,866,873,884,896,898,912,925,933,935,937,\
    940,941,944,945,949,1064,1051,981,987,990,1017,1034,1038,1050
    ]

    carpaint_list=[0,1,2,3,4,9,10,11,14,15,17,19,23,26,31,32,34,37,39,41,42,43,44,45,50,54,56,60,61,63,68,69,78,85,87,99,104,111,112,113,121,133,\
    136,139,155,159,162,163,179,184,186,188,190,191]

    new_1k_list=[1,3,9,10,11,12,14,16,17,19,23,26,30,32,36,38,40,41,42,43,45,54,61,68,70,84,96,97,101,102,121,133,138,143,147,148,158,\
    169,171,177,180,181,184,187,200,201,207,211,222, 225,226,229,232,250,252,261,267,268,272,274, 297,321,354,355,362,366,369,370,381,\
    392,395,426,428,441,445,485,499,500, 498]

    old_256_cirlist=[9,10,12,16,27,53,61,77,83,86,101,102,106,108,116,118,186,348]



    with torch.no_grad():
        # loading network
        print('Loading networks from "%s"...' % args.network)
        device = torch.device('cuda')
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
        net.load_state_dict(torch.load(args.matunet)['MatUnet'])

        # render
        light, _, size = set_param(device)

        tex_pos = getTexPos(res, size, device).unsqueeze(0)
        if args.tile:
            tex_pos_t = getTexPos(res*2, size, device).unsqueeze(0)

        net.eval()
        time = 3
        fps = 18
        num = time*fps
        all_li_v = rand_light_video(num) # for video
        all_li_f = rand_light_fix() # for 5 image
        all_li_filter = rand_light_filter() # for 5 image

        # this is for 256 model
        if args.eval_seed:
            if res==256:
                # old
                seeds = [305, 306, 310, 313,315, 316, 324, 333, 343, 348, 363, 398, 402, 403, 411, 422, 425, 426, 433, 434, 436, 439, 462, 475, 603]

            # this is for new 512 model
            if res==512:
                seeds = [0,2,5,7,12, 16, 34, 50, 51, 53, 54, 65, 71, 73, 132, 133, 213,311, 351, 453, 488, 515, 625, 629, 678]

            if res==1024:
                seeds = [45, 54,612,59,64,71,73,83, 86, 90, 99, 101, 102, 439, 166, 167,207, 225, 232, 250, 275, 338, 349, 426, 430]

        if args.select_seed:
            if args.carpaint:
                seeds = carpaint_list
            else:
                if res==256:
                    seeds = old_256_list if not args.tile else old_256_cirlist
                    # seeds = carpaint_list

                if res==512:
                    seeds = new_512_list

                if res==1024:
                    seeds = new_1k_list

        print("length of seeds ", len(seeds))

        for seed_idx, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))


            # ----------- this is for filtering out bad examples

            if args.filter_data:

                z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)    
                img_dict = {}
                for li in range(4):
                    cond_li_tmp = torch.from_numpy(all_li_filter[li]).unsqueeze(0).to(device)
                    w = G.mapping(z, None, truncation_psi=args.trunc, truncation_cutoff=args.trunc_cutoff)
                    img = G.synthesis(w, cond_li_tmp, noise_mode='const', test_mode=True, no_shift=True, upsample_fea=False)
                    img_dict[li]=img

                # compute similarity
                filter_loss = compute_sim(img_dict)
                print(f".......................filterng...................... {filter_loss}")

                if filter_loss <= 0.01: # 0.01 is threshold
                    continue

            else:
                # rand sampling
                z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)    
                w = G.mapping(z, None, truncation_psi=args.trunc, truncation_cutoff=args.trunc_cutoff)


            _, fea = G.synthesis(w, torch.from_numpy(all_li_v[0]).unsqueeze(0).to(device), out_fea=True, noise_mode='const' if not args.category else 'random', test_mode=True, no_shift=True)

            # ----------------------------------------------------------------------------------------
            maps = net(fea) * 0.5 + 0.5 # (-1,1) --> (0,1)



            if args.carpaint:
                rgb = maps[:,0:3,:,:].clamp(min=0, max=1)
                top_r = maps[:,3:4,:,:].clamp(min=0.2, max=0.9)
                top_spec = maps[:,4:7,:,:].clamp(min=0, max=1)
                bot_r = maps[:,7:8,:,:].clamp(min=0.2, max=0.9)
                bot_spec = maps[:,8:11,:,:].clamp(min=0, max=1)
                ren_fea = torch.cat((rgb, top_r, top_spec, bot_r, bot_spec), dim=1)
            else:

                if args.n_c==1:
                    N = height_to_normal(maps[:,0:1,:,:], size=size)
                    D = maps[:,1:4,:,:].clamp(min=0, max=1)
                    R = maps[:,4:5,:,:].repeat(1,3,1,1).clamp(min=0.2, max=0.9)
                    if args.out_nc==8:
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


            if args.tile:
                maps_t = tile_maps(maps)

                N = height_to_normal(maps_t[:,0:1,:,:], size=size*2)
                D = maps_t[:,1:4,:,:].clamp(min=0, max=1)
                R = maps_t[:,4:5,:,:].repeat(1,3,1,1).clamp(min=0.2, max=0.9)
                if args.out_nc==8:
                    S = maps_t[:,5:8,:,:].clamp(min=0, max=1)
                    ren_fea_t = torch.cat((N, D, R, S), dim=1)
                else:
                    ren_fea_t = torch.cat((N, D, R), dim=1)


            # search good category
            if args.category:

                # cond_li = torch.from_numpy(all_li_f[random.randint(0,3)]).unsqueeze(0).to(device)
                # rens = render(ren_fea, tex_pos, light, cond_li, isMetallic=False, no_decay=False, cam_pos=None, dir_flag=False).float() #[0,1] [1,C,H,W]
                # rens = (rens.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
                # PIL.Image.fromarray(rens.cpu().numpy(), 'RGB').save(f'{args.outdir}/{seed}_1_ren.png')

                cond_li = torch.from_numpy(all_li_f[4]).unsqueeze(0).to(device)
                rens = render(ren_fea, tex_pos, light, cond_li, isMetallic=False, no_decay=False, cam_pos=None, dir_flag=False).float() #[0,1] [1,C,H,W]
                rens = (rens.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
                PIL.Image.fromarray(rens.cpu().numpy(), 'RGB').save(f'{args.outdir}/{seed}_2_ren.png')

            else:

                save_image_grid(ren_fea.cpu().numpy(), os.path.join(args.outdir, f'{seed}_maps.png'), drange=[0,1], grid_size=(1,1))

                if args.eval_seed or args.select_seed:

                    if args.carpaint:

                        # save seperate maps
                        rgb = (rgb.permute(0, 2, 3, 1)**(1/2.2) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
                        top_r = (top_r.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).repeat(1,1,1,3).squeeze(0)
                        top_spec = (top_spec.permute(0, 2, 3, 1)**(1/2.2) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
                        bot_r = (bot_r.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).repeat(1,1,1,3).squeeze(0)
                        bot_spec = (bot_spec.permute(0, 2, 3, 1)**(1/2.2) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)

                        PIL.Image.fromarray(rgb.cpu().numpy(), 'RGB').save(os.path.join(args.outdir, f"{seed}_rgb.png"))
                        PIL.Image.fromarray(top_r.cpu().numpy(), 'RGB').save(os.path.join(args.outdir, f"{seed}_top_r.png"))
                        PIL.Image.fromarray(top_spec.cpu().numpy(), 'RGB').save(os.path.join(args.outdir, f"{seed}_top_spec.png"))
                        PIL.Image.fromarray(bot_r.cpu().numpy(), 'RGB').save(os.path.join(args.outdir, f"{seed}_bot_r.png"))
                        PIL.Image.fromarray(bot_spec.cpu().numpy(), 'RGB').save(os.path.join(args.outdir, f"{seed}_bot_spec.png"))


                    else:
                        # save seperate maps
                        N = (N.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).squeeze(0)
                        D = (D.permute(0, 2, 3, 1)**(1/2.2) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
                        R = (R.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
                        S = (S.permute(0, 2, 3, 1)**(1/2.2) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)

                        PIL.Image.fromarray(N.cpu().numpy(), 'RGB').save(os.path.join(args.outdir, f"{seed}_N.png"))
                        PIL.Image.fromarray(D.cpu().numpy(), 'RGB').save(os.path.join(args.outdir, f"{seed}_D.png"))
                        PIL.Image.fromarray(R.cpu().numpy(), 'RGB').save(os.path.join(args.outdir, f"{seed}_R.png"))
                        PIL.Image.fromarray(S.cpu().numpy(), 'RGB').save(os.path.join(args.outdir, f"{seed}_S.png"))


                        h = (maps[:,0:1,:,:].permute(0, 2, 3, 1).repeat(1,1,1,3) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
                        PIL.Image.fromarray(h.cpu().numpy(), 'RGB').save(f'{args.outdir}/{seed}_H.png')

                # ----------------------------------------------------------------------------------------
                # sampled render
                if args.save_video:
                    mp4 = os.path.join(args.outdir, f'{seed}_sampled_pt.mp4') 
                    video_out = imageio.get_writer(mp4, mode='I', fps=fps, codec='libx264', bitrate='10M')
                    for frame_idx in tqdm(range(num)):
                        cond_li = torch.from_numpy(all_li_v[frame_idx]).unsqueeze(0).to(device)
                        img, fea = G.synthesis(w, cond_li, out_fea=True, noise_mode='const', test_mode=True, no_shift=True)
                        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).squeeze(0)
                        img = img.cpu().numpy()
                        video_out.append_data(img)
                    video_out.close()


                for fix_li in range(5):
                    cond_li = torch.from_numpy(all_li_f[fix_li]).unsqueeze(0).to(device)
                    img, fea = G.synthesis(w, cond_li, out_fea=True, noise_mode='const', test_mode=True, no_shift=True)
                    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).squeeze(0)
                    PIL.Image.fromarray(img.cpu().numpy(), 'RGB').save(f'{args.outdir}/{seed}_{fix_li}_sampled.png')


                # ---------------------------------------------------------------------------------------
                # point light rendering
                if args.save_video:
                    mp4 = os.path.join(args.outdir, f'{seed}_rendered_pt.mp4') 
                    video_out = imageio.get_writer(mp4, mode='I', fps=fps, codec='libx264', bitrate='10M')
                    for frame_idx in tqdm(range(num)):
                        cond_li = torch.from_numpy(all_li_v[frame_idx]).unsqueeze(0).to(device)  
                        if args.carpaint:
                            rens = render_carpaint(ren_fea, tex_pos, light, cond_li, isMetallic=False, no_decay=False, cam_pos=None, dir_flag=False).float() #[0,1] [1,C,H,W]
                        else:
                            rens = render(ren_fea, tex_pos, light, cond_li, isMetallic=False, no_decay=False, cam_pos=None, dir_flag=False).float() #[0,1] [1,C,H,W]
                        rens = (rens.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
                        rens = rens.cpu().numpy()
                        video_out.append_data(rens)
                    video_out.close()

                    if args.tile:
                        mp4_t = os.path.join(args.outdir, f'{seed}_rendered_pt_tile.mp4') 
                        video_out_t = imageio.get_writer(mp4_t, mode='I', fps=fps, codec='libx264', bitrate='10M')
                        for frame_idx in tqdm(range(num)):
                            cond_li = torch.from_numpy(all_li_v[frame_idx]).unsqueeze(0).to(device)  
                            rens_t = render(ren_fea_t, tex_pos_t, light, cond_li, isMetallic=False, no_decay=False, cam_pos=None, dir_flag=False).float() #[0,1] [1,C,H,W]
                            rens_t = (rens_t.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
                            rens_t = rens_t.cpu().numpy()
                            video_out_t.append_data(rens_t)
                        video_out_t.close()



                for fix_li in range(5):
                    cond_li = torch.from_numpy(all_li_f[fix_li]).unsqueeze(0).to(device)
                    if args.carpaint:
                        rens = render_carpaint(ren_fea, tex_pos, light, cond_li, isMetallic=False, no_decay=False, cam_pos=None, dir_flag=False).float() #[0,1] [1,C,H,W]
                    else:
                        rens = render(ren_fea, tex_pos, light, cond_li, isMetallic=False, no_decay=False, cam_pos=None, dir_flag=False).float() #[0,1] [1,C,H,W]
                    rens = (rens.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
                    PIL.Image.fromarray(rens.cpu().numpy(), 'RGB').save(f'{args.outdir}/{seed}_{fix_li}_ren.png')

                    if args.tile:
                        cond_li = torch.from_numpy(all_li_f[fix_li]).unsqueeze(0).to(device)
                        rens_t = render(ren_fea_t, tex_pos_t, light, cond_li, isMetallic=False, no_decay=False, cam_pos=None, dir_flag=False).float() #[0,1] [1,C,H,W]
                        rens_t = (rens_t.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).squeeze(0)
                        PIL.Image.fromarray(rens_t.cpu().numpy(), 'RGB').save(f'{args.outdir}/{seed}_{fix_li}_ren_tile.png')                        


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
    n_c: int,
    layer_n: int,
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
    eval_seed: bool,
    select_seed: bool,
    filter_data: bool,
    size: float,

    # MatUnet 
    save_video: bool,
    batch_norm: bool,
    category: bool,
    carpaint: bool,
    tile: bool,

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
    args.eval_seed = eval_seed
    args.select_seed = select_seed
    args.size = size
    args.filter_data = filter_data
    args.category = category
    args.carpaint = carpaint
    args.tile = tile
    args.layer_n = layer_n
    args.n_c = n_c

    # for testing
    args.save_video = save_video
    args.matunet = matunet
    args.light_type = light_type
    args.dataset = dataset


    if args.n_c==2:
        args.out_nc=9

    if args.carpaint:
        args.out_nc = 11

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
@click.option('--out_nc', help='output channel of MatUnet: 5 | 8', type=int, default=5, metavar='INT')
@click.option('--n_c', help='number of channel of normal', type=int, default=1, metavar='INT')
@click.option('--layer_n', help='number of MatUnet', type=int, default=5, metavar='INT')
@click.option('--li_range', type=float, help='range of light', default=0.5, show_default=True)

@click.option('--use_mlp', help='use MLP', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--falloff', help='add falloff', type=float, required=False, default=False, show_default=True)
@click.option('--batch_norm', help='use batch normalizatin in the Unet', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--force_shift', help='force to shift feature during training', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--size', help=' size of sample', type=float, required=False, default=4, show_default=True)

# ------------- for testing---------------------
@click.option('--test_mode', help='this is test mode', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--opt_mode', help='this is optimization mode', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--shift', help='tile shift or not', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--matunet', help='loading MatUnet', type=str, required=False, metavar='DIR')
@click.option('--seeds', type=num_range, help='List of random seeds', required=False)
@click.option('--light_type', help='using directional light for all texel or point light or mixing: pt | dir | ptdir', type=str, default="pt", required=False)
@click.option('--dataset', default='../Dataset/TestData2', type=str, required=False)
@click.option('--save_video', help='save video for this', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--eval_seed', help='use eval seed for this', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--select_seed', help='use selected seed for this', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--filter_data', help='filtering during sampling', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--category', help='only save rendered for stone and leather', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--carpaint', help='use car paint model', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--tile', help='tile output or not', type=bool, required=False, metavar='BOOL', default=False, show_default=True)


def main(outdir, test_mode, opt_mode, seeds, **config_kwargs):

    args = set_args(**config_kwargs)

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    args.outdir = outdir
    maps_test(args, seeds)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
