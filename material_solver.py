# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate lerp videos using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import imageio
import numpy as np
import scipy.interpolate
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm

from PIL import Image

import legacy

from torch_utils.render import set_param, getTexPos, render, height_to_normal
from torch_utils import misc

from training.networks import Generator

import copy
from training.loss import VGGLoss, TDLoss

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

# rand point light pos
def rand_light(num, device, li_range=0.5):
    x = np.random.rand(num, 1)*li_range*2 - li_range
    y = np.random.rand(num, 1)*li_range*2 - li_range
    light_pos = np.concatenate((x,y,np.ones_like(y)), axis=-1) * 4.
    light_pos = torch.from_numpy(light_pos).float().to(device)
    return light_pos

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=False)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--lr_init', type=float, help='Truncation psi', default=1e-3, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--col_camli', help='collocate camera and light?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--w_vgg', help='weight of vgg loss', type=float, required=False, default=0, show_default=True)
@click.option('--w_td', help='weight of regularization loss', type=float, required=False, default=0, show_default=True)
@click.option('--debug', help='debug mode (syn)', type=bool, required=False, default=False, show_default=True)
@click.option('--dir_li', help='use dir light', type=bool, required=False, default=False, show_default=True)
@click.option('--inter', help='inter every # iterations', type=int, required=False, default=0, show_default=True)
@click.option('--total_iter', help='total number of iterations', type=int, required=False, default=2000, show_default=True)
@click.option('--number', help='# of samples', type=int, required=False, default=20, show_default=True)
@click.option('--pos_meth', help='gamma|reinhard', type=str, required=False, default='gamma', show_default=True)
@click.option('--n_c', help='# of normal channel', type=int, required=False, default=1, show_default=True)

def maps_sovler(
    network: str,
    seeds: List[int],
    truncation_psi: float,
    lr_init: float,
    truncation_cutoff: int,
    total_iter: int,
    inter: int,
    number: int,
    outdir: str,
    class_idx: Optional[int],
    reload_modules: bool,
    col_camli: bool,
    w_vgg: float,
    w_td: float,
    debug: bool,
    dir_li: bool,
    pos_meth: str,
    n_c: int,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained FFHQ model.
    python gen_samples.py --outdir=output --trunc=0.7 --seeds=0-5 --shapes=True\\
        --network=ffhq-rebalanced-128.pkl
    """

    print('Loading networks from "%s"...' % network)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network) as f:
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

    if reload_modules:
        print("Reloading Modules!")
        G = Generator(*G_tmp.init_args, **init_kwargs_tmp).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G_tmp, G, require_all=True)
    
    G_ema = copy.deepcopy(G).eval()

    os.makedirs(outdir, exist_ok=True)

    # settings
    batch = number
    iterations = total_iter
    light, cam_pos, size = set_param(device)
    tex_pos = getTexPos(res, size, device).unsqueeze(0)  

    # metrics
    # criterion_vgg = VGGLoss()
    # Load VGG16 feature detector.
    if w_vgg!=0.:
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with dnnlib.util.open_url(url) as f:
            vgg16 = torch.jit.load(f).eval().to(device)

    # loss
    loss_L1 = torch.nn.L1Loss()
    loss_TD = TDLoss(device, 2)
    loss_VGG = VGGLoss(device)


    if res==256:
        # old
        seeds = [305, 306, 310, 313,315, 316, 324, 333, 343, 348, 363, 398, 402, 403, 411, 422, 425, 426, 433, 434, 436, 439, 462, 475, 603]

    # this is for new 512 model
    if res==512:
        seeds = [0,2,5,7,12, 16, 34, 50, 51, 53, 54, 65, 71, 73, 132, 133, 213,311, 351, 453, 488, 515, 625, 629, 678]
        # seeds = [132, 213]

    if res==1024:
        seeds = [45, 54,612,59,64,71,73,83, 86, 90, 99, 101, 102, 439, 166, 167,207, 225, 232, 250, 275, 338, 349, 426, 430]


    # Generate images.
    for seed_idx, seed in enumerate(seeds):

        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G_ema.z_dim)).to(device)
        w = G_ema.mapping(z, None)

        # define materials maps (1,5,H,W)
        maps_init = torch.full((1,8,res,res), 0.5)
        maps_opt = torch.tensor(maps_init, dtype=torch.float32, device=device, requires_grad=True)
        # inten_opt = torch.tensor(0.1, dtype=torch.float32, device=device, requires_grad=True)

        # optimizer
        optimizer = torch.optim.Adam([maps_opt], betas=(0.9, 0.999), lr=lr_init,)
        scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)

        # optimization
        for i in range(iterations+1):

            # generate RGB re-renders and light
            cond_li = rand_light(1, device)
            imgs, _ = G_ema.synthesis(w, cond_li, noise_mode="const",out_fea=True, test_mode=True, no_shift=True) # noise_mode='const'
            imgs = imgs.float()*0.5 + 0.5
            imgs  = imgs.clamp(0,1) # clamp [0,1]
            # -------------------- different loading strategy -----------------------------

            # render maps
            if n_c==1:
                N = height_to_normal(maps_opt[:,0:1,:,:], size=size)
                D = maps_opt[:,1:4,:,:].clamp(min=0, max=1)
                R = maps_opt[:,4:5,:,:].repeat(1,3,1,1).clamp(min=0.2, max=0.9)
                S = maps_opt[:,5:8,:,:].clamp(min=0, max=1)
                ren_fea = torch.cat((N, D, R, S), dim=1)
  
            elif n_c==2:
                N = xy_to_normal(maps_opt[:,0:2,:,:])
                D = maps_opt[:,2:5,:,:].clamp(min=0, max=1)
                R = maps_opt[:,5:6,:,:].repeat(1,3,1,1).clamp(min=0.2, max=0.9)
                S = maps_opt[:,6:9,:,:].clamp(min=0, max=1)
                ren_fea = torch.cat((N, D, R, S), dim=1)


            rens = render(ren_fea, tex_pos, light, cond_li, isMetallic=False, no_decay=False, cam_pos=None, dir_flag=False).float() #[0,1] [1,C,H,W]

            # regularization of material maps
            reg_loss = 0
            vgg_loss = 0
            td_loss = 0
            # reg_loss = maps_opt.norm()*w_reg_r if w_reg_r!=0 else 0

            if w_vgg!=0:
                vgg_loss = loss_VGG(rens, imgs)*w_vgg

            if w_td!=0:
                td_loss = loss_TD(rens, imgs)*w_td


            L1_loss = loss_L1(imgs, rens)*0.1
            full_loss = L1_loss + reg_loss + vgg_loss + td_loss

            optimizer.zero_grad()
            full_loss.backward()
            optimizer.step()
            scheduler.step()

            if i%5000 == 0:

                out_lr = optimizer.param_groups[0]['lr']

                print(f'iteration: {i}, lr : {out_lr}, L1 loss: {L1_loss}, vgg loss: {vgg_loss}, td loss: {td_loss}, reg loss: {reg_loss}')

                maps_out = torch.cat([0.5*N+0.5, D**(1/2.2), R, S**(1/2.2)], dim=-1)
                maps_out = (maps_out.permute(0, 2, 3, 1)*255).clamp(0, 255).to(torch.uint8)

                # save maps
                Image.fromarray(maps_out[0].detach().cpu().numpy(), 'RGB').save(f'{outdir}/{seed:04d}_fea.png')

                # save images
                rens_out = (rens.permute(0, 2, 3, 1)*255).clamp(0, 255).to(torch.uint8)
                imgs_out = (imgs.permute(0, 2, 3, 1)*255).clamp(0, 255).to(torch.uint8)

                Image.fromarray(rens_out[0].detach().cpu().numpy(), 'RGB').save(f'{outdir}/{seed:04d}_out{i}.png')
                Image.fromarray(imgs_out[0].detach().cpu().numpy(), 'RGB').save(f'{outdir}/{seed:04d}_rgb{i}.png')




#----------------------------------------------------------------------------

if __name__ == "__main__":
    maps_sovler() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
