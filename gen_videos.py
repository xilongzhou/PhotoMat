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

import random
import click
import dnnlib
import imageio
import numpy as np
import scipy.interpolate
import torch
from tqdm import tqdm

from PIL import Image

import legacy


from torch_utils import misc

from training.networks import Generator

# syn ----------------------------------------------------------------------------

def rand_light(num, li_range=0.45):

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


#----------------------------------------------------------------------------

def gen_interp_video(G, mp4: str, seed, li_range=0.45, psi=1, truncation_cutoff=14, device=torch.device('cuda'), tile_shift=False, upsample_fea=False, out = "tt", **video_kwargs):

    time = 2
    fps = 12
    num = time*fps

    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)


    # generate light
    all_li = rand_light(num, li_range=li_range)


    if tile_shift:

        shift_num = 3
        for k in range(shift_num):

            # Render video.
            name = mp4.split('.mp4')[0] + f'_{k}.mp4'
            video_out = imageio.get_writer(name, mode='I', fps=fps, codec='libx264', **video_kwargs)

            w_fix = random.randint(-255,255)
            h_fix = random.randint(-255,255)
            print(w_fix)

            for frame_idx in tqdm(range(num)):
                cond_li = torch.from_numpy(all_li[frame_idx]).unsqueeze(0).to(device)

                w = G.mapping(z, cond_li, truncation_psi=psi, truncation_cutoff=truncation_cutoff)
                img = G.synthesis(w, cond_li, noise_mode='const', test_mode=True, shift=(w_fix, h_fix), upsample_fea=upsample_fea)

                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).squeeze(0)
                img = img.cpu().numpy()

                video_out.append_data(img)
            video_out.close()

    else:

        # Render video.
        video_out = imageio.get_writer(mp4, mode='I', fps=fps, codec='libx264', **video_kwargs)

        for frame_idx in tqdm(range(num)):
            cond_li = torch.from_numpy(all_li[frame_idx]).unsqueeze(0).to(device)

            w = G.mapping(z, cond_li, truncation_psi=psi, truncation_cutoff=truncation_cutoff)
            img = G.synthesis(w, cond_li, noise_mode='const', test_mode=True, no_shift=True, upsample_fea=upsample_fea)

            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).squeeze(0)
            img = img.cpu().numpy()

            # if frame_idx%15==0:
            #     Image.fromarray(img, 'RGB').save(f'{out}/{seed}_{frame_idx}.png')

            video_out.append_data(img)
        video_out.close()


#----------------------------------------------------------------------------

def debug(G, seed, out, psi=1, truncation_cutoff=14, device=torch.device('cuda')):


    num = 1

    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

    # generate light
    all_li = rand_light(num)

    all_poses = []

    for frame_idx in tqdm(range(num)):
        cond_li = torch.from_numpy(all_li[frame_idx]).unsqueeze(0).to(device)

        w = G.mapping(z, cond_li, truncation_psi=psi, truncation_cutoff=truncation_cutoff)

        img, fea = G.synthesis(w, cond_li,  out_fea=True, noise_mode='const', test_mode=True)

        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).squeeze(0)
        img = img.cpu().numpy()

        fea = (fea.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).squeeze(0)
        fea = fea.cpu().numpy()

        # tile hack
        fea_t = np.tile(fea, (2,2,1))

        print(fea_t.shape)

        for k in range(fea_t.shape[-1]):
            Image.fromarray(fea_t[:,:,k], 'L').save(f'{out}/seed{seed:04d}_fea{k}.png')

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

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--outdir', help='Output directory', type=str, required=True, metavar='DIR')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--tile_shift', help='tile shift when sampling', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--vis_fea', help='debug feature', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--upsample_fea', help='upsample feature or not', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--li_range', type=float, help='range of light', default=0.45, show_default=True)

def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    reload_modules: bool,
    tile_shift: bool,
    upsample_fea: bool,
    vis_fea: bool,
    li_range: float,

):
    """Render a latent vector interpolation video.

    Examples:

    \b
    # Render a 4x2 grid of interpolations for seeds 0 through 31.
    python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    Animation length and seed keyframes:

    The animation length is either determined based on the --seeds value or explicitly
    specified using the --num-keyframes option.

    When num keyframes is specified with --num-keyframes, the output video length
    will be 'num_keyframes*w_frames' frames.

    If --num-keyframes is not specified, the number of seeds given with
    --seeds must be divisible by grid size W*H (--grid).  In this case the
    output video length will be '# seeds/(w*h)*w_frames' frames.
    """

    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    init_kwargs_tmp = G.init_kwargs
    init_tmp = G.init_args

    print("init_kwargs_tmp 1", init_kwargs_tmp)

    try:
        if 'mlp_fea' not in init_kwargs_tmp:
            init_kwargs_tmp['mlp_fea']=32

        if 'mlp_hidden' not in init_kwargs_tmp:
            init_kwargs_tmp['mlp_hidden']=64

        init_kwargs_tmp["synthesis_kwargs"].pop('high_res', None)

    except KeyError:
        print("""""""""""""""""""")
        pass

    print("init_kwargs_tmp 2", init_kwargs_tmp)


    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        print("Reloading Modules!")
        G_new = Generator(*G.init_args, **init_kwargs_tmp).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G = G_new



    if truncation_cutoff == 0:
        truncation_psi = 1.0 # truncation cutoff of 0 means no truncation anyways
    if truncation_psi == 1.0:
        truncation_cutoff = 14 # no truncation so doesn't matter where we cutoff

    for seed in seeds:
        output = os.path.join(outdir, f'{seed}.mp4') 
        if not vis_fea:
            gen_interp_video(G=G, mp4=output, bitrate='10M', seed=seed, li_range=li_range,psi=truncation_psi, truncation_cutoff=truncation_cutoff, tile_shift=tile_shift, upsample_fea=upsample_fea, out = outdir)
        else:
            debug(G=G, seed=seed, out = outdir, psi=truncation_psi, truncation_cutoff=truncation_cutoff)



#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
