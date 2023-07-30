# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate images and shapes using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

import legacy

from torch_utils import misc
from training.networks import Generator



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

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

# syn ----------------------------------------------------------------------------

# def rand_light():
#     # u_1 = np.abs(np.random.normal(0,0.2,(1))).clip(0,0.9)
#     u_1 = np.array([0.3])
#     u_2 = np.random.uniform(0,1,(1))
#     theta = 2*np.pi*u_2

#     r = np.sqrt(u_1)
#     z = np.sqrt(1-r*r)
#     x = r*np.cos(theta)
#     y = r*np.sin(theta)

#     light_pos = np.concatenate((x,y,z),axis=0) * 4.

#     return light_pos

# real ----------------------------------------------------------------------------

def rand_light():

    r = 0.5

    tmp = np.random.uniform(-r,r,(2,))
    x = np.array([tmp[0]])
    y = np.array([tmp[1]])

    # print("x: ", x)
    # print("y: ", y)

    light_pos = np.concatenate((x,y,np.array([1])),axis=0)*4

    return light_pos


def rand_light_fix(num=4, li_range=0.5):

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

def compute_sim(img_dict):

    # 0-2, 1-3, 0-1, 2-3
    L2 = torch.nn.MSELoss()
    total = 0

    total += L2(img_dict[0], img_dict[2])
    total += L2(img_dict[1], img_dict[3])
    total += L2(img_dict[0], img_dict[1])
    total += L2(img_dict[2], img_dict[3])

    total /= 4
    print(total)

    return total




#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    class_idx: Optional[int],
    reload_modules: bool,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained FFHQ model.
    python gen_samples.py --outdir=output --trunc=0.7 --seeds=0-5 --shapes=True\\
        --network=ffhq-rebalanced-128.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # for key in G.init_kwargs:
    #     if key=="synthesis_kwargs":
    #         print(key)
    #         for key2 in G.init_kwargs[key]:
    #             print(key2)
    init_kwargs_tmp = G.init_kwargs

    try:
        
        if 'mlp_fea' not in init_kwargs_tmp:
            init_kwargs_tmp['mlp_fea']=32

        if 'mlp_hidden' not in init_kwargs_tmp:
            init_kwargs_tmp['mlp_hidden']=64

        init_kwargs_tmp["synthesis_kwargs"].pop('high_res', None)

    except KeyError:
        print("""""""""""""""""""")
        pass


    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        print("Reloading Modules!")
        # G_new = Generator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        G_new = Generator(*G.init_args, **init_kwargs_tmp).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G = G_new

    os.makedirs(outdir, exist_ok=True)


    # sample list
    """
    good sample/bad sample
    """
    # if res==256:
    #     good_list=[]
    #     bad_list=[]

    # if res==512:
    test_list=[0,2,5,7,12, 16, 34, 50, 51, 53, 54, 65, 71, 73, 132, 133, 213,311, 351, 453, 488, 515, 625, 629, 678]
    good_list=[0,2,5,7,12, 16, 17, 30, 34, 45, 47, 50, 51, 53, 54, 65, 71, 73, 132, 213,311, 351, 453, 488, 515, 625, 629, 678,\
    694,142, 148, 170,230,222,216,219,236,424,238, 239,249,258,310,326,325,323,329,337,345,346,350,362,363, 376,389,406,419,453]
    bad_list=[13, 24, 28, 32, 36, 40, 59, 62, 63, 68, 72, 78, 84, 90, 91, 100, 106, 107, 110,111,112,113,141,162,182,183,\
    181, 193, 200,227,226,248, 277, 304, 313, 330, 331,335,336, 360,356,392,430,456]

    # if res==1024


    all_li = rand_light_fix()
    all_losses = []
    # Generate images.
    # for seed_idx, seed in enumerate(seeds):
    #     print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
    #     z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)    
    # for i in range(5000):
    for seed in bad_list:
        img_dict={}

        print('Generating image %d ...' % (seed))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

        for li in range(4):
            cond_li = torch.from_numpy(all_li[li]).unsqueeze(0).to(device)
            ws = G.mapping(z, cond_li, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            img = G.synthesis(ws, cond_li, noise_mode='const', test_mode=True, no_shift=True, upsample_fea=False)
            img_dict[li]=img
            # img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

            # PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')
            # PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed_{seed:04d}_{li}.png')

        # compute similarity
        loss = compute_sim(img_dict)
        all_losses.append(loss)


        with open(f'{outdir}/out.txt', 'a') as f:
            f.write(f"{seed}: {loss} \n")

        f.close()

        # for tiled

    print(f"min is {max(all_losses)} ")


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
