# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

# from torch_utils.common_batch import height_to_normal, getTexPos, read_all_envs
# from torch_utils.env_render_batch import env_render

from torch_utils.render import set_param, render, getTexPos,height_to_normal
from torch_utils.misc import tile_shift
import random

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, svbrdf=False, D_s=None, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.D_s = D_s
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.svbrdf = svbrdf

        # if self.svbrdf:
        #     self.sample_env = 5
        #     # some rendering setting
        #     self.camera_pos = torch.tensor([0.0, 0.0,4], dtype=torch.float32, device=self.device)
        #     self.size = 4.0
        #     self.tex_pos = getTexPos(256, self.size, self.device).unsqueeze(0)

    def run_G(self, z, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws, c)

        # tile shift
        if self.svbrdf:
            img = tile_shift(img, img.shape[-1])
            
        # # add rendering pipeline:
        # if self.svbrdf:
        #     img = img*0.5+0.5 # [-1,1] --> [0,1]
        #     # light, light_pos, _ = set_param(self.device, Num = self.args.batch_size, rand=self.args.rand_light)
        #     N = height_to_normal(img[:,0:1,:,:], self.size)
        #     # fake_fea = torch.cat((N,img[:,1:4,:,:],img[:,4:5,:,:].repeat(1,3,1,1)),dim=1)

        #     # self.env = env['env_map'] if self.args.est_light else self.envs[random.randint(0, self.num_envs), :, :, :].repeat(temp.shape[0],1,1,1)
        #     # self.env = self.envs[random.sample(range(0, self.num_envs-1), temp.shape[0]), :, :, :]
        #     env = torch.ones(img.shape[0],3,8,8).to(self.device)
        #     img, _, _ = env_render(self.sample_env, self.camera_pos, self.tex_pos, albedo=img[:,1:4,:,:], normal=N, rough=img[:,4:5,:,:], env=env, device=self.device)
        #     assert img.isnan().any()==False, f'nan in self.fake_img'
        #     # elif self.args.render_opt=='pt':
        #     #     self.fake_img = render(fake_fea, self.tex_pos_crop if self.args.crop_out else self.tex_pos, light, light_pos, isMetallic=False, no_decay=False) #[0,1]

        #     img = 2*img - 1

        return img, ws

    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits

    def run_Ds(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D_s, sync):
            logits = self.D_s(img, c)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain, real_img2=None, real_c2=None, gen_c2=None, multi_in=False, multi_scale=0):

        # print(phase)
        if multi_scale:
            assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'D_smain', 'D_sreg', 'Dboth']
        else:        
            assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']

        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        if multi_scale:
            do_Dsmain = (phase in ['D_smain', 'Dboth'])
            do_Dsr1   = (phase in ['D_sreg', 'Dboth']) and (self.r1_gamma != 0)


        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)

                if multi_in:
                    gen_img2, _gen_ws2 = self.run_G(gen_z, gen_c2, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                    gen_logits2 = self.run_D(gen_img2, gen_c2, sync=False)
                    loss_Gmain2 = torch.nn.functional.softplus(-gen_logits2) # -log(sigmoid(gen_logits))

                if multi_scale:
                    # downsample gen_img to 64x64
                    # gen_img_s = torch.nn.functional.interpolate(gen_img, size=(multi_scale, multi_scale), mode='bilinear', align_corners=False)
                    gen_logits_s = self.run_Ds(gen_img, gen_c, sync=False)
                    loss_Gmain_s = torch.nn.functional.softplus(-gen_logits_s) # -log(sigmoid(gen_logits))


            with torch.autograd.profiler.record_function('Gmain_backward'):

                if multi_in:
                    loss_Gmain_tmp = (loss_Gmain + loss_Gmain2)*0.5
                    loss_Gmain_tmp.mean().mul(gain).backward()
                elif multi_scale:
                    loss_Gmain_tmp = (loss_Gmain + loss_Gmain_s)*0.5
                    loss_Gmain_tmp.mean().mul(gain).backward()
                else:
                    loss_Gmain.mean().mul(gain).backward()


        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
                gen_logits = self.run_D(gen_img, gen_c, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))

                if multi_in:
                    gen_img2, _gen_ws2 = self.run_G(gen_z, gen_c2, sync=False)
                    gen_logits2 = self.run_D(gen_img2, gen_c2, sync=False) # Gets synced by loss_Dreal.
                    loss_Dgen2 = torch.nn.functional.softplus(gen_logits2) # -log(1 - sigmoid(gen_logits))

            with torch.autograd.profiler.record_function('Dgen_backward'):
                if multi_in:
                    loss_Dgen_tmp = (loss_Dgen + loss_Dgen2)*0.5
                    loss_Dgen_tmp.mean().mul(gain).backward()                
                else:
                    loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                loss_Dreal2 = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                    if multi_in:
                        real_img_tmp2 = real_img2.detach()#.requires_grad_(do_Dr1)
                        real_logits2 = self.run_D(real_img_tmp2, real_c2, sync=sync)
                        loss_Dreal2 = torch.nn.functional.softplus(-real_logits2) # -log(sigmoid(real_logits))

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)


            with torch.autograd.profiler.record_function(name + '_backward'):
                if multi_in:    
                    (real_logits * 0 + (loss_Dreal+loss_Dreal2)*0.5 + loss_Dr1).mean().mul(gain).backward()
                else:
                    (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

        # for multi scale discrminator:
        if multi_scale:

            loss_Dsgen = 0
            if do_Dsmain:
                with torch.autograd.profiler.record_function('Dsgen_forward'):
                    gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
                    gen_logits = self.run_Ds(gen_img, gen_c, sync=False) # Gets synced by loss_Dreal.
                    training_stats.report('Loss/scores/fake', gen_logits)
                    training_stats.report('Loss/signs/fake', gen_logits.sign())
                    loss_Dsgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))

                with torch.autograd.profiler.record_function('Dsgen_backward'):
                    loss_Dsgen.mean().mul(gain).backward()

            # Dsmain: Maximize logits for real images.
            # Dsr1: Apply R1 regularization.
            if do_Dsmain or do_Dsr1:
                name = 'Dsreal_Dsr1' if do_Dsmain and do_Dsr1 else 'Dsreal' if do_Dsmain else 'Dsr1'
                with torch.autograd.profiler.record_function(name + '_forward'):

                    real_img_tmp = real_img.detach().requires_grad_(do_Dsr1)
                    real_logits = self.run_Ds(real_img_tmp, real_c, sync=sync)
                    training_stats.report('Loss/scores/real', real_logits)
                    training_stats.report('Loss/signs/real', real_logits.sign())

                    loss_Dsreal = 0
                    if do_Dsmain:
                        loss_Dsreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                        training_stats.report('Loss/D/loss', loss_Dsgen + loss_Dsreal)

                    loss_Dsr1 = 0
                    if do_Dsr1:
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                        r1_penalty = r1_grads.square().sum([1,2,3])
                        loss_Dsr1 = r1_penalty * (self.r1_gamma / 2)
                        training_stats.report('Loss/r1_penalty', r1_penalty)
                        training_stats.report('Loss/D/reg', loss_Dsr1)


                with torch.autograd.profiler.record_function(name + '_backward'):
                    (real_logits * 0 + loss_Dsreal + loss_Dsr1).mean().mul(gain).backward()

#-------------------------------------VGG network------------------------------------

class VGGLoss(torch.nn.Module):
    def __init__(self, device):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().to(device)
        self.criterion = torch.nn.MSELoss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

        self.mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(-1).unsqueeze(-1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(-1).unsqueeze(-1).to(device)


    def forward(self, x, y):  # x is output, y is GT 

        # preprocess
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())   
        return loss

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


#------------------------------------- TD loss------------------------------------


from torchvision.models.vgg import vgg19

class TextureDescriptor(torch.nn.Module):

    def __init__(self, device, low_level=False):
        super(TextureDescriptor, self).__init__()
        self.device = device
        self.outputs = []

        # get VGG19 feature network in evaluation mode
        self.net = vgg19(True).features.to(device)
        self.net.eval()

        # change max pooling to average pooling
        for i, x in enumerate(self.net):
            if isinstance(x, torch.nn.MaxPool2d):
                self.net[i] = torch.nn.AvgPool2d(kernel_size=2)

        def hook(module, input, output):
            self.outputs.append(output)

        #for i in [6, 13, 26, 39]: # with BN
        if low_level:
            for i in [4, 9]: # without BN
                self.net[i].register_forward_hook(hook)         
        else:
            for i in [4, 9, 18, 27]: # without BN
                self.net[i].register_forward_hook(hook)

        # weight proportional to num. of feature channels [Aittala 2016]
        self.weights = [1, 2, 4, 8, 8]

        # this appears to be standard for the ImageNet models in torchvision.models;
        # takes image input in [0,1] and transforms to roughly zero mean and unit stddev
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)

    def forward(self, x):
        self.outputs = []

        # run VGG features
        x = self.net(x)
        self.outputs.append(x)

        result = []
        batch = self.outputs[0].shape[0]

        for i in range(batch):
            temp_result = []
            for j, F in enumerate(self.outputs):

                # print(j, ' shape: ', F.shape)

                F_slice = F[i,:,:,:]
                f, s1, s2 = F_slice.shape
                s = s1 * s2
                F_slice = F_slice.view((f, s))

                # Gram matrix
                G = torch.mm(F_slice, F_slice.t()) / s
                temp_result.append(G.flatten())
            temp_result = torch.cat(temp_result)

            result.append(temp_result)
        return torch.stack(result)

    def eval_CHW_tensor(self, x):
        "only takes a pytorch tensor of size B * C * H * W"
        assert len(x.shape) == 4, "input Tensor cannot be reduced to a 3D tensor"
        x = (x - self.mean) / self.std
        return self.forward(x.to(self.device))




class TDLoss(torch.nn.Module):
    def __init__(self, device, num_pyramid, low_level=False):
        super(TDLoss, self).__init__()
        # create texture descriptor
        self.net_td = TextureDescriptor(device, low_level=low_level) 
        # fix parameters for evaluation 
        for param in self.net_td.parameters(): 
            param.requires_grad = False 

        self.num_pyramid = num_pyramid

        # self.GT_td = self.compute_td_pyramid(GT_img.to(device))


    def forward(self, img1, img2):

        td1 = self.compute_td_pyramid(img1)
        td2 = self.compute_td_pyramid(img2)

        tdloss = (td2 - td1).abs().mean() 

        return tdloss


    def compute_td_pyramid(self, img): # img: [0,1]
        """compute texture descriptor pyramid

        Args:
            img (tensor): 4D tensor of image (NCHW)
            num_pyramid (int): pyramid level]

        Returns:
            Tensor: 2-d tensor of texture descriptor
        """    
        # print('img type',img[0,:,0,0])
        # print('img type',img.dtype)

        # if img.dtype=='torch.uint8':

        td = self.net_td.eval_CHW_tensor(img) 
        for scale in range(self.num_pyramid):
            td_ = self.net_td.eval_CHW_tensor(torch.nn.functional.interpolate(img, scale_factor = 1.0/(2.0**(scale+1)), mode='bilinear', align_corners=True, recompute_scale_factor=True))
            td = torch.cat([td, td_], dim=1) 
        return td

