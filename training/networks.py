# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma
import random

from torch_utils.render import getTexPos, norm
from torch_utils.misc import tile_shift

import math
from torch.nn import functional as F

#----------------------------------------------------------------------------

@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

#----------------------------------------------------------------------------

@misc.profiled_function
def modulated_conv2d(
    x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                     # Modulation coefficients of shape [batch_size, in_channels].
    noise           = None,     # Optional noise tensor to add to the output activations.
    up              = 1,        # Integer upsampling factor.
    down            = 1,        # Integer downsampling factor.
    padding         = 0,        # Padding with respect to the upsampled image.
    resample_filter = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate      = True,     # Apply weight demodulation?
    flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
    circular        = False,
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        if circular:
            x = conv2d_resample.conv2d_resample_circular(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        else:
            x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings(): # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)


    if circular:
        x = conv2d_resample.conv2d_resample_circular(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    else:
        x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        kernel_size,                    # Width and height of the convolution kernel.
        bias            = True,         # Apply additive bias before the activation function?
        activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
        up              = 1,            # Integer upsampling factor.
        down            = 1,            # Integer downsampling factor.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
        channels_last   = False,        # Expect the input to have memory_format=channels_last?
        trainable       = True,         # Update the weights of this layer during training?
        circular        = False,        # circular or not

    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        self.circular = circular

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1) # slightly faster
        if self.circular:
            x = conv2d_resample.conv2d_resample_circular(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)
        else:
            x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
        no_cond_map     = False,    # no condition mapping network even with cond feature
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        self.no_cond_map = no_cond_map

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0 or self.no_cond_map:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0 and not self.no_cond_map:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0 and not self.no_cond_map:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        w_dim,                          # Intermediate latent (W) dimensionality.
        resolution,                     # Resolution of this layer.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        use_noise       = True,         # Enable noise input?
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last   = False,        # Use channels_last format for the weights?
        circular        = False,        # circular or not
    ):
        super().__init__()
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.circular = circular

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = self.resolution // self.up
        misc.assert_shape(x, [None, self.weight.shape[1], in_resolution, in_resolution])
        styles = self.affine(w)

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1) # slightly faster
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
            padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv, circular=self.circular)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class ToRGBLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False, circular=False):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.circular = circular

    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv, circular=self.circular)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        out_channels,                       # Number of output channels.
        w_dim,                              # Intermediate latent (W) dimensionality.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of output color channels.
        is_last,                            # Is this the last block?
        architecture        = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        circular            = False,        # not circular
        **layer_kwargs,                     # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        print('architecture G: ', self.architecture)
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0
        self.circular = circular


        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2,
                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, circular=self.circular, **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
            conv_clamp=conv_clamp, channels_last=self.channels_last, circular=self.circular,**layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            # print('rgb: ', out_channels)
            # print('img_channels: ', img_channels)
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last, circular=self.circular)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=2,
                resample_filter=resample_filter, channels_last=self.channels_last, circular=self.circular)

    def forward(self, x, img, ws, force_fp32=False, fused_modconv=None, **layer_kwargs):
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            with misc.suppress_tracer_warnings(): # this value will be treated as a constant
                fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

        # print(f'fused_modconv is {fused_modconv}')

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB.
        if img is not None:
            misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            if self.circular:
                img = upfirdn2d.upsample2d_circular(img, self.resample_filter)
            else:
                img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        deco_mlp,                   # use decoder render .       
        superres_scale  = 1,        # use super res module not not : 1: not || 2,4: use    
        batch_shift     = False,    # batch tile shift or no     
        # high_res        = False,    # extend to high res .       
        no_shift        = False,    # not tile shift .       
        use_ray         = False,    # add ray to decoder render .       
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 0,        # Use FP16 for the N highest resolutions.
        mlp_fea         = 32,       # number of MLP feature
        mlp_hidden      = 64,       # number of MLP hidden dim
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim

        self.superres_scale = superres_scale

        self.img_resolution = img_resolution #* self.superres_scale
        self.img_resolution_log2 = int(np.log2(self.img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)
        print('block_kwargs: ', block_kwargs)

        self.no_shift = no_shift
        self.deco_mlp = deco_mlp
        self.batch_shift = batch_shift
        self.circular = block_kwargs['circular']

        if self.deco_mlp:
            print('using deco_mlp.....')
            self.MLP_decoder = OSGDecoder(mlp_fea, 3, use_ray=use_ray, res=img_resolution, deco_mlp=self.deco_mlp, hidden_dim=mlp_hidden)

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

        # 

        # add superresolution module
        self.add_super = True if self.superres_scale!=1 else False
        if self.add_super: 
            print(f'using super res module {self.superres_scale}.....')
            self.supernet = Superres_module(scale_factor=self.superres_scale)
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb

            print(self.supernet)


    def forward(self, ws, c, out_fea=False, shift=None, test_mode=False, no_shift=False, upsample_fea=False, **block_kwargs):
        """
        out_Fea: output featuer or not
        shift: shift offset (tuple)
        test_mode: ONLY works for high res, crop 256 or 512
        no_shift: force no shift during inference (bool)
        """
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
            # print(f'res: {res}, img: {img.shape}')

        if self.deco_mlp:
            fea = img

            if not self.no_shift and not no_shift:
                # if self.superres_scale > 1:
                #     if not test_mode:

                #         fea = tile_shift(fea, fea.shape[-1], not_batch = not self.batch_shift)
                #     else:
                #         fea = tile_shift(fea, fea.shape[-1], not_batch = not self.batch_shift)

                # else:
                #     if self.circular:
                #         if shift is not None:
                #             fea = tile_shift(fea, fea.shape[-1], fix=shift, not_batch = not self.batch_shift)
                #         else:
                #             fea = tile_shift(fea, fea.shape[-1], not_batch = not self.batch_shift)

                if test_mode:
                    # print("test_mode............")
                    if shift is not None:
                        fea = tile_shift(fea, fea.shape[-1], fix=shift, not_batch = not self.batch_shift)
                    else:
                        fea = tile_shift(fea, fea.shape[-1], not_batch = not self.batch_shift)
                else:
                    if self.circular:
                        if shift is not None:
                            # print("training mode....................", shift)
                            fea = tile_shift(fea, fea.shape[-1], fix=shift, not_batch = not self.batch_shift)
                        else:
                            # print("training mode....................")
                            fea = tile_shift(fea, fea.shape[-1], not_batch = not self.batch_shift)



            # if upsampling 
            if upsample_fea:
                # print("before up fea: ", fea.shape)
                fea = torch.nn.functional.interpolate(fea, scale_factor = 2, mode='bicubic')
                # print("fea upsampled: ", fea.shape)

            img = self.MLP_decoder(fea,c)

        if out_fea:
            return img, fea

        else:
            return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        deco_mlp,                   # use decoder render .
        mlp_fea,                    # number of MLP feature
        mlp_hidden,                 # number of MLP hidden dim
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        synthesis_kwargs    = {},   # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        print('synthesis_kwargs: ', synthesis_kwargs)
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim

        self.img_resolution = img_resolution
        self.img_channels = img_channels if not deco_mlp else mlp_fea
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=self.img_channels, deco_mlp=deco_mlp, mlp_fea=mlp_fea, mlp_hidden=mlp_hidden, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, out_fea=False, truncation_psi=1, truncation_cutoff=None, test_mode=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = self.synthesis(ws, c, out_fea=out_fea, test_mode=test_mode, **synthesis_kwargs)

        return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        tmp_channels,                       # Number of intermediate channels.
        out_channels,                       # Number of output channels.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of input color channels.
        first_layer_idx,                    # Index of the first layer.
        architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
        activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        freeze_layers       = 0,            # Freeze-D: Number of layers to freeze.
        circular            = False,        # tileability
        crop                = False,        # crop
        cdir_d              = '0',        # conditional per direction to D

    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.circular = circular
        self.crop = crop
        self.cdir_d = cdir_d

        self.num_layers = 0
        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable
        trainable_iter = trainable_gen()

        if in_channels == 0 or architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels if self.cdir_d=='0' or self.cdir_d=='2' else img_channels+3, tmp_channels, kernel_size=1, activation=activation,
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last, circular=self.circular)

        self.conv0 = Conv2dLayer(tmp_channels, tmp_channels, kernel_size=3, activation=activation,
            trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last, circular=self.circular)

        self.conv1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, activation=activation, down=2,
            trainable=next(trainable_iter), resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, circular=self.circular)

        if architecture == 'resnet':
            self.skip = Conv2dLayer(tmp_channels, out_channels, kernel_size=1, bias=False, down=2,
                trainable=next(trainable_iter), resample_filter=resample_filter, channels_last=self.channels_last, circular=self.circular)

    def forward(self, x, img, force_fp32=False):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:
            if not self.crop:
                misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == 'skip':
            if not self.crop:
                if self.cdir_d=='0' or self.cdir_d=='2':
                    misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
                else:
                    misc.assert_shape(img, [None, self.img_channels+3, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img, self.resample_filter) if self.architecture == 'skip' else None

        # Main layers.
        if self.architecture == 'resnet':
            # print('res x1: ', x.shape)
            y = self.skip(x, gain=np.sqrt(0.5))
            # print('res y: ', y.shape)
            x = self.conv0(x)
            # print('res x2: ', x.shape)
            x = self.conv1(x, gain=np.sqrt(0.5))
            # print('res x3: ', x.shape)
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img

#----------------------------------------------------------------------------

@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings(): # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        cmap_dim,                       # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,                     # Resolution of this block.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        circular            = False,    # tileability
        crop                = False,    # crop
        cdir_d              = '0',    # conditional per direction to D

    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture
        self.circular = circular
        self.crop = crop
        self.cdir_d = cdir_d

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation, circular=self.circular)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp, circular=self.circular)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 or self.cdir_d=='1' else cmap_dim)

    def forward(self, x, img, cmap, force_fp32=False):
        if not self.crop:
            misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution]) # [NCHW]
        _ = force_fp32 # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            if not self.crop:
                misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)

        # Conditioning 0.
        if self.cmap_dim > 0 and self.cdir_d=='0':
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        assert x.dtype == dtype
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
        circular            = False,    # tileability
        crop                = False,    # crop
        cdir_d              = '0',      # conditional per direction to D
        scale               = 0,        # add multi scale or not
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution if scale==0 else scale
        self.img_resolution_log2 = int(np.log2(self.img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)
        self.circular = circular
        self.crop = crop
        self.cdir_d = cdir_d
        print('architecture D: ', architecture)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0
        self.cmap_dim = cmap_dim

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for idx, res in enumerate(self.block_resolutions):
            in_channels = channels_dict[res] if res < self.img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, circular=self.circular, crop=self.crop, cdir_d=self.cdir_d, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers

        # condition 0: mapping 3 --> 512
        if c_dim > 0 and self.cdir_d=='0':
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4 if not self.crop else 2, circular=self.circular, crop=self.crop, cdir_d=self.cdir_d, **epilogue_kwargs, **common_kwargs)

        # condition 2: ray dir [b,3,h,w] --> [b,512,4,4]
        if c_dim > 0 and self.cdir_d=='2':
            for idx, res in enumerate(self.block_resolutions):
                in_channels = channels_dict[res] if res < self.img_resolution else 0
                tmp_channels = channels_dict[res]
                out_channels = channels_dict[res // 2]
                use_fp16 = (res >= fp16_resolution)
                block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                    first_layer_idx=cur_layer_idx, use_fp16=use_fp16, circular=self.circular, crop=self.crop, cdir_d=self.cdir_d, **block_kwargs, **common_kwargs)
                setattr(self, f'cb{res}', block)
                cur_layer_idx += block.num_layers

            self.cb4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4 if not self.crop else 2, circular=self.circular, crop=self.crop, cdir_d=self.cdir_d, **epilogue_kwargs, **common_kwargs)

        self.crop_size = 160

        self.tex_pos = getTexPos(self.img_resolution, 4).unsqueeze(0)
        print("res in D: ", self.img_resolution)

        self.scale = self.img_resolution/256.
        # print('scale: ', self.scale, self.img_resolution ,scale)

    def forward(self, img, c, **block_kwargs):

        # crop all img from 256 to 512 (rm tiled results)
        if self.crop:
            x = random.randint(0, img.shape[-1] - self.crop_size)
            y = random.randint(0, img.shape[-1] - self.crop_size)
            img=img[:,:,x:x+self.crop_size, y:y+self.crop_size]

        x = None
        cx = None

        # condition 1: concat 3 dir to 3 img --> [b, 6, h, w]
        if self.cdir_d=='1' and self.c_dim>0:
            tex_pos = self.tex_pos.repeat(c.shape[0],1,1,1).to(c.device)

            # scale down light pos and input image
            if self.scale!=1:
                c *= self.scale
                img = torch.nn.functional.interpolate(img, size=(self.img_resolution, self.img_resolution), mode='bilinear', align_corners=False)

            ray_dir = c.unsqueeze(-1).unsqueeze(-1) - tex_pos
            ray_dir = norm(ray_dir)
            # print('in img D: ', img.shape, ray_dir.shape)
            img = torch.cat([img, ray_dir], dim=1)

        # condition 2: [b,3,h,w,]-->[b,512,4,4]
        if self.cdir_d=='2' and self.c_dim>0:
            tex_pos = self.tex_pos.repeat(c.shape[0],1,1,1).to(c.device)
            ray_dir = c.unsqueeze(-1).unsqueeze(-1) - tex_pos
            ray_dir = norm(ray_dir)

        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

            # condition 2: [b,3,h,w,]-->[b,512,4,4]
            if self.cdir_d=='2' and self.c_dim>0:
                cblock = getattr(self, f'cb{res}')
                cx, ray_dir = cblock(cx, ray_dir, **block_kwargs)                


        cmap = None

        # condition 0
        if self.c_dim > 0 and self.cdir_d=='0':
            cmap = self.mapping(None, c)

        x = self.b4(x, img, cmap)

        # Conditioning 2
        if self.c_dim > 0 and self.cdir_d=='2':
            cx = self.cb4(cx, ray_dir, None)
            misc.assert_shape(cx, [None, self.cmap_dim])
            x = (x * cx).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        return x

#----------------------------------------------------------------------------

class OSGDecoder(torch.nn.Module):
    def __init__(self, 
        n_features, 
        out_channel, 
        use_ray=False, 
        res=256,
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        deco_mlp        = 1,        # which decoder MLP to use
        hidden_dim      = 64,       # number of hidden dim

    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.use_ray = use_ray
        self.deco_mlp = deco_mlp


        if self.use_ray:

            if self.deco_mlp==1:
                print('using deco MLP 1')

                self.net = torch.nn.Sequential(
                    FullyConnectedLayer(n_features+3, self.hidden_dim),
                    torch.nn.Softplus(),
                    FullyConnectedLayer(self.hidden_dim, out_channel)
                )     
            elif self.deco_mlp==2:
                print('using deco MLP 2')
                self.net = torch.nn.Sequential(
                    FullyConnectedLayer(n_features+3, self.hidden_dim, activation=activation, lr_multiplier=lr_multiplier),
                    FullyConnectedLayer(self.hidden_dim, self.hidden_dim, activation=activation, lr_multiplier=lr_multiplier),
                    FullyConnectedLayer(self.hidden_dim, self.hidden_dim, activation=activation, lr_multiplier=lr_multiplier),
                    FullyConnectedLayer(self.hidden_dim, self.hidden_dim, activation=activation, lr_multiplier=lr_multiplier),
                    FullyConnectedLayer(self.hidden_dim, self.hidden_dim, activation=activation, lr_multiplier=lr_multiplier),
                    FullyConnectedLayer(self.hidden_dim, self.hidden_dim, activation=activation, lr_multiplier=lr_multiplier),
                    FullyConnectedLayer(self.hidden_dim, self.hidden_dim, activation=activation, lr_multiplier=lr_multiplier),
                    FullyConnectedLayer(self.hidden_dim, out_channel, lr_multiplier=lr_multiplier),
                )      


        else:
            self.net = torch.nn.Sequential(
                FullyConnectedLayer(n_features, self.hidden_dim),
                torch.nn.Softplus(),
                FullyConnectedLayer(self.hidden_dim, out_channel)
            )

        self.tex_pos = getTexPos(res, 4).unsqueeze(0)
        print("res in MLP: ", res)

    def forward(self, sampled_features, ray_dir=None):

        if self.use_ray:

            # this is for point light
            if ray_dir.dim()==2:
                # print("point light")
                # new version (norm + reverse light dir)
                # print("sample_fea: ", sampled_features.shape)
                if self.tex_pos.shape[-1] != sampled_features.shape[-1]:
                    tex_pos = getTexPos(sampled_features.shape[-1], 4).unsqueeze(0).repeat(ray_dir.shape[0],1,1,1).to(ray_dir.device)
                    # print("high res: ", tex_pos.shape)
                else:
                    tex_pos = self.tex_pos.repeat(ray_dir.shape[0],1,1,1).to(ray_dir.device)
                # print("ray_dir: ", ray_dir)
                ray_dir = ray_dir.unsqueeze(-1).unsqueeze(-1) - tex_pos
                ray_dir = norm(ray_dir)

                # old version
                # tex_pos = self.tex_pos.repeat(ray_dir.shape[0],1,1,1).to(ray_dir.device)
                # ray_dir = tex_pos - ray_dir.unsqueeze(-1).unsqueeze(-1)

            # this is for point light
            else:
                # print("directional light")
                # if self.tex_pos.shape[-1] != sampled_features.shape[-1]:
                #     tex_pos = getTexPos(sampled_features.shape[-1], 4).unsqueeze(0).repeat(ray_dir.shape[0],1,1,1).to(ray_dir.device)
                #     # print("high res: ", tex_pos.shape)
                # else:
                #     tex_pos = self.tex_pos.repeat(ray_dir.shape[0],1,1,1).to(ray_dir.device)
                # ray_dir = ray_dir - tex_pos
                ray_dir = norm(ray_dir)
                # print(ray_dir.shape)
                # print(tex_pos.shape)
                # ray_dir = ray_dir[:,:,0:1,0:1].repeat(1,1,tex_pos.shape[-1],tex_pos.shape[-1])
                # print(ray_dir)


            # print("ray_dir:", ray_dir[0,:,253:257,253:257])

 
            x = torch.cat([sampled_features, ray_dir], dim=1)

            N,C,H,W = x.shape
            x = x.permute(0,2,3,1).contiguous().view(N*H*W,C)

            x = self.net(x)
            rgb = x.view(N, H, W, -1).permute(0,3,1,2)

            return rgb
         
        else:
            x = sampled_features

            N,C,H,W = x.shape
            x = x.permute(0,2,3,1).contiguous().view(N*H*W,C)

            x = self.net(x)
            rgb = x.view(N, H, W, -1).permute(0,3,1,2)

            return rgb

#----------------------------------------------------------------------------

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_normal_(m.weight)     
        if m.bias is not None:
            m.bias.data.normal_(0,0.02)

        # n = m.in_channels
        # for k in m.kernel_size:
        #     n*=k
        # stdv = 1./np.sqrt(n)
        # m.weight.data.normal_(0.0,stdv)
        # if m.bias is not None:
        #     m.bias.data.normal_(0,0.02)
        #     # m.bias.data.fill_(0)

    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class CircularConv2d(torch.nn.Module):
    def __init__(self, in_c, out_c, ks, stride, pad):
        super(CircularConv2d, self).__init__()

        self.weight = torch.nn.Parameter( torch.randn(out_c, in_c, ks, ks) )
        # self.scale = 1 / math.sqrt(in_c * ks ** 2)
        self.ks = ks
        self.stride = stride
        self.pad = pad
        self.bias = torch.nn.Parameter(torch.zeros(out_c))

        # init
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, input):
        input = F.pad(input, (self.pad, self.pad, self.pad, self.pad), mode ='circular')
        out = F.conv2d(input, self.weight, bias=self.bias, stride=self.stride)
        return out


class UNet2dBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, deco=False, use_batchnorm=False, is_last=False):
        super(UNet2dBlock, self).__init__()
        self.in_c = in_c
        self.out_c = out_c

        # for decoder
        if deco:
            # self.model = torch.nn.Sequential(
            #     torch.nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            #     torch.nn.LeakyReLU(),
            #     # torch.nn.BatchNorm2d(out_c),
            #     torch.nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
            #     torch.nn.LeakyReLU(),
            #     # torch.nn.BatchNorm2d(out_c)
            # )
            if use_batchnorm:
                self.model = torch.nn.Sequential(
                    CircularConv2d(in_c=in_c, out_c=out_c, ks=3, stride=1, pad=1),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm2d(out_c),
                    CircularConv2d(in_c=out_c, out_c=out_c, ks=3, stride=1, pad=1),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm2d(out_c)
                )                
            else:
                self.model = torch.nn.Sequential(
                    CircularConv2d(in_c=in_c, out_c=out_c, ks=3, stride=1, pad=1),
                    torch.nn.LeakyReLU(),
                    CircularConv2d(in_c=out_c, out_c=out_c, ks=3, stride=1, pad=1),
                    torch.nn.LeakyReLU() if not is_last else torch.nn.Tanh(),
                )

        # for encoder
        else:
            # self.model = torch.nn.Sequential(
            #     torch.nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
            #     torch.nn.LeakyReLU(),
            #     # torch.nn.BatchNorm2d(out_c),
            #     torch.nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
            #     torch.nn.LeakyReLU(),
            #     # torch.nn.BatchNorm2d(out_c)
            # )
            if use_batchnorm:
                self.model = torch.nn.Sequential(
                    CircularConv2d(in_c=in_c, out_c=out_c, ks=4, stride=2, pad=1),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm2d(out_c),
                    CircularConv2d(in_c=out_c, out_c=out_c, ks=3, stride=1, pad=1),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm2d(out_c)
                )                
            else:
                self.model = torch.nn.Sequential(
                    CircularConv2d(in_c=in_c, out_c=out_c, ks=4, stride=2, pad=1),
                    torch.nn.LeakyReLU(),
                    CircularConv2d(in_c=out_c, out_c=out_c, ks=3, stride=1, pad=1),
                    torch.nn.LeakyReLU(),
                )

    def forward(self, x):
        return self.model(x)


class MatUnet(torch.nn.Module):
    def __init__(self, in_c=32, out_c=5, ngf=32, batch_norm=False, layer_n=5):
        super(MatUnet, self).__init__()

        self.layer_n = layer_n

        if self.layer_n==5:
            self.block0 = UNet2dBlock(in_c, ngf, use_batchnorm=batch_norm)
            self.block1 = UNet2dBlock(ngf, ngf*2, use_batchnorm=batch_norm)
            self.block2 = UNet2dBlock(ngf*2, ngf*4, use_batchnorm=batch_norm)
            self.block3 = UNet2dBlock(ngf*4, ngf*8, use_batchnorm=batch_norm)
            self.block4 = UNet2dBlock(ngf*8, ngf*8, use_batchnorm=batch_norm)

            self.block5 = UNet2dBlock(ngf*16, ngf*8, deco=True, use_batchnorm=batch_norm)
            self.block6 = UNet2dBlock(ngf*12, ngf*4, deco=True, use_batchnorm=batch_norm)
            self.block7 = UNet2dBlock(ngf*6, ngf*2, deco=True, use_batchnorm=batch_norm)
            self.block8 = UNet2dBlock(ngf*3, ngf, deco=True, use_batchnorm=batch_norm)

            # self.out = torch.nn.Conv2d(ngf, out_c, kernel_size=3, stride=1, padding=1)
            self.out = CircularConv2d(in_c=ngf, out_c=out_c, ks=3, stride=1, pad=1)

        elif self.layer_n==7:
            self.block0 = UNet2dBlock(in_c, ngf, use_batchnorm=batch_norm)
            self.block1 = UNet2dBlock(ngf, ngf*2, use_batchnorm=batch_norm)
            self.block2 = UNet2dBlock(ngf*2, ngf*4, use_batchnorm=batch_norm)
            self.block3 = UNet2dBlock(ngf*4, ngf*8, use_batchnorm=batch_norm)
            self.block4 = UNet2dBlock(ngf*8, ngf*8, use_batchnorm=batch_norm)
            self.block5 = UNet2dBlock(ngf*8, ngf*8, use_batchnorm=batch_norm)
            self.block6 = UNet2dBlock(ngf*8, ngf*8, use_batchnorm=batch_norm)

            self.block7 = UNet2dBlock(ngf*16, ngf*8, deco=True, use_batchnorm=batch_norm)
            self.block8 = UNet2dBlock(ngf*16, ngf*8, deco=True, use_batchnorm=batch_norm)
            self.block9 = UNet2dBlock(ngf*16, ngf*8, deco=True, use_batchnorm=batch_norm)
            self.block10 = UNet2dBlock(ngf*12, ngf*4, deco=True, use_batchnorm=batch_norm)
            self.block11 = UNet2dBlock(ngf*6, ngf*2, deco=True, use_batchnorm=batch_norm)
            self.block12 = UNet2dBlock(ngf*3, ngf, deco=True, use_batchnorm=batch_norm)

            # self.out = torch.nn.Conv2d(ngf, out_c, kernel_size=3, stride=1, padding=1)
            self.out = CircularConv2d(in_c=ngf, out_c=out_c, ks=3, stride=1, pad=1)


        self.tanh = torch.nn.Tanh()

        self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # bilinear
        self.leakyrelu = torch.nn.LeakyReLU(inplace=True)

    def forward(self, x):

        # conv1 = self.leakyrelu(self.block0(x))
        if self.layer_n==5:
            conv1 = self.block0(x)
            conv2 = self.block1(conv1)
            conv3 = self.block2(conv2)
            conv4 = self.block3(conv3)
            conv5 = self.block4(conv4)

            conv6 = self.block5(torch.cat([self.up(conv5), conv4], 1))
            conv7 = self.block6(torch.cat([self.up(conv6), conv3], 1))
            conv8 = self.block7(torch.cat([self.up(conv7), conv2], 1))
            conv9 = self.block8(torch.cat([self.up(conv8), conv1], 1))
            out = self.tanh(self.out(self.up(conv9)))

        elif self.layer_n==7:

            conv1 = self.block0(x)
            conv2 = self.block1(conv1)
            conv3 = self.block2(conv2)
            conv4 = self.block3(conv3)
            conv5 = self.block4(conv4)
            conv6 = self.block5(conv5)
            conv7 = self.block6(conv6)

            conv8 = self.block7(torch.cat([self.up(conv7), conv6], 1))
            conv9 = self.block8(torch.cat([self.up(conv8), conv5], 1))
            conv10 = self.block9(torch.cat([self.up(conv9), conv4], 1))
            conv11 = self.block10(torch.cat([self.up(conv10), conv3], 1))
            conv12 = self.block11(torch.cat([self.up(conv11), conv2], 1))
            conv13 = self.block12(torch.cat([self.up(conv12), conv1], 1))
            out = self.tanh(self.out(self.up(conv13)))
        
        return out



# class MatUnet(torch.nn.Module):
#     def __init__(self, in_c=32, out_c=5, ngf=32, batch_norm=False, layer_n=5):
#         super(MatUnet, self).__init__()

#         self.layer_n = layer_n

#         self.block0 = UNet2dBlock(in_c, ngf, use_batchnorm=batch_norm)
#         self.block1 = UNet2dBlock(ngf, ngf*2, use_batchnorm=batch_norm)
#         self.block2 = UNet2dBlock(ngf*2, ngf*4, use_batchnorm=batch_norm)
#         self.block3 = UNet2dBlock(ngf*4, ngf*8, use_batchnorm=batch_norm)
#         self.block4 = UNet2dBlock(ngf*8, ngf*8, use_batchnorm=batch_norm)

#         self.block5 = UNet2dBlock(ngf*16, ngf*8, deco=True, use_batchnorm=batch_norm)
#         self.block6 = UNet2dBlock(ngf*12, ngf*4, deco=True, use_batchnorm=batch_norm)
#         self.block7 = UNet2dBlock(ngf*6, ngf*2, deco=True, use_batchnorm=batch_norm)
#         self.block8 = UNet2dBlock(ngf*3, ngf, deco=True, use_batchnorm=batch_norm)

#         # self.out = torch.nn.Conv2d(ngf, out_c, kernel_size=3, stride=1, padding=1)
#         self.out = CircularConv2d(in_c=ngf, out_c=out_c, ks=3, stride=1, pad=1)

#         self.tanh = torch.nn.Tanh()

#         self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # bilinear
#         self.leakyrelu = torch.nn.LeakyReLU(inplace=True)

#     def forward(self, x):

#         # conv1 = self.leakyrelu(self.block0(x))
#         conv1 = self.block0(x)
#         conv2 = self.block1(conv1)
#         conv3 = self.block2(conv2)
#         conv4 = self.block3(conv3)
#         conv5 = self.block4(conv4)

#         conv6 = self.block5(torch.cat([self.up(conv5), conv4], 1))
#         conv7 = self.block6(torch.cat([self.up(conv6), conv3], 1))
#         conv8 = self.block7(torch.cat([self.up(conv7), conv2], 1))
#         conv9 = self.block8(torch.cat([self.up(conv8), conv1], 1))
#         out = self.tanh(self.out(self.up(conv9)))
        
#         return out

# -------------------------------------- this is for super res module

class Superres_module(torch.nn.Module):
    def __init__(self, scale_factor=2):
        super(Superres_module, self).__init__()

        self.num_layer = int(np.log2(scale_factor))
        self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True ) # bilinear


        for i in range(self.num_layer):
            if i==self.num_layer-1:
                block = UNet2dBlock(32, 32, deco=True, is_last=True)
            else:
                block = UNet2dBlock(32, 32, deco=True)
            
            setattr(self, f'super{i}', block)

    def forward(self, x):

        for i in range(self.num_layer):

            block = getattr(self, f'super{i}')
            x = block(self.up(x))

        return x


