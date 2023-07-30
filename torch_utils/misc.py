# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import re
import contextlib
import numpy as np
import torch
import warnings
import dnnlib
import random
#----------------------------------------------------------------------------
# Cached construction of constant tensors. Avoids CPU=>GPU copy when the
# same constant is used multiple times.

_constant_cache = dict()

def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('cpu')
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (value.shape, value.dtype, value.tobytes(), shape, dtype, device, memory_format)
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor

#----------------------------------------------------------------------------
# Replace NaN/Inf with specified numerical values.

try:
    nan_to_num = torch.nan_to_num # 1.8.0a0
except AttributeError:
    def nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None): # pylint: disable=redefined-builtin
        assert isinstance(input, torch.Tensor)
        if posinf is None:
            posinf = torch.finfo(input.dtype).max
        if neginf is None:
            neginf = torch.finfo(input.dtype).min
        assert nan == 0
        return torch.clamp(input.unsqueeze(0).nansum(0), min=neginf, max=posinf, out=out)

#----------------------------------------------------------------------------
# Symbolic assert.

try:
    symbolic_assert = torch._assert # 1.8.0a0 # pylint: disable=protected-access
except AttributeError:
    symbolic_assert = torch.Assert # 1.7.0

#----------------------------------------------------------------------------
# Context manager to suppress known warnings in torch.jit.trace().

class suppress_tracer_warnings(warnings.catch_warnings):
    def __enter__(self):
        super().__enter__()
        warnings.simplefilter('ignore', category=torch.jit.TracerWarning)
        return self

#----------------------------------------------------------------------------
# Assert that the shape of a tensor matches the given list of integers.
# None indicates that the size of a dimension is allowed to vary.
# Performs symbolic assertion when used in torch.jit.trace().

def assert_shape(tensor, ref_shape):
    if tensor.ndim != len(ref_shape):
        raise AssertionError(f'Wrong number of dimensions: got {tensor.ndim}, expected {len(ref_shape)}')
    for idx, (size, ref_size) in enumerate(zip(tensor.shape, ref_shape)):
        if ref_size is None:
            pass
        elif isinstance(ref_size, torch.Tensor):
            with suppress_tracer_warnings(): # as_tensor results are registered as constants
                symbolic_assert(torch.equal(torch.as_tensor(size), ref_size), f'Wrong size for dimension {idx}')
        elif isinstance(size, torch.Tensor):
            with suppress_tracer_warnings(): # as_tensor results are registered as constants
                symbolic_assert(torch.equal(size, torch.as_tensor(ref_size)), f'Wrong size for dimension {idx}: expected {ref_size}')
        elif size != ref_size:
            raise AssertionError(f'Wrong size for dimension {idx}: got {size}, expected {ref_size}')

#----------------------------------------------------------------------------
# Function decorator that calls torch.autograd.profiler.record_function().

def profiled_function(fn):
    def decorator(*args, **kwargs):
        with torch.autograd.profiler.record_function(fn.__name__):
            return fn(*args, **kwargs)
    decorator.__name__ = fn.__name__
    return decorator

#----------------------------------------------------------------------------
# Sampler for torch.utils.data.DataLoader that loops over the dataset
# indefinitely, shuffling items as it goes.

class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1

#----------------------------------------------------------------------------
# Utilities for operating with torch.nn.Module parameters and buffers.

def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())

def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())

def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = {name: tensor for name, tensor in named_params_and_buffers(src_module)}
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)

#----------------------------------------------------------------------------
# Context manager for easily enabling/disabling DistributedDataParallel
# synchronization.

@contextlib.contextmanager
def ddp_sync(module, sync):
    assert isinstance(module, torch.nn.Module)
    if sync or not isinstance(module, torch.nn.parallel.DistributedDataParallel):
        yield
    else:
        with module.no_sync():
            yield

#----------------------------------------------------------------------------
# Check DistributedDataParallel consistency across processes.

def check_ddp_consistency(module, ignore_regex=None):
    assert isinstance(module, torch.nn.Module)
    for name, tensor in named_params_and_buffers(module):
        fullname = type(module).__name__ + '.' + name
        if ignore_regex is not None and re.fullmatch(ignore_regex, fullname):
            continue
        tensor = tensor.detach()
        other = tensor.clone()
        torch.distributed.broadcast(tensor=other, src=0)
        assert (nan_to_num(tensor) == nan_to_num(other)).all(), fullname

#----------------------------------------------------------------------------
# Print summary table of module hierarchy.

def print_module_summary(module, inputs, max_nesting=3, skip_redundant=True):
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list))

    # Register hooks.
    entries = []
    nesting = [0]
    def pre_hook(_mod, _inputs):
        nesting[0] += 1
    def post_hook(mod, _inputs, outputs):
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
            entries.append(dnnlib.EasyDict(mod=mod, outputs=outputs))
    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # Run module.
    outputs = module(*inputs)
    for hook in hooks:
        hook.remove()

    # Identify unique outputs, parameters, and buffers.
    tensors_seen = set()
    for e in entries:
        e.unique_params = [t for t in e.mod.parameters() if id(t) not in tensors_seen]
        e.unique_buffers = [t for t in e.mod.buffers() if id(t) not in tensors_seen]
        e.unique_outputs = [t for t in e.outputs if id(t) not in tensors_seen]
        tensors_seen |= {id(t) for t in e.unique_params + e.unique_buffers + e.unique_outputs}

    # Filter out redundant entries.
    if skip_redundant:
        entries = [e for e in entries if len(e.unique_params) or len(e.unique_buffers) or len(e.unique_outputs)]

    # Construct table.
    rows = [[type(module).__name__, 'Parameters', 'Buffers', 'Output shape', 'Datatype']]
    rows += [['---'] * len(rows[0])]
    param_total = 0
    buffer_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}
    for e in entries:
        name = '<top-level>' if e.mod is module else submodule_names[e.mod]
        param_size = sum(t.numel() for t in e.unique_params)
        buffer_size = sum(t.numel() for t in e.unique_buffers)
        output_shapes = [str(list(e.outputs[0].shape)) for t in e.outputs]
        output_dtypes = [str(t.dtype).split('.')[-1] for t in e.outputs]
        rows += [[
            name + (':0' if len(e.outputs) >= 2 else ''),
            str(param_size) if param_size else '-',
            str(buffer_size) if buffer_size else '-',
            (output_shapes + ['-'])[0],
            (output_dtypes + ['-'])[0],
        ]]
        for idx in range(1, len(e.outputs)):
            rows += [[name + f':{idx}', '-', '-', output_shapes[idx], output_dtypes[idx]]]
        param_total += param_size
        buffer_total += buffer_size
    rows += [['---'] * len(rows[0])]
    rows += [['Total', str(param_total), str(buffer_total), '-', '-']]

    # Print table.
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    print()
    for row in rows:
        print('  '.join(cell + ' ' * (width - len(cell)) for cell, width in zip(row, widths)))
    print()
    return outputs

#----------------------------------------------------------------------------

# self-defined tile shift
#input [B,C,H,W]
def tile_shift(x, size, fix=None, not_batch=True):

    assert len(x.shape)==4, "dim of x is incorrect"
    assert x.shape[-1]==x.shape[-2], "x is not square" 
    
    b,c,h,w = x.shape

    # p = x.clone()
    p = torch.ones([b,c,size,size], dtype=x.dtype, device=x.device)

    # not_batch = True
    if fix is not None:
        not_batch = False


    if not not_batch:
        if fix is None:
            w0 = np.random.randint(-size+1,w-1)
            h0 = np.random.randint(-size+1,h-1)
        else:
            w0, h0 = fix[0], fix[1]

        wc = w0 + size
        hc = h0 + size

    # seperate crop and stitch them manually
    # [7 | 8 | 9]
    # [4 | 5 | 6]
    # [1 | 2 | 3]

    if not_batch:

        for k in range(b):
            # print("instance shift")
            w0 = np.random.randint(-size+1,w-1)
            h0 = np.random.randint(-size+1,h-1)

            # print("step: ", k, "offset: ",  w0 , h0)
            wc = w0 + size
            hc = h0 + size

            # 1
            if h0<=0 and w0<=0:
                p[k:k+1,:,0:-h0,0:-w0] = x[k:k+1,:, h+h0:h, w+w0:w]
                p[k:k+1,:,-h0:,0:-w0] = x[k:k+1,:, 0:hc, w+w0:w]
                p[k:k+1,:,0:-h0,-w0:] = x[k:k+1,:, h+h0:h, 0:wc]
                p[k:k+1,:,-h0:,-w0:] = x[k:k+1,:, 0:hc, 0:wc]
            # 2
            elif h0<=0 and (w0<w-size and w0>0):
                p[k:k+1,:,0:-h0,:] = x[k:k+1,:, h+h0:h,w0:wc]
                p[k:k+1,:,-h0:,:] = x[k:k+1,:, 0:hc, w0:wc]
            # 3
            elif h0<=0 and w0 >=w-size:
                p[k:k+1,:,0:-h0,0:w-w0] = x[k:k+1,:, h+h0:h, w0:w]
                p[k:k+1,:,-h0:,0:w-w0] = x[k:k+1,:, 0:hc, w0:w]
                p[k:k+1,:,0:-h0,w-w0:] = x[k:k+1,:, h+h0:h, 0:wc-w]
                p[k:k+1,:,-h0:,w-w0:] = x[k:k+1,:, 0:hc, 0:wc-w]

            # 4
            elif (h0>0 and h0<h-size) and w0<=0:
                p[k:k+1,:,:,0:-w0] = x[k:k+1,:, h0:hc, w+w0:w]
                p[k:k+1,:,:,-w0:] = x[k:k+1,:, h0:hc, 0:wc]
            # 5
            elif (h0>0 and h0<h-size) and (w0<w-size and w0>0):
                p[k:k+1,...] = x[k:k+1,:, h0:hc, w0:wc]
            # 6
            elif (h0>0 and h0<h-size) and w0 >=w-size:
                p[k:k+1,:,:,0:w-w0] = x[k:k+1,:, h0:hc, w0:w]
                p[k:k+1,:,:,w-w0:] = x[k:k+1,:, h0:hc, 0:wc-w]

            # 7
            elif h0 >=h-size and w0<=0:
                p[k:k+1,:,0:h-h0,0:-w0] = x[k:k+1,:, h0:h, w+w0:w]
                p[k:k+1,:,h-h0:,0:-w0] = x[k:k+1,:, 0:hc-h, w+w0:w]
                p[k:k+1,:,0:h-h0,-w0:] = x[k:k+1,:, h0:h, 0:wc]
                p[k:k+1,:,h-h0:,-w0:] = x[k:k+1,:, 0:hc-h, 0:wc]
            # 8
            elif h0 >=h-size and (w0<w-size and w0>0):
                p[k:k+1,:,0:h-h0,:] = x[k:k+1,:, h0:h,w0:wc]
                p[k:k+1,:,h-h0:,:] = x[k:k+1,:, 0:hc-h, w0:wc]
            # 9
            elif h0 >=h-size and w0 >=w-size:
                p[k:k+1,:,0:h-h0,0:w-w0] = x[k:k+1,:, h0:h, w0:w]
                p[k:k+1,:,h-h0:,0:w-w0] = x[k:k+1,:, 0:hc-h, w0:w]
                p[k:k+1,:,0:h-h0,w-w0:] = x[k:k+1,:, h0:h, 0:wc-w]
                p[k:k+1,:,h-h0:,w-w0:] = x[k:k+1,:, 0:hc-h, 0:wc-w]

    else:
        # 1
        if h0<=0 and w0<=0:
            p[:,:,0:-h0,0:-w0] = x[:,:, h+h0:h, w+w0:w]
            p[:,:,-h0:,0:-w0] = x[:,:, 0:hc, w+w0:w]
            p[:,:,0:-h0,-w0:] = x[:,:, h+h0:h, 0:wc]
            p[:,:,-h0:,-w0:] = x[:,:, 0:hc, 0:wc]
        # 2
        elif h0<=0 and (w0<w-size and w0>0):
            p[:,:,0:-h0,:] = x[:,:, h+h0:h,w0:wc]
            p[:,:,-h0:,:] = x[:,:, 0:hc, w0:wc]
        # 3
        elif h0<=0 and w0 >=w-size:
            p[:,:,0:-h0,0:w-w0] = x[:,:, h+h0:h, w0:w]
            p[:,:,-h0:,0:w-w0] = x[:,:, 0:hc, w0:w]
            p[:,:,0:-h0,w-w0:] = x[:,:, h+h0:h, 0:wc-w]
            p[:,:,-h0:,w-w0:] = x[:,:, 0:hc, 0:wc-w]

        # 4
        elif (h0>0 and h0<h-size) and w0<=0:
            p[:,:,:,0:-w0] = x[:,:, h0:hc, w+w0:w]
            p[:,:,:,-w0:] = x[:,:, h0:hc, 0:wc]
        # 5
        elif (h0>0 and h0<h-size) and (w0<w-size and w0>0):
            p = x[:,:, h0:hc, w0:wc]
        # 6
        elif (h0>0 and h0<h-size) and w0 >=w-size:
            p[:,:,:,0:w-w0] = x[:,:, h0:hc, w0:w]
            p[:,:,:,w-w0:] = x[:,:, h0:hc, 0:wc-w]

        # 7
        elif h0 >=h-size and w0<=0:
            p[:,:,0:h-h0,0:-w0] = x[:,:, h0:h, w+w0:w]
            p[:,:,h-h0:,0:-w0] = x[:,:, 0:hc-h, w+w0:w]
            p[:,:,0:h-h0,-w0:] = x[:,:, h0:h, 0:wc]
            p[:,:,h-h0:,-w0:] = x[:,:, 0:hc-h, 0:wc]
        # 8
        elif h0 >=h-size and (w0<w-size and w0>0):
            p[:,:,0:h-h0,:] = x[:,:, h0:h,w0:wc]
            p[:,:,h-h0:,:] = x[:,:, 0:hc-h, w0:wc]
        # 9
        elif h0 >=h-size and w0 >=w-size:
            p[:,:,0:h-h0,0:w-w0] = x[:,:, h0:h, w0:w]
            p[:,:,h-h0:,0:w-w0] = x[:,:, 0:hc-h, w0:w]
            p[:,:,0:h-h0,w-w0:] = x[:,:, h0:h, 0:wc-w]
            p[:,:,h-h0:,w-w0:] = x[:,:, 0:hc-h, 0:wc-w]


    del x

    return p


# def tileshift_crop(x, size):

#     # tile 
#     x = mycrop(x, size)

#     x = 