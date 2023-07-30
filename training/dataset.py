# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import random

import torchvision.transforms.functional as F
from torch_utils.render import pos_to_crop


try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        d_res = None,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

        
        self.d_res = d_res 

    def _get_raw_labels(self):
        if self._raw_labels is None:
            print('self._use_labels: ', self._use_labels)
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            print('self._raw_labels.dtype: ', self._raw_labels.dtype)

            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1 or self._raw_labels.ndim == 2
                assert np.all(self._raw_labels >= 0)
            print('_raw_labels: ',self._raw_labels.shape)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]


        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        # print('self._get_raw_labels(): ',self._get_raw_labels().shape)
        label = self._get_raw_labels()[self._raw_idx[idx]]
        # print("get label: ", label, self._raw_idx)
        # if label.dtype == np.int64:
        #     onehot = np.zeros(self.label_shape, dtype=np.float32)
        #     onehot[label] = 1
        #     label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        if self.d_res:
            return self.d_res
        else:
            return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        col_camli       = False,# collocated camera and light        
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        # if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
        #     raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------

class MaterialsDataset_nocond(Dataset):
    ''' 
    material dataset no cond
    '''
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        col_camli       = False,# collocated camera and light        
        light_mode      = 'uni',# collocated camera and light        
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        # if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
        #     raise IOError('Image files do not match the specified resolution')

        self.col_camli = col_camli
        self.light_mode = light_mode

        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)


    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]

        # rotation
        # image = np.rot90(image, random.randint(0,4), (1,2))
            
        return image.copy(), self.get_label(idx)


    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        # print('img: ', image.shape)
        _,h,_ = image.shape

        H = image[0:1,:,0:h]
        D = image[:,:,h:2*h]
        R = image[0:1,:,2*h:3*h]


        image = np.concatenate((H,D,R), axis=0)

        # print('..............image:', image.shape, 256)

        # image = np.array(PIL.Image.fromarray(image).resize((256, 256)))
        # print('..............image:', image.shape)

        return image

    def _load_raw_labels(self):
        return None

#----------------------------------------------------------------------------

class MaterialsDataset(Dataset):
    ''' 
    conditional material dataset without fixed light
    '''
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        col_camli       = False,# collocated camera and light        
        light_mode      = 'uni',# collocated camera and light        
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        # if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
        #     raise IOError('Image files do not match the specified resolution')

        self.col_camli = col_camli
        self.light_mode = light_mode

        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)


    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]

        # rotation
        # image = np.rot90(image, random.randint(0,4), (1,2))
            
        return image.copy(), self.get_label(idx)


    def get_label(self, idx):
        # random sampled light
        label = self._load_raw_labels()
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._load_raw_labels().copy()
        return d


    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        # print('img: ', image.shape)
        _,h,_ = image.shape

        H = image[0:1,:,0:h]
        D = image[:,:,h:2*h]
        # R = image[0:1,:,2*h:3*h]
        # R = np.random.randint(80,170, image[0:1,:,2*h:3*h].shape, dtype=np.uint8)

        # for debugging limit roughness < 0.6
        tmp_r = image[0:1,:,2*h:3*h].mean()
        if tmp_r >=229:
            exp = 4
        elif tmp_r < 229 and tmp_r >= 204:
            exp = 3
        elif tmp_r < 204 and tmp_r >= 160:
            exp = 2
        elif tmp_r < 160 and tmp_r >= 103:
            exp = 1.5
        else:
            exp = 1

        R = (image[0:1,:,2*h:3*h]/255.)**exp
        R = (R*255.0).astype(np.uint8)
        
        image = np.concatenate((H,D,R), axis=0)

        return image

    def _load_raw_labels(self):

        light = self._rand_light()
        return light


    def _rand_light(self):


        if self.light_mode=='uni':
            # u_1 = np.random.uniform(0,0.8,(1))
            u_1 = np.random.uniform(0,0.6,(1))
        elif self.light_mode=='randn':
            tmp = 0.3 if not self.col_camli else 0.1
            u_1 = np.abs(np.random.normal(0,tmp,(1))).clip(0,0.9)

        u_2 = np.random.uniform(0,1,(1))
        theta = 2*np.pi*u_2

        r = np.sqrt(u_1)
        z = np.sqrt(1-r*r)
        x = r*np.cos(theta)
        y = r*np.sin(theta)

        light_pos = np.concatenate((x,y,z),axis=0) * 4.
        print('light_pos: ', light_pos)

        return light_pos



    #-------- hard code
    @property
    def has_onehot_labels(self):
        return self._load_raw_labels().dtype == np.int64

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._load_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape
        return list(self._label_shape)


    @property
    def label_dim(self):
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    def get_label_std(self):
        return 0

    # @property
    # def resolution(self):
    #     assert len(self.image_shape) == 3 # CHW
    #     return self.image_shape[1]

    @property
    def image_shape(self):
        tmp = self._raw_shape[-1]
        return [3,tmp,tmp]

#----------------------------------------------------------------------------

class MaterialsDataset_FixedLight(Dataset):
    ''' 
    conditional material dataset with fixed light
    '''
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        col_camli       = False,# collocated camera and light
        light_mode      = 'uni',# uniform or randn light pos
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        # if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
        #     raise IOError('Image files do not match the specified resolution')
        self.col_camli = col_camli

        self.light_mode = light_mode  # uni or rand

        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    def _get_raw_labels(self):
        if self._raw_labels is None:
            # print('self._use_labels: ', self._use_labels)
            self._raw_labels = self._load_raw_labels()
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
            # print('_raw_labels: ',self._raw_labels.shape)
        return self._raw_labels


    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)


    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        # print('get item: ', self.get_label(idx).shape)
        return image.copy(), self.get_label(idx)


    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        # print('img: ', image.shape)

        _,h,_ = image.shape
        H = image[0:1,:,0:h]
        D = image[:,:,h:2*h]
        R = image[0:1,:,2*h:3*h]
        image = np.concatenate((H,D,R), axis=0)
        return image

    def _load_raw_labels(self):

        light = self._rand_light().astype(np.float32)
        print(' ------- load raw label -------', light.shape)
        return light


    def _rand_light(self):

        # assert self.light_mode=='uni' or self.light_mode=='randn', "light mode incorrect !!!"

        if self.light_mode=='uni':
            u_1 = np.random.uniform(0,0.8,(self._raw_shape[0],1))
            # u_1 = np.random.uniform(0,0.1,(self._raw_shape[0],1))
        elif self.light_mode=='randn':
            tmp = 0.3 if not self.col_camli else 0.1
            u_1 = np.abs(np.random.normal(0,tmp,(self._raw_shape[0],1))).clip(0,0.9)

        u_2 = np.random.uniform(0,1,(self._raw_shape[0],1))
        theta = 2*np.pi*u_2

        r = np.sqrt(u_1)
        z = np.sqrt(1-r*r)
        x = r*np.cos(theta)
        y = r*np.sin(theta)

        # print('x: ', x.shape)
        # print('y: ', y.shape)
        # print('z: ', z.shape)
        light = np.concatenate((x,y,z),axis=-1) * 4.

        print('light: ', light)

        return light


    # hard code
    # @property
    # def has_onehot_labels(self):
    #     return self._load_raw_labels().dtype == np.int64

    # @property
    # def label_shape(self):
    #     if self._label_shape is None:
    #         raw_labels = self._load_raw_labels()
    #         if raw_labels.dtype == np.int64:
    #             self._label_shape = [int(np.max(raw_labels)) + 1]
    #         else:
    #             self._label_shape = raw_labels[1:]
    #     return list(self._label_shape)


    # @property
    # def label_dim(self):
    #     # print('label_dim: ', self.label_shape)
    #     return self.label_shape[0]

    # @property
    # def has_labels(self):
    #     return True

    # def get_label_std(self):
    #     return 0

    # @property
    # def resolution(self):
    #     assert len(self.image_shape) == 3 # CHW
    #     return self.image_shape[1]

    @property
    def image_shape(self):
        tmp = self._raw_shape[-1]
        return [3,tmp,tmp]

#----------------------------------------------------------------------------

class RealImageDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        col_camli       = False,# collocated camera and light        
        debug_mode      = False,# debug mode or not        
        light_mode      = 'uni',# collocated camera and light        
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        self.hue_aug = False

        if "carpaint" in self._path:
            self.hue_aug = True
            print("self.hue_aug: ", self.hue_aug)

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        # if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
        #     raise IOError('Image files do not match the specified resolution')

        self.col_camli = col_camli
        self.light_mode = light_mode

        self.crop_size = 256

        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)


    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]

        label = self.get_label(idx)


        # rotation
        image = np.rot90(image, random.randint(0,4), (1,2))

        # crop
        # c_x = int(128 + label[1]*256)
        # c_y = int(128 - label[0]*256)
        # image = image[:, c_x:c_x+256, c_y:c_y+256]

        # light to crop 
        ch,cw = pos_to_crop(label)

        image = image[:, ch:ch+256, cw:cw+256]
        # print("img in datloader", image.shape, image.dtype)

        # print('light: ', label)
        return image.copy(), label


    def get_label(self, idx):
        # random sampled light
        label = self._load_raw_labels()
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._load_raw_labels().copy()
        return d

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            # if pyspng is not None and self._file_ext(fname) == '.png':
            #     image = pyspng.load(f.read())
            # else:

            # hue augment if needed
            if self.hue_aug:
                image = PIL.Image.open(f)
                image = F.adjust_hue(image, random.random()-0.5)
                image = np.array(image)

            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW

        return image

    def _load_raw_labels(self):

        c_h,c_w = self._rand_crop()
        c_h = c_h - 128
        c_w = c_w - 128
        FOV = 4
        light = np.array([-c_w/256., c_h/256., 1])*FOV
        return light


    def _rand_crop(self):
        c_h, c_w = np.random.randint(0,256), np.random.randint(0,256)
        # c_h, c_w = 256, 256
        return c_h, c_w 


    #-------- hard code
    @property
    def has_onehot_labels(self):
        return self._load_raw_labels().dtype == np.int64

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._load_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape
        return list(self._label_shape)


    @property
    def label_dim(self):
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    def get_label_std(self):
        return 0

    # @property
    # def resolution(self):
    #     assert len(self.image_shape) == 3 # CHW
    #     return self.image_shape[1]

    @property
    def image_shape(self):
        tmp = self._raw_shape[-1]
        return [3,tmp,tmp]

#----------------------------------------------------------------------------

# Real Image Dataset with label
class RealImageDataset_label(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        col_camli       = False,# collocated camera and light        
        light_mode      = 'uni',# collocated camera and light        
        d_res           = None,# collocated camera and light        
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None
        self.FOV = 4

        self.d_res = d_res
        self.ratio = 2048/(self.d_res*2) # ratio for downsampling labels
        print("self.d_res: ", self.d_res)
        print("self.ratio: ", self.ratio)

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        # print("all images: ", self._all_fnames)

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        print("dataloader image length: ", len(self._image_fnames))
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        # if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
        #     raise IOError('Image files do not match the specified resolution')


        super().__init__(name=name, raw_shape=raw_shape, d_res=self.d_res,**super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)



    def __getitem__(self, idx):

        """
        1) load image and label
        2) downsample image and label
        3) rotate
        4) label to light pos
        """

        # rotate label and image
        flip_x = random.randint(0,1)
        flip_y = random.randint(0,1)

        # crop
        # c_h, c_w = np.random.randint(0,self.d_res), np.random.randint(0,self.d_res)
        c_h, c_w = 128, 128

        # get image
        image = self._load_raw_image(self._raw_idx[idx], flip_x=flip_x, flip_y=flip_y, c_h=c_h, c_w=c_w)
        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8

        # get label (light pos)
        label = self.get_label(idx, flip_x=flip_x, flip_y=flip_y, c_h=c_h, c_w=c_w) #HW

        # print("img in datloader", image.shape, image.dtype)

        return image.copy(), label


    # def cropimg_light(self, img, label):

    #     # crop img
    #     # c_h, c_w = 0, 0

    #     # crop + label --> light pos
    #     li_x = (label[1] - c_w)/self.d_res - 0.5
    #     li_y = - (c_h - label[0])/self.d_res + 0.5
    #     light = np.array([li_x, li_y, 1]) * self.FOV

    #     return img, light

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() 

        return self._raw_labels



    def _load_raw_image(self, raw_idx, flip_x=0, flip_y=0, c_h=0, c_w=0):
        fname = self._image_fnames[raw_idx]
        # print("image name: ", fname)
        with self._open_file(fname) as f:
            image = PIL.Image.open(f)

            # downsample image if needed
            if image.size[0] != self.d_res*2:
                image = image.resize((self.d_res*2, self.d_res*2), resample=PIL.Image.LANCZOS)

            image = np.array(image)

        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW

        # rotate
        # if flip_x:
        #     image = image[:, :, ::-1] # CHW
        # if flip_y:
        #     image = image[:, ::-1, :] # CHW

        # crop
        image = image[:, c_h:c_h+self.d_res, c_w:c_w+self.d_res]


        return image

    def _load_raw_labels(self):
        # fname = 'dataset.json'
        # if fname not in self._all_fnames:
        #     return None
        # with self._open_file(fname) as f:
        #     labels = json.load(f)

        # labels = dict(labels)

        # labels = [labels[fname.split("/")[-1].split(".")[0]] for fname in self._image_fnames]
        # labels = np.array(labels).astype(np.float32)
        # labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])


        # c_h, c_w = np.random.randint(0,256), np.random.randint(0,256)
        # c_h = c_h - 128
        # c_w = c_w - 128
        FOV = 4
        labels = np.array([0.0, 0.0, 1])*FOV

        return labels


    def get_label(self, idx, flip_x=0, flip_y=0, c_h=0, c_w=0):

        # label = self._get_raw_labels()[self._raw_idx[idx]]
        label = self._load_raw_labels()

        label = label/self.ratio # recale light pos
        # print(f"label load2: {label}")

        # rotate
        if flip_x:
            label[1] = 2*self.d_res - label[1]
        if flip_y:
            label[0] = 2*self.d_res - label[0]

        # crop + label --> light pos
        li_x = (label[1] - c_w)/self.d_res - 0.5
        li_y = - (label[0]-c_h)/self.d_res + 0.5

        # print(f"data lix liy: {li_x}, {li_y}")
        light = np.array([li_x, li_y, 1]) * self.FOV

        return light.copy()

    @property
    def label_shape(self):
        return [3]

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

#----------------------------------------------------------------------------

# Real Image Dataset with label
class RealImageDataset_label2(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        col_camli       = False,# collocated camera and light        
        light_mode      = 'uni',# collocated camera and light        
        d_res           = None,# collocated camera and light        
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None
        self.FOV = 4

        self.d_res = d_res
        self.ratio = 2048/(self.d_res*2) # ratio for downsampling labels
        print("self.d_res: ", self.d_res)
        print("self.ratio: ", self.ratio)

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        # print("all images: ", self._all_fnames)

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        print("dataloader image length: ", len(self._image_fnames))
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        # if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
        #     raise IOError('Image files do not match the specified resolution')


        super().__init__(name=name, raw_shape=raw_shape, d_res=self.d_res,**super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)



    def __getitem__(self, idx):

        """
        1) load image and label
        2) downsample image and label
        3) rotate
        4) label to light pos
        """

        # rotate label and image
        flip_x = random.randint(0,1)
        flip_y = random.randint(0,1)

        # crop
        c_h, c_w = np.random.randint(0,self.d_res), np.random.randint(0,self.d_res)
        # c_h, c_w = 128, 128

        # get image
        image = self._load_raw_image(self._raw_idx[idx], flip_x=flip_x, flip_y=flip_y, c_h=c_h, c_w=c_w)
        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8

        # get label (light pos)
        label = self.get_label(idx, flip_x=flip_x, flip_y=flip_y, c_h=c_h, c_w=c_w) #HW

        # print("img in datloader", image.shape, image.dtype)

        return image.copy(), label


    # def cropimg_light(self, img, label):

    #     # crop img
    #     # c_h, c_w = 0, 0

    #     # crop + label --> light pos
    #     li_x = (label[1] - c_w)/self.d_res - 0.5
    #     li_y = - (c_h - label[0])/self.d_res + 0.5
    #     light = np.array([li_x, li_y, 1]) * self.FOV

    #     return img, light

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() 

        return self._raw_labels



    def _load_raw_image(self, raw_idx, flip_x=0, flip_y=0, c_h=0, c_w=0):

        fname = self._image_fnames[raw_idx]
        # print("image name: ", fname)
        with self._open_file(fname) as f:
            image = PIL.Image.open(f)

            # downsample image if needed
            if image.size[0] != self.d_res*2:
                image = image.resize((self.d_res*2, self.d_res*2), resample=PIL.Image.LANCZOS)

            image = np.array(image)

        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW

        # rotate
        if flip_x:
            image = image[:, :, ::-1] # CHW
        if flip_y:
            image = image[:, ::-1, :] # CHW

        # crop
        image = image[:, c_h:c_h+self.d_res, c_w:c_w+self.d_res]


        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)

        labels = dict(labels)

        labels = [labels[fname.split("/")[-1].split(".")[0]] for fname in self._image_fnames]
        labels = np.array(labels).astype(np.float32)
        # labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])


        # c_h, c_w = np.random.randint(0,256), np.random.randint(0,256)
        # c_h = c_h - 128
        # c_w = c_w - 128
        # FOV = 4
        # labels = np.array([-c_w/256., c_h/256., 1])*FOV
        # labels = np.random.rand(2)*2-1

        # c_h, c_w = np.random.randint(0,512), np.random.randint(0,512)
        # c_h = c_h - 256
        # c_w = c_w - 256
        # FOV = 4
        # labels = np.array([-c_w/256., c_h/256., 1])*FOV

        # labels = np.random.randint(0,self.d_res, size=(2))


        return labels


    def get_label(self, idx, flip_x=None, flip_y=None, c_h=None, c_w=None):

        # this is only for training to fetch all_gen_c
        if flip_x is None and c_h is None:
            # rotate label and image
            flip_x = random.randint(0,1)
            flip_y = random.randint(0,1)

            # crop
            c_h, c_w = np.random.randint(0,self.d_res), np.random.randint(0,self.d_res)

        label = self._get_raw_labels()[self._raw_idx[idx]]
        # label = self._load_raw_labels()

        label = label/self.ratio # recale light pos
        # print(f"label load2: {label}")

        # rotate
        if flip_x:
            label[1] = 2*self.d_res - label[1]
        if flip_y:
            label[0] = 2*self.d_res - label[0]

        # crop + label --> light pos
        li_x = (label[1] - c_w)/self.d_res - 0.5
        li_y = - (label[0]-c_h)/self.d_res + 0.5

        # print(f"data lix liy: {li_x}, {li_y}")
        label = np.array([li_x, li_y, 1]) * self.FOV

        return label.copy()

    # @property
    # def label_shape(self):
    #     return [3]

    # @property
    # def label_dim(self):
    #     assert len(self.label_shape) == 1
    #     return self.label_shape[0]

    #-------- hard code
    @property
    def has_onehot_labels(self):
        return self._load_raw_labels().dtype == np.int64

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = np.array([1, 1, 1]) # hardcode
            # if raw_labels.dtype == np.int64:
            #     self._label_shape = [int(np.max(raw_labels)) + 1]
            # else:
            self._label_shape = raw_labels.shape
        return list(self._label_shape)


    @property
    def label_dim(self):
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    def get_label_std(self):
        return 0

    # @property
    # def resolution(self):
    #     assert len(self.image_shape) == 3 # CHW
    #     return self.image_shape[1]

    @property
    def image_shape(self):
        tmp = self._raw_shape[-1]
        return [3,tmp,tmp]

#----------------------------------------------------------------------------