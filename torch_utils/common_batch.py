import torch as th
import numpy as np
from PIL import Image

eps = 1e-5

def save_img(image, path,gamma=True):
    if gamma:
        image = image**(1/2.2)
    image = np.clip(image*255 ,0, 255)
    image_pil = Image.fromarray(image.astype(np.uint8))
    image_pil.save(path)

def vec(x, y, z):
    if x.dim()==2: # [h, w]
        return th.stack((x, y, z), dim=0) # [3, h, w]
    elif x.dim()==3: # [b, h, w]
        return th.stack((x, y, z), dim=1) # [b, 3, h, w]
    elif x.dim()==4: # [b, k, h, w]
        return th.stack((x, y, z), dim=2) # [b, k, 3, h, w]


def getx(v): 
    if v.dim()==3:
        return v[0,...] 
    elif v.dim()==4:
        return v[:, 0,...] 
    elif v.dim()==5:
        return v[:, :, 0:1,...] 


def gety(v): 
    if v.dim()==3:
        return v[1,...] 
    elif v.dim()==4:
        return v[:, 1,...] 
    elif v.dim()==5:
        return v[:, :, 1:2,...] 

def getz(v): 
    if v.dim()==3:
        return v[2,...] 
    elif v.dim()==4:
        return v[:, 2,...]         
    elif v.dim()==5:
        return v[:, :, 2:3,...] 

def normalize(v):
    if v.dim()==3:
        s = th.sqrt((v ** 2).sum(0)) + eps
        return v / vec(s, s, s)
    elif v.dim()==4:
        s = th.sqrt((v ** 2).sum(1)) + eps
        return v / vec(s, s, s)       
    elif v.dim()==5:
        s = th.sqrt((v ** 2).sum(2)) + eps
        return v / vec(s, s, s) 


def dot(a, b):
    if a.dim()==5:
        return (a * b).sum(2, keepdim=True)
    elif a.dim()==3:
        return (a * b).sum(0, keepdim=True)


def schlick(cos, f0):
  return f0 + (1 - f0) * (1 - cos)**5


def ggx(cos_h, alpha):
    c2 = cos_h**2
    a2 = alpha**2
    den = c2 * a2 + (1 - c2)
    return a2 / (np.pi * den**2 + 1e-6)


def img2th(img):
    t = th.tensor(img)
    if len(t.shape) == 3: t = t.permute((2, 0, 1))
    return t


def th2img(t):
    return t.detach().squeeze().permute((1, 2, 0)).cpu().numpy()


def getTexPos(res, size, device):
    x = th.arange(res, dtype=th.float32)
    x = ((x + 0.5) / res - 0.5) * size

    # surface positions,
    y, x = th.meshgrid((x, x))
    z = th.zeros_like(x)
    pos = th.stack((x, -y, z), 0).to(device)

    return pos



def get_cam_dir(camera_pos, surface_pos):
    v = camera_pos.unsqueeze(1).unsqueeze(1) - surface_pos
    cam_dir = normalize(v)
    dist_sq = (v**2).sum(0)
    return cam_dir, dist_sq


class Frame:
    def __init__(self, v, device):
        # assume v is normalized
        self.normal = v.to(device)

        # assume v[2] is non-zero (typically positive)
        self.bitangent = normalize(vec(getz(self.normal), th.zeros_like(getz(v)), -getx(v))).to(device)
        self.tangent = th.cross(self.normal, self.bitangent, dim=0 if v.dim()==3 else 1).to(device)

    def to_world(self, v):

        assert v.dim()==5, "dimension error in frame"

        b,k,_,_,_ = v.shape

        tangent = self.tangent.unsqueeze(1).repeat(1, k, 1, 1, 1)
        bitangent = self.bitangent.unsqueeze(1).repeat(1, k, 1, 1, 1)
        normal = self.normal.unsqueeze(1).repeat(1, k, 1, 1, 1)

        return  tangent * getx(v) + bitangent * gety(v) + normal * getz(v)
        # return  bitangent
        # return  getx(v) + gety(v) + getz(v)





# in numpy
def bump_to_normal(bump, scale, size):
    n = bump.shape[0]

    # compute horizontal differences, i.e. dx(i,j) = bump(i,j+1) - bump(i,j-1) with wraparound
    dx = np.zeros_like(bump)
    for j in range(n):
        dx[:, j] = bump[:, (j+1) % n] - bump[:, (j-1) % n]

    # compute vertical differences, i.e. dy(i,j) = bump(i-1,j) - bump(i+1,j) with wraparound
    dy = np.zeros_like(bump)
    for i in range(n):
        dy[i, :] = bump[(i-1) % n, :] - bump[(i+1) % n, :]

    # normalize by pixel size
    pixSize = size / n
    dx /= 2 * pixSize
    dy /= 2 * pixSize

    # create and normalize the normals
    normal = np.dstack((-dx, -dy, np.ones_like(bump)))
    norms = np.sqrt((normal ** 2).sum(2))
    norms = np.dstack([norms] * 3)
    normal /= norms

    return normal


#[B,c,H,W]
def height_to_normal(img_in, size, intensity=0.2):
    """Atomic function: Normal (https://docs.substance3d.com/sddoc/normal-172825289.html)

    Args:
        img_in (tensor): Input image.
        mode (str, optional): 'tangent space' or 'object space'. Defaults to 'tangent_space'.
        normal_format (str, optional): 'gl' or 'dx'. Defaults to 'dx'.
        use_input_alpha (bool, optional): Use input alpha. Defaults to False.
        use_alpha (bool, optional): Enable the alpha channel. Defaults to False.
        intensity (float, optional): Normalized height map multiplier on dx, dy. Defaults to 1.0/3.0.
        max_intensity (float, optional): Maximum height map multiplier. Defaults to 3.0.

    Returns:
        Tensor: Normal image.
    """
    # grayscale_input_check(img_in, "input height field")
    assert img_in.shape[1]==1, 'should be grayscale image'

    def roll_row(img_in, n):
        return img_in.roll(n, 2)

    def roll_col(img_in, n):
        return img_in.roll(n, 3)

    def norm(vec): #[B,C,W,H]
        vec = vec.div(vec.norm(2.0, 1, keepdim=True))
        return vec

    img_size = img_in.shape[2]
    
    img_in = img_in*intensity

    dx = (roll_col(img_in, 1) - roll_col(img_in, -1))
    dy = (roll_row(img_in, 1) - roll_row(img_in, -1))
    
    pixSize = size / img_in.shape[-1]
    dx /= 2 * pixSize
    dy /= 2 * pixSize

    img_out = th.cat((dx, -dy, th.ones_like(dx)), 1)
    img_out = norm(img_out)
    # img_out = img_out / 2.0 + 0.5 #[-1,1]->[0,1]
    
    return img_out

def read_all_envs(path, device):
    all_env = []
    for env in os.listdir(path):
        single_env  = read_exr(os.path.join(path, env))
        single_env = th.from_numpy(single_env).to(device).permute(2,0,1).unsqueeze(0)
        all_env.append(single_env)
    all_env = th.cat(all_env, dim=0)
    return all_env


def read_exr(filename):
    img = OpenEXR.InputFile(filename)
    header = img.header()
    dw = header['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    def chan(c):
        s = img.channel(c, pt)
        arr = np.fromstring(s, dtype=np.float32)
        arr.shape = size[1], size[0]
        return arr

    # single-channel file
    channels = list(header['channels'])
    # print(channels)
    if len(channels) == 1:
        return chan(channels[0])
    elif len(channels) == 3:
        return np.dstack([chan('R'), chan('G'), chan('B')])
    elif len(channels) == 4:
        return np.dstack([chan('R'), chan('G'), chan('B'), chan('A')])
    else:
        assert False

## self-defined grid_sample
def grid_sample(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW-1);
    iy = ((iy + 1) / 2) * (IH-1);
    with th.no_grad():
        ix_nw = th.floor(ix);
        iy_nw = th.floor(iy);
        ix_ne = ix_nw + 1;
        iy_ne = iy_nw;
        ix_sw = ix_nw;
        iy_sw = iy_nw + 1;
        ix_se = ix_nw + 1;
        iy_se = iy_nw + 1;

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)
    
    with th.no_grad():
        th.clamp(ix_nw, 0, IW-1, out=ix_nw)
        th.clamp(iy_nw, 0, IH-1, out=iy_nw)

        th.clamp(ix_ne, 0, IW-1, out=ix_ne)
        th.clamp(iy_ne, 0, IH-1, out=iy_ne)
 
        th.clamp(ix_sw, 0, IW-1, out=ix_sw)
        th.clamp(iy_sw, 0, IH-1, out=iy_sw)
 
        th.clamp(ix_se, 0, IW-1, out=ix_se)
        th.clamp(iy_se, 0, IH-1, out=iy_se)

    image = image.view(N, C, IH * IW)


    nw_val = th.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = th.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = th.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = th.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val
