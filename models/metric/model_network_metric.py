import matplotlib
import numpy as np
import math
import random
import time

import torch
import torch.nn.functional as F

from torch.nn import Linear
from torch.nn import LayerNorm, InstanceNorm1d
from torch import Tensor
from torch.nn import Conv3d
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable, grad
from torch.cuda.amp import autocast
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
#from EikoNet import database as db
#from models import data_mlp as db
import copy

import matplotlib
import matplotlib.pylab as plt

from timeit import default_timer as timer

torch.backends.cudnn.benchmark = True

#3dnewserver
def sigmoid_out(input):
 
    return torch.sigmoid(0.1*input)

class Sigmoid_out(torch.nn.Module):
    def __init__(self):
        
        super().__init__() 

    def forward(self, input):
       
        return sigmoid_out(input) 

def to_euler_angles(rotvec, seq='xyz', degrees=False):
    """Convert an axis-angle rotation vector to Euler angles via SciPy.

    Uses ``scipy.spatial.transform.Rotation`` -- NOT differentiable (goes through
    NumPy), so use it only for preprocessing / evaluation, never inside the loss.

    Args:
        rotvec: (..., 3) axis-angle / rotation vector (direction = axis,
            magnitude = angle in radians).  NOTE: this codebase stores rotation
            coords normalized by 2*pi, so multiply by ``2*np.pi`` first if you
            pass stored coords.
        seq: Euler convention string for ``Rotation.as_euler`` (lowercase =
            intrinsic, uppercase = extrinsic), e.g. 'xyz', 'ZYX'.
        degrees: if True, return degrees instead of radians.

    Returns:
        (..., 3) array of Euler angles.  Returns a torch.Tensor (on the input's
        device/dtype) if given one, otherwise a NumPy array.
    """
    from scipy.spatial.transform import Rotation

    is_tensor = isinstance(rotvec, torch.Tensor)
    if is_tensor:
        arr = rotvec.detach().cpu().numpy()
    else:
        arr = np.asarray(rotvec)

    flat = arr.reshape(-1, 3)
    euler = Rotation.from_rotvec(flat).as_euler(seq, degrees=degrees)
    euler = euler.reshape(arr.shape)

    if is_tensor:
        return torch.as_tensor(euler, dtype=rotvec.dtype, device=rotvec.device)
    return euler


def axis_angle_to_euler(rotvec, eps=1e-8):
    """Differentiable axis-angle -> intrinsic ZYX Euler angles (radians).

    Use this one inside the network (autograd flows through it), unlike the
    SciPy ``to_euler_angles`` which detaches.

    Args:
        rotvec: (..., 3) axis-angle rotation vector in RADIANS (direction=axis,
            magnitude=angle).
    Returns:
        (..., 3) Euler angles (roll_x, pitch_y, yaw_z) for
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    theta = torch.norm(rotvec, dim=-1, keepdim=True)
    k = rotvec / (theta + eps)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    th = theta[..., 0]
    s, c = torch.sin(th), torch.cos(th)
    C = 1.0 - c

    # Rodrigues' rotation matrix entries (R = I + sin*K + (1-cos)*K^2).
    r00 = c + kx * kx * C
    r10 = ky * kx * C + kz * s
    r11 = c + ky * ky * C
    r12 = ky * kz * C - kx * s
    r20 = kz * kx * C - ky * s
    r21 = kz * ky * C + kx * s
    r22 = c + kz * kz * C

    cy = torch.sqrt(r00 * r00 + r10 * r10)          # cos(pitch)
    pitch = torch.atan2(-r20, cy)
    yaw = torch.atan2(r10, r00)
    roll = torch.atan2(r21, r22)

    # Gimbal-lock fallback (cy ~ 0).
    gimbal = cy < 1e-6
    roll = torch.where(gimbal, torch.atan2(-r12, r11), roll)
    yaw = torch.where(gimbal, torch.zeros_like(yaw), yaw)

    return torch.stack([roll, pitch, yaw], dim=-1)


def euler_to_rotation_features(euler, mode='6d'):
    """Rotation-matrix entries as trig products of ZYX Euler angles.

    These are the *continuous* combinations (sin a cos b, sin a sin b sin c, ...)
    that make up R = Rz(yaw) @ Ry(pitch) @ Rx(roll).  Feed the result DIRECTLY
    into the encoder -- do NOT wrap it in sin/cos Fourier features (the entries
    are already cosines in [-1,1], not angles).

    Args:
        euler: (..., 3) = (roll_x, pitch_y, yaw_z) in radians.
        mode: '6d' -> first two columns (6 entries, minimal continuous rep);
              '9d' -> full rotation matrix (9 entries).
    Returns:
        (..., 6) or (..., 9) differentiable feature tensor.
    """
    a, b, g = euler[..., 0], euler[..., 1], euler[..., 2]   # roll, pitch, yaw
    sa, ca = torch.sin(a), torch.cos(a)
    sb, cb = torch.sin(b), torch.cos(b)
    sg, cg = torch.sin(g), torch.cos(g)

    # First two columns of R = Rz(g) Ry(b) Rx(a).
    r00 = cg * cb
    r10 = sg * cb
    r20 = -sb
    r01 = cg * sb * sa - sg * ca
    r11 = sg * sb * sa + cg * ca
    r21 = cb * sa
    if mode == '6d':
        return torch.stack([r00, r10, r20, r01, r11, r21], dim=-1)

    r02 = cg * sb * ca + sg * sa
    r12 = sg * sb * ca - cg * sa
    r22 = cb * ca
    return torch.stack([r00, r10, r20, r01, r11, r21, r02, r12, r22], dim=-1)


class NN(torch.nn.Module):
    
    def __init__(self, device, dim ,B):#10
        super(NN, self).__init__()
        self.dim = dim

        h_size = 256

        # Split the configuration into a translational route (first half of the
        # dims: x,y,z) and a rotational route (last half: rx,ry,rz).  Each route
        # owns its own Fourier frequencies (a column-slice of B) and a completely
        # independent set of weights -- pe_gate, encoder, gate, encoder_norm.
        half = self.dim // 2
        self.B_trans = B[:, :half].T.to(device)   # (half, input_size)
        self.B_rot   = B[:, half:].T.to(device)   # (half, input_size)

        input_size = B.shape[0]

        self.scale = 10



        self.act = torch.nn.Softplus(beta=self.scale)
        self.actout = Sigmoid_out()
        self.nl1=2

        # Translational route (operates on the first `half` coordinates).
        self.encoder_t, self.gate_t, self.pe_gate_t, self.encoder_norm_t = self._build_route(h_size)
        # Rotational route (operates on the last `half` coordinates).
        self.encoder_r, self.gate_r, self.pe_gate_r, self.encoder_norm_r = self._build_route(h_size)


        # rot6_freq_scale = 2.0
        # self.register_buffer(
        #    'B_rot6', (rot6_freq_scale * torch.normal(0, 1, size=(6, 128))).to(device))



        self.fuse_len = 3
        self.fuse = torch.nn.ModuleList()
        self.fuse.append(Linear(2*h_size, h_size))   # 512 -> 256
        for i in range (self.fuse_len):
            self.fuse.append(Linear(h_size, h_size))     
            self.fuse.append(Linear(h_size, h_size))     

    def _build_route(self, h_size):
        """Build one independent embedding route (same architecture as the
        original single route).  Returns (encoder, gate, pe_gate, encoder_norm)."""
        encoder = torch.nn.ModuleList()
        encoder.append(Linear(252, h_size))
        for i in range(0, 3*self.nl1):
            encoder.append(Linear(h_size, h_size))
        encoder.append(Linear(h_size, h_size))

        gate = torch.nn.ModuleList()
        for i in range(self.nl1):
            gate.append(Linear(1, 1))

        pe_gate = torch.nn.ModuleList()
        pe_gate.append(Linear(h_size, h_size))
        pe_gate.append(Linear(h_size, h_size))

        encoder_norm = InstanceNorm1d(h_size)
        return encoder, gate, pe_gate, encoder_norm


    #'''
    def init_weights(self, m):
        
        if type(m) == torch.nn.Linear:
            stdv = np.sqrt(2.0 / (m.weight.size(0)+m.weight.size(1)))
            torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=stdv, a=-2.0*stdv, b=2.0*stdv)
            m.bias.data.fill_(0)
        
        for i in range(self.nl1):
            self.gate_t[i].weight.data.fill_(0)
            self.gate_t[i].bias.data.fill_(0)
            self.gate_r[i].weight.data.fill_(0)
            self.gate_r[i].bias.data.fill_(0)

        
   
    #'''
    def input_mapping(self, x, B):
        w = 2.*np.pi*B
        x_proj = x @ w
        #x_proj = (2.*np.pi*x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)    #  2*len(B)

    def rotation_input_mapping(self, x):
        # Replacement for input_mapping on the rotation route. `x` is the stored
        # rotation coords (axis-angle rotvec normalized by 2*pi). Convert to the
        # continuous 6D rotation representation, then apply RANDOM FOURIER
        # FEATURES over it (B_rot6 mixes the 6 entries into 128 frequencies).
        # This keeps continuity (6D) AND high-frequency richness (Fourier),
        # unlike a plain linear lift (underfit) or plain sin/cos of the entries.
        euler = axis_angle_to_euler(x * (2.0 * np.pi))
        feats = euler_to_rotation_features(euler, mode='6d')   # (N, 6), in [-1,1]
        x_proj = feats @ self.B_rot6                           # (N, 128)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # (N, 256)
    
    def lip_norm(self, w):
        absrowsum = torch.sqrt(torch.sum ( w**2 , dim =1)).detach()
        scale = 1 + 1e-5 - self.act(1 - 1 / absrowsum)
        return w * scale.unsqueeze(1) #[: , None ]
    

    def to_translational_embedding(self, x, B, pe_gate, gate, encoder, encoder_norm):
        x = self.input_mapping(x, B)

        w = self.lip_norm(pe_gate[0].weight)
        b = pe_gate[0].bias
        u = torch.sin(x@w.T+b)

        w = self.lip_norm(pe_gate[1].weight)
        b = pe_gate[1].bias
        v = torch.sin(x@w.T+b)

        for ii in range(0,self.nl1):
            x_tmp = x

            w = self.lip_norm(encoder[3*ii+1].weight)
            b = encoder[3*ii+1].bias
            y = x@w.T+b

            x  = u*torch.sin(y)+v*(1-torch.sin(y))

            w = self.lip_norm(encoder[3*ii+2].weight)
            b = encoder[3*ii+2].bias
            y = x@w.T+b

            x  = u*torch.sin(y)+v*(1-torch.sin(y))

            w = self.lip_norm(encoder[3*ii+3].weight)
            b = encoder[3*ii+3].bias
            y = x@w.T+b

            weight = torch.sigmoid(0.1*gate[ii].weight)
            x  = (1-weight)*x_tmp+(weight)*torch.sin(y)

        w = self.lip_norm(encoder[-1].weight)
        b = encoder[-1].bias

        y = x@w.T+b
        y = encoder_norm(y)

        return y


    def to_rotational_embedding(self, x, B, pe_gate, gate, encoder, encoder_norm):

        #x = self.rotation_input_mapping(x)
        x = self.input_mapping(x, B)

        w = self.lip_norm(pe_gate[0].weight)
        b = pe_gate[0].bias
        u = torch.sin(x@w.T+b)

        w = self.lip_norm(pe_gate[1].weight)
        b = pe_gate[1].bias
        v = torch.sin(x@w.T+b)

        for ii in range(0,self.nl1):
            x_tmp = x

            w = self.lip_norm(encoder[3*ii+1].weight)
            b = encoder[3*ii+1].bias
            y = x@w.T+b

            x  = u*torch.sin(y)+v*(1-torch.sin(y))

            w = self.lip_norm(encoder[3*ii+2].weight)
            b = encoder[3*ii+2].bias
            y = x@w.T+b

            x  = u*torch.sin(y)+v*(1-torch.sin(y))

            w = self.lip_norm(encoder[3*ii+3].weight)
            b = encoder[3*ii+3].bias
            y = x@w.T+b

            weight = torch.sigmoid(0.1*gate[ii].weight)
            x  = (1-weight)*x_tmp+(weight)*torch.sin(y)

        w = self.lip_norm(encoder[-1].weight)
        b = encoder[-1].bias

        y = x@w.T+b
        y = encoder_norm(y)

        return y


    def out(self, coords):
        
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        size = coords.shape[0]
        x0 = coords[:,:self.dim]
        x1 = coords[:,self.dim:]
        
        x = torch.vstack((x0,x1))


        half = 3
        x_trans = x[:, :half]
        x_rot   = x[:, half:]

        translational_embedding = self.to_translational_embedding(
            x_trans, self.B_trans, self.pe_gate_t, self.gate_t, self.encoder_t, self.encoder_norm_t)
        rotational_embedding = self.to_rotational_embedding(
            x_rot, self.B_rot, self.pe_gate_r, self.gate_r, self.encoder_r, self.encoder_norm_r)

        y = torch.cat([translational_embedding, rotational_embedding], dim=-1)


        residual_connection = y@self.fuse[0].weight.T + self.fuse[0].bias
        for i in range (self.fuse_len):
            y1 = residual_connection@self.fuse[2*i+1].weight.T + self.fuse[2*i+1].bias
            y1 = self.act(y1)
            y2 = y1@self.fuse[2*i+2].weight.T + self.fuse[2*i+2].bias
            
            residual_connection = residual_connection + y2
            residual_connection = self.act(residual_connection)


        # w = self.lip_norm(self.fuse[0].weight)
        # y = y@w.T + self.fuse[0].bias
        # w = self.lip_norm(self.fuse[1].weight)
        # y = y@w.T + self.fuse[1].bias



        x0 = residual_connection[:size,...]
        x1 = residual_connection[size:,...]

        #OURS
        x = torch.sqrt((x0-x1)**2+1e-6)
        x = x.view(x.shape[0],-1,16)
        x = (torch.logsumexp(10*x, 2)-np.log(16))/10
        x = 0.2*(torch.sum(x,dim=1,keepdim=True))
        
        #test
        #x = torch.exp(x)
        #L1
        # x = 0.01*torch.norm(x0-x1,p=1,dim=1).unsqueeze(1)

        
        
        return x, None, coords
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input

        output, coords = self.out(coords)
        return output, coords
