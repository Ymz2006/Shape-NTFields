import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable, grad
#import open3d as o3d

class _numpy2dataset(torch.utils.data.Dataset):
    def __init__(self, points, speed, normal, speed_angles, speed_dists,
                 trans_n, rot_n):
        # Creating identical pairs
        points    = Variable(Tensor(points))
        speed  = Variable(Tensor(speed))
        normal  = Variable(Tensor(normal))
        speed_angles  = Variable(Tensor(speed_angles))
        speed_dists = Variable(Tensor(speed_dists))
        # trans_n / rot_n: per-endpoint unit directions that *reduce* the
        # translational / rotational clearance the most (same (N, 2*dim) layout
        # as ``normal``).  Appended LAST so all existing column offsets stay put.
        trans_n = Variable(Tensor(trans_n))
        rot_n = Variable(Tensor(rot_n))
        self.data=torch.cat((points,speed,normal, speed_dists,speed_angles,
                             trans_n, rot_n),dim=1)
        #self.grid  = Variable(Tensor(grid))

    def send_device(self,device):
        self.data    = self.data.to(device)

    def __getitem__(self, index):
        data = self.data[index]
        #print(index)
        return data, index
    def __len__(self):
        return self.data.shape[0]

def Database(PATH):
    
    #try:
    points = np.load('{}/sampled_points.npy'.format(PATH))#[:100000,:]
    speed = np.load('{}/speed.npy'.format(PATH))#[:100000,:]
    normal = np.load('{}/normal.npy'.format(PATH))#[:100000,:]
    speed_angles = np.load('{}/speed_angles.npy'.format(PATH))#[:100000,:]
    speed_dists = np.load('{}/speed_dists.npy'.format(PATH))#[:100000,:]
    # Per-endpoint clearance-reduction directions (preprocess_obj output).  These
    # are optional: datasets generated before they existed (and the arm task) lack
    # them, so fall back to zeros of the same width as ``normal`` -- the column
    # layout downstream is then identical regardless of whether the files exist.
    def _load_or_zeros(name):
        import os
        f = '{}/{}.npy'.format(PATH, name)
        if os.path.exists(f):
            return np.load(f)
        print('  (no {}.npy -- using zeros)'.format(name))
        return np.zeros_like(normal)
    trans_n = _load_or_zeros('trans_n')
    rot_n = _load_or_zeros('rot_n')
    #occupancies = np.unpackbits(np.load('{}/voxelized_point_cloud_128res_20000points.npz'.format(PATH))['compressed_occupancies'])
    #input = np.reshape(occupancies, (128,)*3)
    #grid = np.array(input, dtype=np.float32)
    #print(tau.min())
    #p0 = np.random.rand(100000,2)-0.5
    #s0 = sdf(p0)
    #p1 = np.random.rand(100000,2)-0.5
    #s1 = sdf(p1)
    #points = np.concatinate((p0,p1),axis=1)
    #speed = np.concatinate((s0,s1),axis=1)
    print(points.shape)
    print(speed.shape)
    print(normal.shape)
        #points[:,2:]=0
        #speed[:,1:]=1
    #except ValueError:
    #    print('Please specify a correct source path, or create a dataset')
    rows=points.shape[0]


    print(points.shape,speed.shape)
    #print(np.shape(grid))
    #print(XP.shape,YP.shape)
    database = _numpy2dataset(points,speed,normal,speed_angles,speed_dists,
                              trans_n,rot_n)
    #database = _numpy2dataset(XP,YP)
    return database





