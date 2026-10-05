import matplotlib
import numpy as np
import math
import random
import time

import torch
import torch.nn.functional as F

from torch.nn import Linear
from torch import Tensor
from torch.nn import Conv3d
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable, grad
from torch.cuda.amp import autocast
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
#from EikoNet import database as db
#from models import data_mlp as db
#from models import model_network_one as model_network
import igl 
import copy

import matplotlib
import matplotlib.pylab as plt

from timeit import default_timer as timer

# Prior versions relied on ``torch_kdtree`` for nearest neighbour queries.
# The implementation no longer depends on that package.

# Optional Weights & Biases logging. Guarded by ``wandb.run is not None`` so it
# is a no-op unless a trainer started a run (see train/wandb_utils.py).
try:
    import wandb
except ImportError:
    wandb = None

torch.backends.cudnn.benchmark = True


def log_dsdn(trans_dsdn, rot_dsdn, trans_mask, rot_mask, epoch,
             every=50, verbose=True):
    """Summarise ds/dn for the translation and rotation probes.

    ``trans_dsdn`` / ``rot_dsdn`` are the per-sample speed slopes computed in
    ``Loss``; the masks select the samples inside the tight clearance band
    (speed < 1), which are the only ones the trans/rot error acts on.

    Always returns ``(trans_median, rot_median)`` (``nan`` when a mask is
    empty) so the caller can log the medians every epoch alongside the loss.
    The fuller quantile summary is printed / pushed to wandb only every
    ``every`` epochs, since it is the expensive and noisy part.
    """
    verbose = verbose and (epoch % every == 0)
    medians = {}
    stats = {}

    for name, t, m in (('trans', trans_dsdn, trans_mask),
                       ('rot', rot_dsdn, rot_mask)):
        v = t.detach()[m].flatten()
        if v.numel() == 0:
            medians[name] = float('nan')
            if verbose:
                print('  ds/dn {}: no in-band samples'.format(name))
            continue

        qs = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95],
                          device=v.device, dtype=v.dtype)
        p5, p25, med, p75, p95 = torch.quantile(v, qs).tolist()
        medians[name] = med
        stats.update({
            'dsdn/{}_n'.format(name): int(v.numel()),
            'dsdn/{}_mean'.format(name): v.mean().item(),
            'dsdn/{}_min'.format(name): v.min().item(),
            'dsdn/{}_p5'.format(name): p5,
            'dsdn/{}_p25'.format(name): p25,
            'dsdn/{}_p75'.format(name): p75,
            'dsdn/{}_p95'.format(name): p95,
            'dsdn/{}_max'.format(name): v.max().item(),
        })

        if verbose:
            print('  ds/dn {} (n={}): med={:.4g}  mean={:.4g}  '
                  'p5={:.4g}  p95={:.4g}  min={:.4g}  max={:.4g}'.format(
                      name, v.numel(), med, v.mean().item(),
                      p5, p95, v.min().item(), v.max().item()))

    if verbose and stats and wandb is not None and wandb.run is not None:
        wandb.log(stats, step=epoch)

    return medians['trans'], medians['rot']


class Function():
    def __init__(self, path, device, network, dim, data_path=None):

        # ======================= JSON Template =======================
        self.path = path
        self.device = device
        self.dim = dim

        self.network = network

        # Pass the JSON information
        #self.Params['Device'] = device

        # Parameters to alter during training
        self.total_train_loss = []
        self.total_val_loss = []
        #input_file = "datasets/gibson/Cabin/mesh_z_up_scaled.off"
        #self.kdtree, self.v_obs, self.n_obs = self.pc_kdtree(input_file)

        # Latest per-batch medians of ds/dn, refreshed by every Loss() call that
        # runs the trans/rot probes; nan until then. Read by the training loop.
        self.dsdn_trans_median = float('nan')
        self.dsdn_rot_median = float('nan')

        # Latest per-batch loss-term contributions, refreshed by every Loss()
        # call. eik/tr are per-sample and PRE-cap; see the stash site in Loss.
        self.eik_contrib = float('nan')
        self.tr_contrib = float('nan')
        self.tr_over_eik = float('nan')
        self.tr_cap_scale = float('nan')

        self.alpha = 1.025
        limit = 0.5
        self.margin = limit/15.0
        self.offset = self.margin/10.0

        # Environment boundary points (for the 2-D shape task) -- used only by
        # plot() to overlay the obstacles.  Optional: stays None if absent.
        self.env = None
        if data_path is not None:
            import os
            env_file = os.path.join(data_path, 'env.npy')
            if os.path.exists(env_file):
                self.env = np.load(env_file)
    
    def gradient(self, y, x, create_graph=True):                                                               
                                                                                  
        grad_y = torch.ones_like(y)                                                                 

        grad_x = torch.autograd.grad(y, x, grad_y, only_inputs=True, retain_graph=True, create_graph=create_graph)[0]
        
        return grad_x                                                                                                    
    
    def Loss(self, points, Yobs, normal, beta, gamma, epoch, speed_dist, speed_angle,
             trans_n=None, rot_n=None):
        
        tau, w, Xp = self.network.out(points)
        dtau = self.gradient(tau, Xp)

        
        D = torch.norm(Xp[:,self.dim:]-Xp[:,:self.dim], p=2, dim =1)
        
        
        DT0 = dtau[:,:self.dim]
        DT1 = dtau[:,self.dim:]
        
        
        S0 = torch.einsum('ij,ij->i', DT0, DT0)
        S1 = torch.einsum('ij,ij->i', DT1, DT1)


        half_dim = 3
        DT0_dist = dtau[:,:half_dim]
        DT0_ang = dtau[:,half_dim : self.dim]
        DT1_dist = dtau[:,self.dim : self.dim+half_dim]        
        DT1_ang = dtau[:,self.dim+half_dim:]  

        DT0_dist_mag = torch.einsum('ij,ij->i', DT0_dist, DT0_dist)
        DT0_ang_mag = torch.einsum('ij,ij->i', DT0_ang, DT0_ang)
        DT1_dist_mag = torch.einsum('ij,ij->i', DT1_dist, DT1_dist)      
        DT1_ang_mag = torch.einsum('ij,ij->i', DT1_ang, DT1_ang)

        LT0_dist_mag = torch.sqrt(DT0_dist_mag + 1e-8) * speed_dist[:,0] -1
        LT0_ang_mag = torch.sqrt(DT0_ang_mag + 1e-8) * speed_angle[:,0] -1
        LT1_dist_mag = torch.sqrt(DT1_dist_mag + 1e-8) * speed_dist[:,1] -1
        LT1_ang_mag = torch.sqrt(DT1_ang_mag + 1e-8) * speed_angle[:,1] -1

        LT0_dist_mag = LT0_dist_mag**2
        LT0_ang_mag = LT0_ang_mag**2
        LT1_dist_mag = LT1_dist_mag**2
        LT1_ang_mag = LT1_ang_mag**2

        mm = 20

        w0_dist = torch.clamp(1.0 / speed_dist[:,0], max=mm)
        w0_ang  = torch.clamp(1.0 / speed_angle[:,0], max=mm)
        w1_dist = torch.clamp(1.0 / speed_dist[:,1], max=mm)
        w1_ang  = torch.clamp(1.0 / speed_angle[:,1], max=mm)

        LT0_dist_mag = torch.where(speed_dist[:,0] < 0.9, LT0_dist_mag * w0_dist, LT0_dist_mag)
        LT0_ang_mag = torch.where(speed_angle[:,0] < 0.9, LT0_ang_mag * w0_ang, LT0_ang_mag)
        LT1_dist_mag = torch.where(speed_dist[:,1] < 0.9, LT1_dist_mag * w1_dist, LT1_dist_mag)
        LT1_ang_mag = torch.where(speed_angle[:,1] < 0.9, LT1_ang_mag * w1_ang, LT1_ang_mag)


        
        diff_4 = LT0_dist_mag + LT0_ang_mag + LT1_dist_mag + LT1_ang_mag
        td_weight = 0#1e-3
        with torch.no_grad():

            length0 = (0.03)/(Yobs[:,0]).unsqueeze(1)#5*torch.rand(Yobs.shape[0],1).cuda()
            Dir0 = length0*(DT0*Yobs[:,0].unsqueeze(1)**2).clone().detach()  
            #Dir1 = 0.03*(DT1/S1.unsqueeze(1)).clone().detach()  
            Xp_new0 = Xp.clone().detach()  
            
            Xp_new0[:,:self.dim] = Xp_new0[:,:self.dim] - Dir0

            tau_new0, w, Xp_new0 = self.network.out(Xp_new0)
            #tau_new1, w, Xp_new1 = self.network.out(Xp_new1)
            tau_new1 = length0#*1/Yobs[:,0].unsqueeze(1)
            del Xp_new0, Dir0#Xp_new1

        tau_loss0 = td_weight*((tau-(tau_new0+tau_new1))**2).squeeze()
        #(1.01-Yobs[:,0])*td_weight*
        #(1.4-Yobs[:,0])*

        with torch.no_grad():

            length1 = (0.03)/(Yobs[:,1]).unsqueeze(1)#5*torch.rand(Yobs.shape[0],1).cuda()
            Dir1 = length1*(DT1*Yobs[:,1].unsqueeze(1)**2).clone().detach()  
            Xp_new0 = Xp.clone().detach()  
            
            #Xp_new0[:,:self.dim] = Xp_new0[:,:self.dim] - Dir0
            Xp_new0[:,self.dim:] = Xp_new0[:,self.dim:] - Dir1
            #Xp_new[:,self.dim:]+=0.04*DT1/S1

            tau_new0, w, Xp_new0 = self.network.out(Xp_new0)
            #tau_new1, w, Xp_new1 = self.network.out(Xp_new1)
            tau_new1 = length1#*1/Yobs[:,1].unsqueeze(1)
            del Xp_new0, Dir1#Xp_new1

        tau_loss1 = td_weight*((tau-(tau_new0+tau_new1))**2).squeeze()
        
        where_d0 = (tau[:,0] < length0.squeeze())
        where_d1 = (tau[:,0] < length1.squeeze())
        tau_loss0[where_d0] = 0 
        tau_loss1[where_d1] = 0 

        tau_loss = tau_loss0+tau_loss1
        #'''

        Ypred0 = torch.sqrt(S0+1e-8)
        Ypred1 = torch.sqrt(S1+1e-8)


        Ypred0_visco = Ypred0
        Ypred1_visco = Ypred1

        sq_Ypred0 = (Ypred0_visco)#+gamma*lap0
        sq_Ypred1 = (Ypred1_visco)#+gamma*lap1


        sq_Yobs0 = (Yobs[:,0])#**2
        sq_Yobs1 = (Yobs[:,1])#**2

        #loss0 = (sq_Yobs0/sq_Ypred0+sq_Ypred0/sq_Yobs0)#**2#+gamma*lap0
        #loss1 = (sq_Yobs1/sq_Ypred1+sq_Ypred1/sq_Yobs1)#**2#+gamma*lap1
        l0 = ((sq_Yobs0*(sq_Ypred0)))
        l1 = ((sq_Yobs1*(sq_Ypred1)))
        
        l0_2 = (torch.sqrt(l0))#**(1/4)
        l1_2 = (torch.sqrt(l1))#**(1/4)    

        #w_num = w.clone().detach()
        loss_weight = 1e-2 #1e-2
        loss0 = loss_weight*(l0_2-1)**2  #/scale#+relu_loss0#**2#+gamma*lap0#**2
        loss1 = loss_weight*(l1_2-1)**2  #/scale#+relu_loss1#**2#+gamma*lap1#**2
        
        T = tau[:,0] #* torch.sqrt(T0)
        diff = loss0 + loss1 





        normal_weight = 0#1e-3

        normal0 = normal[:,:self.dim]
        normal1 = normal[:,self.dim:]


        DT0_dist = dtau[:,:half_dim]
        DT0_ang = dtau[:,half_dim : self.dim]
        DT1_dist = dtau[:,self.dim : self.dim+half_dim]        
        DT1_ang = dtau[:,self.dim+half_dim:]  


        # Block-split normal alignment (band point x0 only), mirroring the way the
        # eikonal term is split into dist / angle.  Per block:
        #     (1 - S*) * || S* * grad_block(tau)  +  n_block ||^2
        # with S* the GROUND-TRUTH speed for that block -- a (B,) scalar, so it needs
        # .unsqueeze(1) to broadcast across the 3 components of its gradient block.
        #
        # trans_n / rot_n point toward DECREASING clearance (trans_n = -n_ws, and
        # rot_axis = u_s x u_e closes the min angle), whereas this loss wants
        # grad S*/|grad S*|, which points AWAY from the obstacle -- hence the sign
        # flip.  Both are already unit vectors within their own 3-component block.
        sd0 = speed_dist[:, 0].unsqueeze(1)             # (B,1)
        sa0 = speed_angle[:, 0].unsqueeze(1)            # (B,1)

        n_trans0 = -trans_n[:, :half_dim]               # (B,3)
        n_rot0 = -rot_n[:, half_dim:self.dim]           # (B,3)

        n_loss0 = (1.001 - sd0) * (sd0 * DT0_dist + n_trans0)**2
        n_loss1 = (1.001 - sa0) * (sa0 * DT0_ang + n_rot0)**2



        n_loss = normal_weight*(torch.sum(n_loss0,dim=1)+torch.sum(n_loss1,dim=1))


        # ── Nudged-band-point probes (trans_n / rot_n) ──
        # trans_n / rot_n are generated only for the BAND point (x0 = first dim).
        # Nudge that point by a small ``step`` along each direction, re-pair the
        # nudged point with the original NON-band point (x1, the "start") -- so the
        # new pairs are [x0 + step*trans_n , x1] and [x0 + step*rot_n , x1] -- then
        # read dtau/dconfig for both probes (start point unchanged = non-band x1).
        step = 0.02
        dtau_trans = None
        dtau_rot = None
        if trans_n is not None and rot_n is not None:
            base = Xp.clone().detach()
            non_band = base[:, self.dim:]                 # x1: non-band ("start") point

            band_trans = base[:, :self.dim] + step * trans_n[:, :self.dim]
            band_rot   = base[:, :self.dim] + step * rot_n[:, :self.dim]

            points_trans = torch.cat([band_trans, non_band], dim=1)
            points_rot   = torch.cat([band_rot,   non_band], dim=1)

            tau_trans, w_trans, Xp_trans = self.network.out(points_trans)
            dtau_trans = self.gradient(tau_trans, Xp_trans)   # (B, 2*dim)

            tau_rot, w_rot, Xp_rot = self.network.out(points_rot)
            dtau_rot = self.gradient(tau_rot, Xp_rot)         # (B, 2*dim)


        rate_trans = -1/0.05
        rate_rot = -1/(1/6) # 1/6 pi / 2pi


        # Gradient magnitudes (sqrt of the einsum squared-magnitude, same +1e-8
        # guard as the sqrt(DT*_mag) terms above).  Originals are the un-nudged band
        # point (x0 = first dim): its translation part (first 3) and rotation part
        # (second 3) -- i.e. sqrt(DT0_dist_mag) / sqrt(DT0_ang_mag), recomputed here
        # for readability.  The probe versions use the component we actually nudged
        # along: trans probe -> translation (first 3), rot probe -> rotation (second
        # 3) of the nudged band point.
        dtau_original_trans_mag = torch.sqrt(torch.einsum('ij,ij->i', dtau[:, :half_dim], dtau[:, :half_dim]) + 1e-8)
        dtau_original_rot_mag   = torch.sqrt(torch.einsum('ij,ij->i', dtau[:, half_dim:self.dim], dtau[:, half_dim:self.dim]) + 1e-8)
        dtau_trans_mag = None
        dtau_rot_mag = None
        if dtau_trans is not None and dtau_rot is not None:
            dtau_trans_mag = torch.sqrt(torch.einsum('ij,ij->i', dtau_trans[:, :half_dim], dtau_trans[:, :half_dim]) + 1e-8)
            dtau_rot_mag   = torch.sqrt(torch.einsum('ij,ij->i', dtau_rot[:, half_dim:self.dim], dtau_rot[:, half_dim:self.dim]) + 1e-8)

        if dtau_trans_mag is not None and dtau_rot_mag is not None:
            # def _hist10(name, t):
            #     v = t.detach().flatten().cpu().numpy()
            #     counts, edges = np.histogram(v, bins=10)
            #     cmax = max(int(counts.max()), 1)
            #     print('{} (n={}, min={:.4g}, max={:.4g}, mean={:.4g}):'.format(
            #         name, v.size, v.min(), v.max(), v.mean()))
            #     for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
            #         print('  [{:9.4g}, {:9.4g}) {:7d} {}'.format(
            #             lo, hi, int(c), '#' * int(40 * c / cmax)))
            # _hist10('d|grad_xyz tau|/d(trans_n)  = (nudged-orig)/step',
            #         (dtau_trans_mag - dtau_original_trans_mag) / step)
            # _hist10('d|grad_rot tau|/d(rot_n)    = (nudged-orig)/step',
            #         (dtau_rot_mag - dtau_original_rot_mag) / step)
            # _hist10('|grad_xyz tau| (original x0)', dtau_original_trans_mag)
            # _hist10('|grad_rot tau| (original x0)', dtau_original_rot_mag)



            trans_slope = (dtau_trans_mag - dtau_original_trans_mag) / step
            rot_slope = (dtau_rot_mag - dtau_original_rot_mag) / step

            trans_dsdn = (1.0 / dtau_trans_mag - 1.0 / dtau_original_trans_mag) / step
            rot_dsdn = (1.0 / dtau_rot_mag - 1.0 / dtau_original_rot_mag) / step

            md = speed_dist[:, 0] < 1
            ma = speed_angle[:, 0] < 1

            # Medians are stashed for the trainer to log every epoch (see
            # model_train_metric.py); the quantile dump is on its own cadence.
            self.dsdn_trans_median, self.dsdn_rot_median = log_dsdn(
                trans_dsdn, rot_dsdn, md, ma, epoch)

            def _median(t):
                return t.median().item() if t.numel() > 0 else float('nan')

            # ---- distribution of ds/dn (within the tight band) to pick clamp bounds ----
            def _dist(name, t):
                t = t.detach().flatten()
                if t.numel() == 0:
                    print('  {}: empty'.format(name)); return
                qs = torch.tensor([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0],
                                  device=t.device, dtype=t.dtype)
                p = torch.quantile(t, qs).tolist()
                print('  {} (n={})'.format(name, t.numel()))
                print('    min={:.3g}  p1={:.3g}  p5={:.3g}  p25={:.3g}  med={:.3g}'
                      '  p75={:.3g}  p95={:.3g}  p99={:.3g}  max={:.3g}'.format(*p))
                # 10-bin histogram over the central p1..p99 range (ignore extreme tails)
                lo, hi = p[1], p[7]
                if hi > lo:
                    counts = torch.histc(t.clamp(lo, hi), bins=10, min=lo, max=hi).tolist()
                    cmax = max(counts) or 1
                    edges = [lo + (hi - lo) * i / 10 for i in range(11)]
                    for i, c in enumerate(counts):
                        print('    [{:8.3g},{:8.3g}) {:7d} {}'.format(
                            edges[i], edges[i + 1], int(c), '#' * int(40 * c / cmax)))


            err_weight = 1e-4 * min(max((epoch - 1500) / 2000.0, 0.0), 1.0)
            trans_err_weight = err_weight
            rot_err_weight = err_weight

            dsdn_clip = 10
            trans_dsdn_c = trans_dsdn.clamp(rate_trans, dsdn_clip)
            rot_dsdn_c = rot_dsdn.clamp(rate_rot, dsdn_clip)



            trans_err = trans_err_weight * (rate_trans - trans_dsdn_c) ** 2
            rot_err = rot_err_weight * (rate_rot - rot_dsdn_c) ** 2
            trans_err = torch.where(md, trans_err, torch.zeros_like(trans_err))
            rot_err = torch.where(ma, rot_err, torch.zeros_like(rot_err))


        print("only trans rot ")
        print(torch.sum((trans_err + rot_err )*torch.exp(-0.5*T)).item()/Yobs.shape[0])
        diff_4 = diff_4 * loss_weight
        # Cap the trans/rot term so its weighted contribution to the loss is at most
        # 1/3 of the eikonal (diff_4) contribution. The scale factor is detached, so
        # it only rescales the magnitude/gradient of the trans/rot term, never the
        # eikonal one.
        weight_T = torch.exp(-0.5 * T)
        eik_contrib = (diff_4 * weight_T).sum()
        tr_contrib = ((trans_err + rot_err) * weight_T).sum()
        tr_cap_scale = torch.clamp((eik_contrib / 1.5) / (tr_contrib + 1e-12),
                                   max=1.0).detach()

        # Stashed for the trainer to log every epoch. Both contributions are
        # PRE-cap and per-sample (same 1/N as loss_n), so their ratio is what
        # the cap reacts to -- not what actually lands in the loss. The cap
        # binds exactly when the ratio exceeds 1/1.5, i.e. tr_cap_scale < 1.
        self.eik_contrib = eik_contrib.item() / Yobs.shape[0]
        self.tr_contrib = tr_contrib.item() / Yobs.shape[0]
        self.tr_over_eik = tr_contrib.item() / (eik_contrib.item() + 1e-12)
        self.tr_cap_scale = tr_cap_scale.item()

        trans_err = trans_err * tr_cap_scale
        rot_err = rot_err * tr_cap_scale
        loss_n = (torch.sum((diff_4+ trans_err + rot_err  + n_loss +tau_loss)*weight_T))/Yobs.shape[0]#*torch.exp(-para*T)
        #loss_n = (torch.sum((diff_4+n_loss +tau_loss) ))/Yobs.shape[0]#*torch.exp(-para*T)



        loss = loss_n #+ cross_term #+ loss_2nd

        return loss, loss_n, diff_4

    def TravelTimes(self, Xp):
     
        tau, w, coords = self.network.out(Xp)        

        TT = tau[:,0] #* torch.sqrt(T0)
            
        return TT

    def Speed(self, Xp):

   

        Xp = Xp.to(torch.device(self.device))

        tau, w, Xp = self.network.out(Xp)
        dtau = self.gradient(tau, Xp)
        #Xp.requires_grad_()
        #tau, dtau, coords = self.network.out_grad(Xp)
        
        
        #D = Xp[:,self.dim:]-Xp[:,:self.dim]
        #T0 = torch.einsum('ij,ij->i', D, D)

        #DT0 = dtau[:,self.dim:]
        DT0 = dtau[:,:self.dim]

        
        
        S = torch.einsum('ij,ij->i', DT0, DT0)

        Ypred = 1/torch.sqrt(S)
        
        del Xp, tau, dtau#, T0#, T1, T2, T3
        return Ypred
    
    def Gradient(self, Xp):
        #Xp = Xp.to(torch.device(self.device))
       
        #Xp.requires_grad_()
        
        #tau, dtau, coords = self.network.out_grad(Xp)
        #print(Xp.shape)
        tau, w, Xp = self.network.out(Xp)
        dtau = self.gradient(tau, Xp)
        
        #D = Xp[:,self.dim:]-Xp[:,:self.dim]
        #T0 = torch.sqrt(torch.einsum('ij,ij->i', D, D)).view(-1,1)

        #A = T0*dtau[:,:self.dim]
        #B = tau/T0*D
        

        Ypred0 = -dtau[:,:self.dim]#-A+B
        #print(Ypred0.shape)
        Spred0 = torch.norm(Ypred0,dim=1).view(-1,1)
        Ypred0 = 1/Spred0**2*Ypred0

        Ypred1 = -dtau[:,self.dim:]#-A-B
        Spred1 = torch.norm(Ypred1,dim=1).view(-1,1)

        Ypred1 = 1/Spred1**2*Ypred1

        #print(Ypred0.shape)
        #print(Ypred1.shape)
        
        return torch.cat((Ypred0, Ypred1),dim=1)
    
    def plot(self, epoch, total_train_loss, alpha, source):
        # Travel-time / speed field over an (x, y) grid, with the rest of the
        # configuration (theta for the 2-D shape task) held fixed at `source`.
        limit = 0.5
        size = 81
        spacing = 2 * limit / (size - 1)
        X, Y = np.meshgrid(np.arange(-limit, limit + 0.1 * spacing, spacing),
                           np.arange(-limit, limit + 0.1 * spacing, spacing))

        Xsrc = source  # goal configuration -- T is measured from here

        XP = np.zeros((len(X.flatten()), 2 * self.dim))
        XP[:, :self.dim] = Xsrc
        XP[:, self.dim:] = Xsrc
        XP[:, 0] = X.flatten()
        XP[:, 1] = Y.flatten()
        XP = Variable(Tensor(XP)).to(self.device)

        tt = self.TravelTimes(XP)
        ss = self.Speed(XP)

        TT = tt.to('cpu').data.numpy().reshape(X.shape)
        S = ss.to('cpu').data.numpy().reshape(X.shape)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        quad1 = ax.pcolormesh(X, Y, S, vmin=0, vmax=1)
        ax.contour(X, Y, TT, np.arange(0, 10, 0.02), cmap='gist_heat', linewidths=0.5)
        if self.env is not None:
            ax.scatter(self.env[:, 0], self.env[:, 1], c='black', s=2, zorder=3)
        ax.scatter([Xsrc[0]], [Xsrc[1]], c='red', s=40, zorder=4, label='source')
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(loc='upper right', fontsize=7)
        plt.colorbar(quad1, ax=ax, pad=0.1, label='Predicted Velocity')
        plt.savefig(self.path + "/plots" + str(epoch) + "_" + str(alpha) + "_"
                    + str(round(total_train_loss, 4)) + "_0.jpg", bbox_inches='tight')
        plt.close(fig)


         
