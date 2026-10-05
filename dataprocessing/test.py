import torch, importlib.util, numpy as np
spec = importlib.util.spec_from_file_location("pp", "dataprocessing/preprocess_obj.py")
pp = importlib.util.module_from_spec(spec); spec.loader.exec_module(pp)

torch.manual_seed(0)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
B, E, bins, num, F, K = 16, 5000, 10, 40, 30, 24

# fake shape: radial surface points sorted into shells, + matching edges
rp = torch.randn(bins*num, 3)*0.05
r  = rp.norm(dim=1); rp = rp[r.argsort()].reshape(bins, num, 3)
rad_pts = rp
radii = rp.reshape(-1,3).norm(dim=1).reshape(bins,num)
edges = torch.empty(bins+1); edges[:bins] = radii[:,0]; edges[bins] = radii.max()
face = torch.randn(F,3,3)*0.05
tet  = torch.randn(K,4,3)*0.05
env  = torch.randn(E,3)*0.5
cfg  = torch.cat([torch.randn(B,3)*0.2, torch.randn(B,3)*0.5], 1)

free, dist, ang, normal, *_ = pp.evaluate_placements(
    cfg, tet, face, env, rad_pts, edges, dev)

# brute-force min_angle reference
import math
t = cfg[:,:3]; Rm = pp.rotvec_to_matrix(cfg[:,3:6])
ref = torch.empty(B)
for b in range(B):
    best = -1.0
    d = (env - t[b]).norm(dim=1)
    edir = (env - t[b]) / (d[:,None] + 1e-12)
    for bi in range(bins):
        m = (d > edges[bi]) & (d <= edges[bi+1])
        if not m.any(): continue
        sdir = (rad_pts[bi] @ Rm[b].T)
        sdir = sdir / (sdir.norm(dim=1, keepdim=True) + 1e-12)
        best = max(best, (sdir @ edir[m].T).max().item())
    ref[b] = math.acos(max(-1.0, min(1.0, best)))

print("max |min_angle - bruteforce|:", (ang - ref).abs().max().item())