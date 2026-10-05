"""Optimized single-case hlB + spread-steering planner -- path generation only.

Same planner as ``_spread_planner.py`` (hlB with the sampled-speed spread
mechanisms, default ``--steer-bias 0.5 --no-gate`` = the ``SR Alternate
Bellman Horizon SD`` column), stripped to what is needed to *produce a path*
and time it, one start/goal pair at a time -- no batching, no collision
checking, no success rate, no viewer.  For each case it reports

  * translation length  -- ``sum ||t_{i+1} - t_i||`` over the waypoints
                           (normalized frame, environment's longest side = 1)
  * rotation length     -- ``sum 2 acos|q_i . q_{i+1}|`` in radians
                           (``|wrap(dyaw)|`` on a planar set), the geodesic
                           angle between consecutive orientations
  * generation time     -- wall clock of the MPPI loop for that case, from
                           the first noise draw to the last goal test,
                           measured with CUDA events

with the same length definitions as ``experiments_rrt_connect.md`` (the
``path_components`` of ``rrt_connect_eval.py``), so the numbers are directly
comparable with the RRT-Connect / Lazy PRM / NTFields 1k tables.

Speed comes from the same exact restructurings ``evaluate_training_3d_opt.py``
uses (see ``FastField`` there): ``lip_norm``, ``2*pi*B`` and the residual
gates are constants at inference and are computed once; the two endpoints of a
tau query are embedded separately, so per MPPI iteration the trunk runs on
``samples * horizon + 1`` rows instead of ``4 * samples * horizon`` (the
reference stacks both endpoints of both the cost-to-go and the local query);
autograd bookkeeping is dropped.  On top of that the alternating chain ends
are embedded only when they move: the mover's embedding is the one computed
for the other end one iteration earlier, and the newly moved end is folded
into the next iteration's candidate batch as its last row.  The MPPI
arithmetic -- noise, step clamp, all-horizon cost, first-step speed spread,
least-squares speed gradient, softmax weights, steer bias, 0.01 goal ball,
alternating ends -- is unchanged, so with ``--verify`` the fast field matches
``womodel.function.TravelTimes`` to float32 roundoff.

Run from the nested ntrl-demo root:

    python er_opt.py --dataPath testing_data_1k_complete/3dshape/rectangle_env1 \
        --checkpoint ./Experiments/3dshape/3dshape_08_06_17_06/latest.pt \
        --device cuda:1 --cases 200 --out results/er_opt_1k/rectangle_env1
"""
import sys
sys.path.append('.')
import os, json, argparse, time, statistics
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from models.metric import model_train_metric as md

DIM = 6
PLANAR_FREE_DIMS = (0, 1, 5)     # preprocess_obj.py --2d frees x, y, rz only

p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
p.add_argument('--dataPath', required=True)
p.add_argument('--checkpoint', required=True)
p.add_argument('--modelPath', default='./Experiments/3dshape')
p.add_argument('--device', default='cuda')
p.add_argument('--cases', type=int, default=200)
p.add_argument('--out', required=True)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--warmup', type=int, default=3,
               help='untimed planning runs on case 0 before the loop (CUDA context, autotune)')
p.add_argument('--step', type=float, default=0.015)
p.add_argument('--steps', type=int, default=200)
p.add_argument('--local-weight', dest='local_w', type=float, default=0.03)
p.add_argument('--samples', type=int, default=50)
p.add_argument('--horizon', type=int, default=5)
p.add_argument('--steer-bias', type=float, default=0.5)
p.add_argument('--steer-cost', type=float, default=0.0)
p.add_argument('--zlocal', type=float, default=0.0)
p.add_argument('--rel-gate', type=float, default=0.0)
p.add_argument('--adapt-lw', type=float, default=0.0)
p.add_argument('--cv0', type=float, default=0.10)
p.add_argument('--cv1', type=float, default=0.16)
p.add_argument('--smean-gate', action='store_true')
p.add_argument('--m0', type=float, default=0.15)
p.add_argument('--m1', type=float, default=0.40)
p.add_argument('--gate', dest='no_gate', action='store_false', default=True,
               help='cv-gate the mechanisms (default is --no-gate, as in the SD column)')
p.add_argument('--steer-trans', action='store_true')
p.add_argument('--2d', dest='two_d', action='store_true',
               help="planar test set: sample only x, y, rz (auto-detected from meta.json's two_d)")
p.add_argument('--verify', action='store_true',
               help='compare the fast field against womodel.function.TravelTimes and print the max abs error')
p.add_argument('--graph', action='store_true',
               help='capture one MPPI iteration per mover as a CUDA graph and replay it; the loop is '
                    'launch-bound (~240 kernels per iteration, ~0.5 ms of GPU work), so this removes '
                    'most of the wall clock.  Same kernels, same arithmetic.')
p.add_argument('--tf32', action='store_true')
a = p.parse_args()

if a.tf32:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(a.seed); np.random.seed(a.seed)
os.makedirs(a.out, exist_ok=True)
dev = a.device
cuda = dev.startswith('cuda')
if cuda:
    torch.cuda.set_device(dev)      # graph capture / side streams use the *current* device


# ──────────────────────────────────────────────────────────────────────────────
# Fast inference-only view of the trained field (evaluate_training_3d_opt.FastField)
# ──────────────────────────────────────────────────────────────────────────────
class FastField:
    """Cached, gradient-free reimplementation of ``NN.out`` -> tau, with the two
    endpoints embedded separately.  Row-independent (the only cross-row op in
    the trunk is InstanceNorm1d on a 2-D input, which normalizes per row)."""

    def __init__(self, network):
        self.dim = network.dim
        self.act = network.act
        self.nl1 = network.nl1
        self.fuse_len = network.fuse_len
        self.half = 3
        self.W_trans = (2.0 * np.pi * network.B_trans).contiguous()
        self.W_rot = (2.0 * np.pi * network.B_rot).contiguous()
        self.route_t = self._cache_route(
            network.pe_gate_t, network.gate_t, network.encoder_t, network.encoder_norm_t)
        self.route_r = self._cache_route(
            network.pe_gate_r, network.gate_r, network.encoder_r, network.encoder_norm_r)
        self.fuse = [(m.weight.T.contiguous(), m.bias) for m in network.fuse]

    def _lip(self, w):
        absrowsum = torch.sqrt(torch.sum(w ** 2, dim=1))
        scale = 1 + 1e-5 - self.act(1 - 1 / absrowsum)
        return (w * scale.unsqueeze(1)).T.contiguous()

    def _cache_route(self, pe_gate, gate, encoder, encoder_norm):
        blocks = []
        for ii in range(self.nl1):
            blocks.append((
                self._lip(encoder[3 * ii + 1].weight), encoder[3 * ii + 1].bias,
                self._lip(encoder[3 * ii + 2].weight), encoder[3 * ii + 2].bias,
                self._lip(encoder[3 * ii + 3].weight), encoder[3 * ii + 3].bias,
                torch.sigmoid(0.1 * gate[ii].weight),
            ))
        return {'pe0': (self._lip(pe_gate[0].weight), pe_gate[0].bias),
                'pe1': (self._lip(pe_gate[1].weight), pe_gate[1].bias),
                'blocks': blocks,
                'final': (self._lip(encoder[-1].weight), encoder[-1].bias),
                'norm': encoder_norm}

    def _route(self, x, W, r):
        x_proj = x @ W
        x = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        w, b = r['pe0']; u = torch.sin(x @ w + b)
        w, b = r['pe1']; v = torch.sin(x @ w + b)
        for w1, b1, w2, b2, w3, b3, g in r['blocks']:
            x_tmp = x
            s = torch.sin(x @ w1 + b1); x = u * s + v * (1 - s)
            s = torch.sin(x @ w2 + b2); x = u * s + v * (1 - s)
            x = (1 - g) * x_tmp + g * torch.sin(x @ w3 + b3)
        w, b = r['final']
        return r['norm'](x @ w + b)

    def embed(self, x):
        y = torch.cat([self._route(x[:, :self.half], self.W_trans, self.route_t),
                       self._route(x[:, self.half:], self.W_rot, self.route_r)], dim=-1)
        w, b = self.fuse[0]
        res = y @ w + b
        for i in range(self.fuse_len):
            w, b = self.fuse[2 * i + 1]; y1 = self.act(res @ w + b)
            w, b = self.fuse[2 * i + 2]; res = self.act(res + (y1 @ w + b))
        return res

    @staticmethod
    def tau(e0, e1):
        x = torch.sqrt((e0 - e1) ** 2 + 1e-6)
        x = x.view(x.shape[0], -1, 16)
        x = (torch.logsumexp(10 * x, 2) - np.log(16)) / 10
        return 0.2 * torch.sum(x, dim=1)

    def travel_times(self, Xp):
        return self.tau(self.embed(Xp[:, :self.dim]), self.embed(Xp[:, self.dim:]))


# ──────────────────────────────────────────────────────────────────────────────
# Planner: one case, alternating ends, spread mechanisms as in _spread_planner.py
# ──────────────────────────────────────────────────────────────────────────────
def solve_spd(M, b):
    """Solve M x = b for a small SPD M with Gauss-Jordan elimination (no pivoting,
    which is stable for SPD), written in plain tensor ops so it can be captured
    in a CUDA graph -- torch.linalg.solve goes through cusolver, which cannot.
    Replaces the reference's torch.linalg.solve(AtA, Aty) to float32 roundoff."""
    n = M.shape[0]
    aug = torch.cat([M, b], dim=1)                                  # (n, n+1)
    for k in range(n):
        row = aug[k:k + 1] / aug[k:k + 1, k:k + 1]
        aug = aug - aug[:, k:k + 1] * row
        aug = torch.cat([aug[:k], row, aug[k + 1:]], dim=0)
    return aug[:, n:]


class Planner:
    """One MPPI iteration is ``_step(k)`` for mover ``k`` (0 = the chain growing
    from the start, 1 = the one growing from the goal), written entirely into
    static buffers so it can be replayed as a CUDA graph (``--graph``): the
    graph records addresses, so the loop state is updated in place.  Eager mode
    calls the same function, so the two modes run identical arithmetic."""

    def __init__(self, field, free_mask, graph):
        self.field = field
        S, H, D = a.samples, a.horizon, DIM
        with torch.no_grad():
            h = field.embed(torch.zeros((1, D), device=dev)).shape[1]
        z = lambda *s, dt=torch.float32: torch.zeros(s, device=dev, dtype=dt)
        self.free_mask = free_mask                                  # (D,) or None
        self.eye = 1e-8 * torch.eye(D, device=dev)
        self.trans_mask = torch.tensor([1., 1., 1., 0., 0., 0.], device=dev)
        self.st = dict(
            ends=z(2, D), e_end=z(2, h),
            noise_a=torch.empty((S, 1, D), device=dev), noise_b=torch.empty((S, H, D), device=dev),
            chain=z(2, a.steps + 2, D), ctr=z(2, dt=torch.long),
            dist=z(), s_ref=z(),
        )
        self.graphs = None
        if graph:
            # Warm up on a side stream first (cuBLAS workspace allocation must not
            # land inside the capture), then record one iteration per mover.
            with torch.no_grad():
                s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        self._step(0); self._step(1)
                torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
                self.graphs = []
                for k in (0, 1):
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        self._step(k)
                    self.graphs.append(g)

    def cost_and_spread(self, f, local, seg, dP0, local_w, s_ref):
        """Mirror of _spread_planner._candidate_cost for a single case (S, H)."""
        slow = local / seg                                          # (S,H)
        s = 1.0 / slow[:, 0]                                        # sampled speeds
        s_mean = s.mean(); s_std = s.std().clamp(min=1e-6)
        cv = s_std / s_mean
        y = (s - s_mean)[:, None]
        g = solve_spd(dP0.T @ dP0 + self.eye, dP0.T @ y)[:, 0]
        if a.steer_trans:
            g = g * self.trans_mask
        if self.free_mask is not None:
            g = g * self.free_mask
        g_unit = g / torch.norm(g).clamp(min=1e-9)
        if a.no_gate:
            gate = torch.ones((), device=dev)
        elif a.smean_gate:
            drop = 1.0 - s_mean / s_ref.clamp(min=1e-6)
            gate = ((drop - a.m0) / (a.m1 - a.m0)).clamp(0, 1)
        else:
            gate = ((cv - a.cv0) / (a.cv1 - a.cv0)).clamp(0, 1)
        lw = local_w * (1.0 + a.adapt_lw * gate)
        f = f + lw * slow
        cost = 10 * f[:, 0] + f[:, 1:].mean(dim=1)
        if a.steer_cost > 0:
            align = (dP0 * g_unit[None, :]).sum(1) / a.step
            cost = cost - a.steer_cost * gate * align
        if a.zlocal > 0:
            cost = cost + a.zlocal * gate * (s_mean - s) / s_std
        if a.rel_gate > 0:
            cost = cost + 1e3 * gate * (s < s_mean - a.rel_gate * s_std).float()
        return cost, s_mean, g_unit, gate

    def _step(self, k):
        """One MPPI iteration moving end k towards end 1-k, in place."""
        st, field, step, S, H, D = self.st, self.field, a.step, a.samples, a.horizon, DIM
        cur, oth = st['ends'][k:k + 1], st['ends'][1 - k:2 - k]
        dP = step * st['noise_a'].normal_() + step * st['noise_b'].normal_()
        if self.free_mask is not None:
            dP = dP * self.free_mask
        dP = dP / (torch.clamp(torch.norm(dP, dim=2, keepdim=True), min=step) / step)
        cand = cur + torch.cumsum(dP, dim=1)                        # (S,H,D)
        # The other end moved last iteration; refresh its embedding as one extra row.
        emb = field.embed(torch.cat([cand.reshape(-1, D), oth], 0))
        st['e_end'][1 - k:2 - k].copy_(emb[-1:])
        emb = emb[:-1]
        e_cur, e_oth = st['e_end'][k:k + 1], st['e_end'][1 - k:2 - k]
        f = field.tau(emb, e_oth).reshape(S, H)                     # cost-to-go of every candidate
        local = field.tau(e_cur, emb).reshape(S, H)                 # tau(cur -> candidate)
        seg = torch.norm(cand - cur, dim=2).clamp(min=1e-6)
        dP0 = dP[:, 0, :]
        cost, s_mean, g_unit, gate = self.cost_and_spread(f, local, seg, dP0, a.local_w, st['s_ref'])
        st['s_ref'].copy_(torch.maximum(st['s_ref'], s_mean))
        weight = torch.softmax(-50 * cost, dim=0)
        step_prior = weight @ dP0
        if a.steer_bias > 0:
            step_prior = step_prior + a.steer_bias * step * gate * g_unit
        cur.add_(step_prior[None, :])
        st['chain'][k].index_copy_(0, st['ctr'][k:k + 1], cur)
        st['ctr'][k:k + 1].add_(1)
        st['dist'].copy_(torch.norm(oth - cur))

    def plan(self, start, goal):
        """start, goal: (1, D).  Returns (path (T, D) numpy, converged, iterations)."""
        st = self.st
        st['ends'][0:1].copy_(start); st['ends'][1:2].copy_(goal)
        st['e_end'][0:1].copy_(self.field.embed(start))             # e_end[1] is folded in by the first step
        st['chain'][0, 0].copy_(start[0]); st['chain'][1, 0].copy_(goal[0])
        st['ctr'].fill_(1); st['s_ref'].zero_()
        converged = False
        it = 0
        for it in range(a.steps):
            k = it % 2
            if self.graphs is not None:
                self.graphs[k].replay()
            else:
                self._step(k)
            if st['dist'].item() < 0.01:
                converged = True
                break
        n0, n1 = st['ctr'].tolist()
        path = torch.cat([st['chain'][0, :n0], st['chain'][1, :n1].flip(0)], 0)
        return path.cpu().numpy(), converged, it + 1


def path_components(P, two_d):
    """(translation length, rotation length in radians) of a (T, 6) config path,
    defined as in baselines/baseline_ompl/rrt_connect_eval.py."""
    P = np.asarray(P, dtype=np.float64)
    trans = float(np.linalg.norm(np.diff(P[:, :3], axis=0), axis=1).sum())
    if two_d:
        d = np.diff(P[:, 5] * 2 * np.pi)
        rot = float(np.abs((d + np.pi) % (2 * np.pi) - np.pi).sum())
    else:
        q = Rotation.from_rotvec(P[:, 3:6] * 2 * np.pi).as_quat()
        dots = np.abs((q[1:] * q[:-1]).sum(1)).clip(max=1.0)
        rot = float((2.0 * np.arccos(dots)).sum())
    return trans, rot


def mean(x): return float(np.mean(x)) if len(x) else float('nan')
def std(x): return float(np.std(x, ddof=1)) if len(x) > 1 else float('nan')


# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────
with open(os.path.join(a.dataPath, 'meta.json')) as fh:
    meta = json.load(fh)
two_d = bool(a.two_d or meta.get('two_d'))
free_mask = None
if two_d:
    free_mask = torch.zeros(DIM, device=dev); free_mask[list(PLANAR_FREE_DIMS)] = 1.0
    print('[--2d] planar rollout: free dims', PLANAR_FREE_DIMS)

womodel = md.Model(a.modelPath, a.dataPath, DIM, [0.0] * DIM, device=dev)
womodel.load(a.checkpoint); womodel.network.eval()
for prm in womodel.network.parameters():
    prm.requires_grad_(False)
field = FastField(womodel.network)

arr = np.load(os.path.join(a.dataPath, 'sampled_points.npy'))[:a.cases]
X = torch.tensor(arr, dtype=torch.float32, device=dev)
print(f'checkpoint : {a.checkpoint}\ndata       : {a.dataPath}\ndevice     : {dev}\ncases      : {len(arr)}')

if a.verify:
    with torch.no_grad():
        probe = X[:min(64, len(X))].contiguous()
        ref = womodel.function.TravelTimes(probe).detach()
        err = (ref - field.travel_times(probe)).abs()
        print(f'verify     : max abs err {err.max().item():.3e}  mean {err.mean().item():.3e}  '
              f'(ref range [{ref.min().item():.4f}, {ref.max().item():.4f}])')

if a.graph and not cuda:
    raise SystemExit('--graph requires a CUDA device.')
planner = Planner(field, free_mask, a.graph)
print('mode       :', 'CUDA graph (one captured iteration per mover, replayed)' if a.graph else 'eager')


def timed_plan(i):
    XP = X[i:i + 1]
    if cuda:
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
    else:
        w0 = time.perf_counter()
    path, conv, iters = planner.plan(XP[:, :DIM], XP[:, DIM:])
    if cuda:
        t1.record(); torch.cuda.synchronize(); dt = t0.elapsed_time(t1) / 1e3
    else:
        dt = time.perf_counter() - w0
    return path, conv, iters, dt


# ──────────────────────────────────────────────────────────────────────────────
# Loop
# ──────────────────────────────────────────────────────────────────────────────
recs, paths = [], []
with (torch.no_grad() if a.graph else torch.inference_mode()):
    for _ in range(min(a.warmup, len(arr))):
        timed_plan(0)
    torch.manual_seed(a.seed)
    wall0 = time.time()
    for i in range(len(arr)):
        path, conv, iters, dt = timed_plan(i)
        tl, rl = path_components(path, two_d)
        recs.append(dict(case=i, converged=bool(conv), iters=int(iters), waypoints=int(len(path)),
                         time_s=float(dt), trans_length=tl, rot_length_rad=rl))
        paths.append(path)
        print(f'[{i:03d}] {"REACHED" if conv else "NO-CONV"}  iters={iters:3d}  wp={len(path):3d}  '
              f'{dt * 1e3:8.2f} ms  trans={tl:6.3f}  rot={rl:6.3f} rad')
    wall = time.time() - wall0

conv = [r for r in recs if r['converged']]
summary = dict(
    args=vars(a), two_d=two_d, n=len(recs), n_converged=len(conv), wall_s=wall,
    # over the converged cases (the counterpart of the baselines' "successful cases only")
    time_mean=mean([r['time_s'] for r in conv]), time_std=std([r['time_s'] for r in conv]),
    time_median=float(np.median([r['time_s'] for r in conv])) if conv else float('nan'),
    trans_length_mean=mean([r['trans_length'] for r in conv]), trans_length_std=std([r['trans_length'] for r in conv]),
    rot_length_mean=mean([r['rot_length_rad'] for r in conv]), rot_length_std=std([r['rot_length_rad'] for r in conv]),
    iters_mean=mean([r['iters'] for r in conv]),
    # over every case, converged or not
    time_mean_all=mean([r['time_s'] for r in recs]), time_std_all=std([r['time_s'] for r in recs]),
)
json.dump(dict(summary=summary, records=recs), open(os.path.join(a.out, 'records.json'), 'w'), indent=1)
np.save(os.path.join(a.out, 'paths.npy'), np.array(paths, dtype=object), allow_pickle=True)
print()
print(f'cases            : {summary["n"]}   converged {summary["n_converged"]}  '
      f'(goal ball reached; NOT a success rate -- no collision checking here)')
print(f'gen time         : {summary["time_mean"] * 1e3:8.2f} ± {summary["time_std"] * 1e3:.2f} ms   '
      f'median {summary["time_median"] * 1e3:.2f} ms   (converged cases; all cases '
      f'{summary["time_mean_all"] * 1e3:.2f} ± {summary["time_std_all"] * 1e3:.2f} ms)')
print(f'trans length     : {summary["trans_length_mean"]:.4f} ± {summary["trans_length_std"]:.4f}')
print(f'rot length       : {summary["rot_length_mean"]:.4f} ± {summary["rot_length_std"]:.4f} rad')
print(f'mean iterations  : {summary["iters_mean"]:.1f}     wall {wall:.0f} s')
