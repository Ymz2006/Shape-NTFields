"""Performance-only 3-D (SE(3)) planning benchmark -- path generation, nothing else.

Stripped-down counterpart of ``evaluate_training_3d.py``.  Everything that is not
needed to *produce a path* has been removed:

  * no viser viewer, no plotly summaries, no matplotlib -- no visuals at all
  * no igl collision checking, no mesh loading, no linear-interpolation baseline
  * no per-waypoint Speed() queries, no de-normalized waypoint bookkeeping

What is left is the MPPI rollout driven by ``tau``, timed per episode, reporting
the AVERAGE and MEDIAN wall-clock time to plan one path.

Speed comes from three exact restructurings of the inference path (see FastField):

  1. ``lip_norm`` is applied to every Linear weight on *every* forward in the
     reference implementation.  At inference the weights are frozen, so the
     normalized weights are constants -- they are computed once here and cached
     (pre-transposed for matmul).  Same for ``2*pi*B`` and the ``sigmoid(0.1*w)``
     residual gates.
  2. The goal endpoint is identical for every MPPI sample and never changes
     within an episode, but ``NN.out`` re-embeds it on every call (it stacks both
     endpoints into one batch).  Here the goal embedding is computed ONCE per
     episode, halving trunk work per iteration.  This is safe because the only
     cross-row op in the trunk is ``InstanceNorm1d`` on a 2-D input, which
     normalizes each row independently.
  3. ``NN.out`` does ``coords.clone().detach().requires_grad_(True)`` so callers
     can differentiate tau.  TravelTimes never needs that gradient, so the clone
     and the autograd bookkeeping are dropped (the loop runs in inference_mode).

Additionally the MPPI inner loop only ever queries horizon steps 0 and -1, so the
(sample_num, horizon, 2*dim) rollout buffer is never materialized -- only the two
queried slices are built.

``--graph`` goes further and captures one MPPI iteration as a CUDA graph (see
GraphMPPI): the loop is kernel-launch bound, not compute bound -- per-iteration
cost is flat from 12 to 400 rollout samples -- so replaying a recorded launch
sequence removes most of the remaining time.

Measured on an idle RTX 3090 (rectangle_env1, 95 timed episodes, all three runs
back to back on the same GPU), against the original evaluate_training_3d loop:

    original loop      279.6 ms avg   258.3 ms median
    this script        107.7 ms avg   101.1 ms median   (2.6x)
    ... with --graph    26.9 ms avg    25.9 ms median   (10.4x / 3.9x over eager)

All three reach the goal on 94/95 episodes.  The fast field matches
``TravelTimes`` to 4.8e-07 max abs error (float32 roundoff), and --graph paths
respect the same 0.015 step cap and 0.01 goal threshold as the eager path.

NOTE: these timings are only meaningful on an unloaded GPU.  Because the loop is
launch bound, a competing job that saturates the GPU erases most of the --graph
advantage (measured 171 vs 182 ms when sharing a card at 99% utilization).

Use ``--verify`` to check the fast field against ``womodel.function.TravelTimes``
on real inputs before benchmarking.

Run from the nested ntrl-demo root:

    python evaluate_training_3d_opt.py --dataPath ./testing_data/3dshape/rectangle_env1 \
        --checkpoint ./Experiments/3dshape/3dshape_07_25_12_37/latest.pt
"""

import sys
sys.path.append('.')

import os
import argparse
import statistics
from glob import glob

import numpy as np
import torch

from models.metric import model_train_metric as md

DIM = 6


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Timing-only MPPI path generation for the 3-D shape planner.')
parser.add_argument('--dataPath', default='./testing_data/3dshape/rectangle_env1',
                    help='Test-data dir holding sampled_points.npy (start/goal pairs).')
parser.add_argument('--modelPath', default='./Experiments/3dshape',
                    help='Experiment root; used to locate the checkpoint when '
                         '--checkpoint is not given.')
parser.add_argument('--checkpoint', default=None,
                    help='Explicit .pt to load. Defaults to <modelPath>/latest.pt, '
                         'falling back to the newest <modelPath>/*/Model_Epoch_*.pt.')
parser.add_argument('--episodes', type=int, default=100,
                    help='Number of start/goal pairs to plan for.')
parser.add_argument('--warmup', type=int, default=5,
                    help='Episodes planned but excluded from the timing stats '
                         '(absorbs CUDA context / kernel-autotune startup cost).')
parser.add_argument('--steps', type=int, default=120, help='Max MPPI iterations.')
parser.add_argument('--samples', type=int, default=50, help='MPPI rollout samples.')
parser.add_argument('--horizon', type=int, default=5, help='MPPI horizon.')
parser.add_argument('--check-every', type=int, default=1, dest='check_every',
                    help='Test the goal-reached condition every N iterations. '
                         'N=1 matches evaluate_training_3d.py exactly and is the '
                         'right choice: N>1 drops the per-iteration GPU->CPU sync '
                         'but measurement shows the sync is not the bottleneck, so '
                         'the only effect is overshooting the goal by up to N-1 '
                         'steps -- which makes it net SLOWER.')
parser.add_argument('--graph', action='store_true',
                    help='Capture one MPPI iteration as a CUDA graph and replay it, '
                         'instead of re-issuing ~100 kernel launches from Python '
                         'every iteration. Same kernels, same math; removes the '
                         'CPU-side launch cost that dominates this loop.')
parser.add_argument('--tf32', action='store_true',
                    help='Allow TF32 matmuls (slightly lower precision).')
parser.add_argument('--verify', action='store_true',
                    help='Compare the fast field against the reference '
                         'TravelTimes on real inputs and print the max abs error.')
parser.add_argument('--save-paths', default=None, dest='save_paths',
                    help='Optional .npz to write the generated paths to.')
parser.add_argument('--device', default='cuda')
args = parser.parse_args()

if args.tf32:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ──────────────────────────────────────────────────────────────────────────────
# Fast inference-only view of the trained field
# ──────────────────────────────────────────────────────────────────────────────
class FastField:
    """Cached, gradient-free reimplementation of ``NN.out`` -> tau.

    Mirrors ``model_network_metric.NN`` exactly: same layer order, same
    activations, same ``InstanceNorm1d``, same logsumexp aggregation.  The only
    differences are that weight-space constants are precomputed and that the two
    endpoints can be embedded separately.
    """

    def __init__(self, network):
        self.net = network
        self.dim = network.dim
        self.act = network.act
        self.nl1 = network.nl1
        self.fuse_len = network.fuse_len
        self.half = 3

        # 2*pi*B, exactly as input_mapping computes it every call.
        self.W_trans = (2.0 * np.pi * network.B_trans).contiguous()
        self.W_rot = (2.0 * np.pi * network.B_rot).contiguous()

        self.route_t = self._cache_route(
            network.pe_gate_t, network.gate_t, network.encoder_t, network.encoder_norm_t)
        self.route_r = self._cache_route(
            network.pe_gate_r, network.gate_r, network.encoder_r, network.encoder_norm_r)

        # The fuse trunk is used raw (no lip_norm); cache the transposes only.
        self.fuse = [(m.weight.T.contiguous(), m.bias) for m in network.fuse]

    def _lip(self, w):
        """``NN.lip_norm``, evaluated once on a frozen weight."""
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
        return {
            'pe0': (self._lip(pe_gate[0].weight), pe_gate[0].bias),
            'pe1': (self._lip(pe_gate[1].weight), pe_gate[1].bias),
            'blocks': blocks,
            'final': (self._lip(encoder[-1].weight), encoder[-1].bias),
            'norm': encoder_norm,
        }

    def _route(self, x, W, r):
        x_proj = x @ W
        x = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        w, b = r['pe0']
        u = torch.sin(x @ w + b)
        w, b = r['pe1']
        v = torch.sin(x @ w + b)

        for w1, b1, w2, b2, w3, b3, g in r['blocks']:
            x_tmp = x
            s = torch.sin(x @ w1 + b1)
            x = u * s + v * (1 - s)
            s = torch.sin(x @ w2 + b2)
            x = u * s + v * (1 - s)
            y = x @ w3 + b3
            x = (1 - g) * x_tmp + g * torch.sin(y)

        w, b = r['final']
        return r['norm'](x @ w + b)

    def embed(self, x):
        """Configs (N, dim) -> trunk embedding (N, h_size)."""
        y = torch.cat([
            self._route(x[:, :self.half], self.W_trans, self.route_t),
            self._route(x[:, self.half:], self.W_rot, self.route_r),
        ], dim=-1)

        w, b = self.fuse[0]
        res = y @ w + b
        for i in range(self.fuse_len):
            w, b = self.fuse[2 * i + 1]
            y1 = self.act(res @ w + b)
            w, b = self.fuse[2 * i + 2]
            res = self.act(res + (y1 @ w + b))
        return res

    @staticmethod
    def tau(e0, e1):
        """Travel time between two trunk embeddings (e1 may broadcast)."""
        x = torch.sqrt((e0 - e1) ** 2 + 1e-6)
        x = x.view(x.shape[0], -1, 16)
        x = (torch.logsumexp(10 * x, 2) - np.log(16)) / 10
        return 0.2 * torch.sum(x, dim=1)

    def travel_times(self, Xp):
        """Reference-compatible entry point: (N, 2*dim) -> (N,)."""
        return self.tau(self.embed(Xp[:, :self.dim]), self.embed(Xp[:, self.dim:]))


# ──────────────────────────────────────────────────────────────────────────────
# MPPI
# ──────────────────────────────────────────────────────────────────────────────
def mppi(field, XP, dim, steps, sample_num, horizon, check_every, buf):
    """Plan one path.  Returns (n_waypoints, iterations, reached_goal).

    Same algorithm as ``evaluate_training_3d.MPPI``.  ``buf`` is a dict of
    preallocated tensors reused across episodes so the loop performs no
    allocations beyond the intermediates of the field evaluation itself.
    """
    path, noise_a, noise_b = buf['path'], buf['noise_a'], buf['noise_b']
    dP_prior = buf['prior'].zero_()

    start = XP[:, 0:dim]
    goal = XP[:, dim:2 * dim]
    emb_goal = field.embed(goal)          # constant for the whole episode

    path[0] = start[0]
    n = 1
    reached = False
    it = 0

    for it in range(steps):
        dP = 0.015 * noise_a.normal_() + 0.015 * noise_b.normal_()
        dP = dP + 2 * dP_prior
        dP = dP / (torch.clamp(torch.norm(dP, dim=2, keepdim=True), min=0.015) / 0.015)

        # Only horizon steps 0 and -1 are ever queried, so build just those two.
        dPc = torch.cumsum(dP, dim=1)
        sel = torch.stack((dPc[:, 0], dPc[:, -1]), dim=1).reshape(-1, dim)
        cand = start + sel                                     # (2*sample_num, dim)

        cost = field.tau(field.embed(cand), emb_goal).reshape(-1, 2)
        cost = 10 * cost[:, 0] + cost[:, 1]

        weight = torch.softmax(-50 * cost, dim=0)
        dP_prior = (weight @ dP[:, 0, :]).unsqueeze(0)

        start = start + dP_prior.squeeze(0)
        path[n] = start[0]
        n += 1

        if (it + 1) % check_every == 0 and torch.norm(goal - start).item() < 0.01:
            reached = True
            break

    path[n] = goal[0]
    n += 1
    return n, it, reached


class GraphMPPI:
    """CUDA-graph MPPI: one iteration captured once, replayed per step.

    The loop's shapes are all static, so the whole per-iteration kernel sequence
    can be recorded as a graph and handed to the GPU with a single CPU call.
    This removes only the launch overhead -- the kernels, their order and their
    arithmetic are identical to ``mppi`` above.

    Everything the captured region touches must live in a preallocated buffer
    (a graph records memory addresses, not values), so the loop state is updated
    with in-place ops: ``copy_`` / ``add_`` for the pose, and ``index_copy_``
    against an in-graph counter for waypoint recording.  The two things that
    cannot be captured stay outside: the goal-reached test (it needs a
    GPU->CPU sync) and the per-episode buffer initialization.

    RNG survives capture -- PyTorch registers the generator during capture and
    advances the philox offset per replay, so each replay draws fresh noise
    rather than repeating the sequence recorded at capture time.
    """

    def __init__(self, field, dim, steps, sample_num, horizon, device):
        self.field = field
        self.dim = dim
        self.steps = steps

        with torch.no_grad():
            h = field.embed(torch.zeros((1, dim), device=device)).shape[1]

        z = lambda *s, dt=torch.float32: torch.zeros(s, device=device, dtype=dt)
        self.st = {
            'start': z(1, dim),
            'goal': z(1, dim),
            'prior': z(1, dim),
            'emb_goal': z(1, h),
            'na': torch.empty((sample_num, 1, dim), device=device),
            'nb': torch.empty((sample_num, horizon, dim), device=device),
            'path': z(steps + 2, dim),
            'ctr': z(1, dt=torch.long),
            'dist': torch.zeros((), device=device),
        }

        # Warm up on a side stream first: cuBLAS/cuDNN allocate their workspaces
        # on first call, and that allocation must not end up inside the graph.
        with torch.no_grad():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(5):
                    self._step()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()

            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._step()

    def _step(self):
        """One MPPI iteration, written entirely into the static buffers."""
        st, dim = self.st, self.dim

        dP = 0.015 * st['na'].normal_() + 0.015 * st['nb'].normal_()
        dP = dP + 2 * st['prior']
        dP = dP / (torch.clamp(torch.norm(dP, dim=2, keepdim=True), min=0.015) / 0.015)

        dPc = torch.cumsum(dP, dim=1)
        sel = torch.stack((dPc[:, 0], dPc[:, -1]), dim=1).reshape(-1, dim)
        cand = st['start'] + sel

        cost = self.field.tau(self.field.embed(cand), st['emb_goal']).reshape(-1, 2)
        cost = 10 * cost[:, 0] + cost[:, 1]

        weight = torch.softmax(-50 * cost, dim=0)
        new_prior = (weight @ dP[:, 0, :]).unsqueeze(0)

        st['prior'].copy_(new_prior)
        st['start'].add_(new_prior)
        st['path'].index_copy_(0, st['ctr'], st['start'])
        st['ctr'].add_(1)
        st['dist'].copy_(torch.norm(st['goal'] - st['start']))

    def run(self, XP, check_every):
        st, dim = self.st, self.dim

        st['start'].copy_(XP[:, :dim])
        st['goal'].copy_(XP[:, dim:])
        st['emb_goal'].copy_(self.field.embed(st['goal']))
        st['prior'].zero_()
        st['path'][0].copy_(st['start'][0])
        st['ctr'].fill_(1)

        reached = False
        it = 0
        for it in range(self.steps):
            self.graph.replay()
            if (it + 1) % check_every == 0 and st['dist'].item() < 0.01:
                reached = True
                break

        n = int(st['ctr'].item())
        st['path'][n].copy_(st['goal'][0])
        return n + 1, it, reached


# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────
modelPath = args.modelPath
dataPath = args.dataPath

if args.checkpoint is not None:
    pt = args.checkpoint
else:
    latest = os.path.join(modelPath, 'latest.pt')
    if os.path.exists(latest):
        pt = latest
    else:
        ckpts = sorted(glob(os.path.join(modelPath, '*', 'Model_Epoch_*.pt')))
        if not ckpts:
            raise FileNotFoundError(
                f'No latest.pt and no checkpoints under {modelPath}/*/Model_Epoch_*.pt')
        pt = ckpts[-1]

# NOTE: a hardcoded override of `pt` used to sit here, silently ignoring
# --checkpoint and loading the rectangle_env1 model for every run.  Removed.
womodel = md.Model(modelPath, dataPath, DIM, [0.0] * DIM, device=args.device)
womodel.load(pt)
womodel.network.eval()
for p in womodel.network.parameters():
    p.requires_grad_(False)

print(f'checkpoint : {pt}')
print(f'data       : {dataPath}')
print(f'device     : {args.device}')

arr = np.load(os.path.join(dataPath, 'sampled_points.npy'))       # (N, 2*DIM)
n_ep = min(args.episodes, arr.shape[0])
pairs = torch.tensor(arr[:n_ep], dtype=torch.float32, device=args.device)

field = FastField(womodel.network)

if args.verify:
    with torch.no_grad():
        probe = pairs[:min(64, n_ep)].contiguous()
        ref = womodel.function.TravelTimes(probe).detach()
        fast = field.travel_times(probe)
        err = (ref - fast).abs()
        print(f'verify     : max abs err {err.max().item():.3e}  '
              f'mean {err.mean().item():.3e}  (ref range '
              f'[{ref.min().item():.4f}, {ref.max().item():.4f}])')

cuda = args.device.startswith('cuda')

gm = None
buf = None
if args.graph:
    if not cuda:
        raise SystemExit('--graph requires a CUDA device.')
    gm = GraphMPPI(field, DIM, args.steps, args.samples, args.horizon, args.device)
    path_buf = gm.st['path']
    # A graph replay is not an autograd-visible op, but the per-episode buffer
    # setup still runs the field; no_grad (not inference_mode) keeps the static
    # buffers ordinary tensors so the captured kernels can write to them.
    run_ctx = torch.no_grad()
    print('mode       : CUDA graph (1 captured iteration, replayed)')
else:
    buf = {
        'path': torch.zeros((args.steps + 2, DIM), device=args.device),
        'noise_a': torch.empty((args.samples, 1, DIM), device=args.device),
        'noise_b': torch.empty((args.samples, args.horizon, DIM), device=args.device),
        'prior': torch.zeros((1, DIM), device=args.device),
    }
    path_buf = buf['path']
    run_ctx = torch.inference_mode()
    print('mode       : eager')


# ──────────────────────────────────────────────────────────────────────────────
# Timed loop
# ──────────────────────────────────────────────────────────────────────────────
times_ms = []
iters = []
n_reached = 0
paths = []

with run_ctx:
    for i in range(n_ep):
        XP = pairs[i].reshape(1, 2 * DIM).clone()

        if cuda:
            torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True) if cuda else None
        if cuda:
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
        else:
            import time as _time
            wall0 = _time.perf_counter()

        if gm is not None:
            n, it, reached = gm.run(XP, args.check_every)
        else:
            n, it, reached = mppi(field, XP, DIM, args.steps, args.samples,
                                  args.horizon, args.check_every, buf)

        if cuda:
            t1.record()
            torch.cuda.synchronize()
            dt = t0.elapsed_time(t1)
        else:
            dt = (_time.perf_counter() - wall0) * 1e3

        if args.save_paths:
            paths.append(path_buf[:n].clone().cpu().numpy())

        if i >= args.warmup:
            times_ms.append(dt)
            iters.append(it + 1)
            n_reached += int(reached)

        print(f'[{i:03d}] {"REACHED" if reached else "NO-CONV"}  '
              f'iters={it + 1:3d}  waypoints={n:3d}  {dt:8.2f} ms'
              + ('  (warmup)' if i < args.warmup else ''))

n_timed = len(times_ms)
if n_timed == 0:
    raise SystemExit('No timed episodes -- lower --warmup or raise --episodes.')

times_ms.sort()
print()
print(f'episodes timed   : {n_timed}  (warmup {min(args.warmup, n_ep)} excluded)')
print(f'converged        : {n_reached} / {n_timed}  ({n_reached / n_timed:.1%})   '
      f'<- NOT the success rate')
print( '                   (goal reached within 0.01; this script does no collision')
print( '                    checking, so a path through an obstacle counts here.')
print( '                    Use evaluate_training_3d.py for the real success rate.)')
print(f'mean iterations  : {statistics.mean(iters):.1f}')
print(f'AVERAGE time     : {statistics.mean(times_ms):8.2f} ms')
print(f'MEDIAN  time     : {statistics.median(times_ms):8.2f} ms')
print(f'min / max        : {times_ms[0]:8.2f} / {times_ms[-1]:.2f} ms')
print(f'p90              : {times_ms[int(0.9 * (n_timed - 1))]:8.2f} ms')
print(f'throughput       : {1000.0 / statistics.mean(times_ms):.1f} paths/s')

if args.save_paths:
    np.savez_compressed(args.save_paths,
                        **{f'path_{i:04d}': p for i, p in enumerate(paths)})
    print(f'wrote {args.save_paths}')
