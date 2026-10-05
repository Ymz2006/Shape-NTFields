"""Bidirectional (start->goal OR goal->start) timing benchmark, both directions
planned CONCURRENTLY in one batched MPPI.

Question this answers: if every case is planned twice -- forward and reverse --
and the case counts as solved when EITHER direction converges, does the wall
clock double?

It does not.  The MPPI loop in ``evaluate_training_3d_opt.py`` is kernel-launch
bound, not compute bound (its docstring measures per-iteration cost as flat from
12 to 400 rollout samples).  Both directions are therefore folded into a single
leading batch axis -- the same trick ``evaluate_training_3d_batched.MPPI_batched``
uses to run B cases together -- so one iteration issues the same number of kernel
launches for two directions as for one, just with wider tensors:

    forward-only :  (1, S, H, dim)  ->  1*2*S rows through the field
    bidirectional:  (2, S, H, dim)  ->  2*2*S rows through the field

and the whole iteration is still captured as ONE CUDA graph and replayed.

Convergence / OR semantics
--------------------------
Row 0 plans start->goal, row 1 plans goal->start.  By default the episode stops
as soon as EITHER row enters the 0.01 goal ball -- that is the OR, and it means
the iteration count is min(fwd, rev) rather than max, so bidirectional can be
*faster* per episode than one direction alone.  ``--wait-both`` instead runs
until both rows converge (or --steps is exhausted), which is what you want if
the OR is to be taken over *collision-free* paths, since a path you stopped
early cannot be collision-checked.

Endpoint screening
------------------
``--screen`` drops cases whose start or goal pose is already in collision, using
the exact filter from ``evaluate_training_3d_batched.py`` (lines 143-188 and
738-765): the same point-in-solid test, the same FAST_WINDING_NUMBER sign
convention, the same 50000 surface points over the FULL environment mesh
(obstacles + walls), and the same mesh normalization out of meta.json.  Those
cases are unsolvable by construction and would otherwise be charged to the
planner.

The travel-time field is ``FastField`` lifted verbatim out of
evaluate_training_3d_opt.py (see _load_fastfield) so the two scripts cannot
drift apart -- the arithmetic is identical, only the batching differs.

    python evaluate_training_3d_bidir.py \
        --dataPath ./testing_data/3dshape/rectangle_env1 \
        --checkpoint ./Experiments/3dshape/3dshape_08_06_17_06/latest.pt \
        --episodes 250 --warmup 5 --screen --mode bidir
"""

import sys
sys.path.append('.')

import os
import ast
import json
import argparse
import statistics as st
from glob import glob

import numpy as np
import torch
import igl

from models.metric import model_train_metric as md
from dataprocessing.preprocess_obj import (
    load_obj, _rotvec_to_matrix_np, sample_surface_points)

DIM = 6
_SDF_SIGN = igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
ENV_COLLISION_POINTS = 50000
_OPT_SRC = 'evaluate_training_3d_opt.py'


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Bidirectional (OR) MPPI timing, both directions in parallel.')
parser.add_argument('--dataPath', default='./testing_data/3dshape/rectangle_env1')
parser.add_argument('--modelPath', default='./Experiments/3dshape')
parser.add_argument('--checkpoint', default=None)
parser.add_argument('--episodes', type=int, default=250,
                    help='Test-set rows to walk (before screening).')
parser.add_argument('--timed', type=int, default=100,
                    help='Timed episodes to keep after warmup + screening.')
parser.add_argument('--warmup', type=int, default=5)
parser.add_argument('--steps', type=int, default=120)
parser.add_argument('--samples', type=int, default=50)
parser.add_argument('--horizon', type=int, default=5)
parser.add_argument('--mode', choices=('fwd', 'bidir'), default='bidir',
                    help='fwd = start->goal only (B=1, the opt-script baseline); '
                         'bidir = both directions concurrently (B=2).')
parser.add_argument('--wait-both', action='store_true', dest='wait_both',
                    help='Run until BOTH directions converge instead of stopping '
                         'at the first (needed if the OR is over collision-free '
                         'paths rather than over convergence).')
parser.add_argument('--screen', action='store_true',
                    help='Drop cases whose start or goal pose is in collision.')
parser.add_argument('--no-graph', action='store_true', dest='no_graph',
                    help='Disable CUDA-graph capture (eager loop).')
parser.add_argument('--verify', action='store_true')
parser.add_argument('--device', default='cuda')
args = parser.parse_args()

B = 2 if args.mode == 'bidir' else 1


# ──────────────────────────────────────────────────────────────────────────────
# FastField, lifted from the opt script so the field math cannot drift
# ──────────────────────────────────────────────────────────────────────────────
def _load_fastfield():
    src = open(_OPT_SRC).read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'FastField':
            block = ast.get_source_segment(src, node)
            ns = {'torch': torch, 'np': np}
            exec(block, ns)
            return ns['FastField']
    raise RuntimeError(f'FastField not found in {_OPT_SRC}')


FastField = _load_fastfield()


def tau_b(e0, e1):
    """Batched form of ``FastField.tau``: (B, M, h) x (B, 1, h) -> (B, M).

    Same arithmetic as the 2-D version applied row by row -- the 16-wide
    logsumexp groups, the log(16) offset and the 0.2 scale are untouched; only a
    leading batch axis is added.
    """
    x = torch.sqrt((e0 - e1) ** 2 + 1e-6)
    b, m, _ = x.shape
    x = x.view(b, m, -1, 16)
    x = (torch.logsumexp(10 * x, 3) - np.log(16)) / 10
    return 0.2 * torch.sum(x, dim=2)


# ──────────────────────────────────────────────────────────────────────────────
# Batched MPPI over B directions, one captured CUDA graph
# ──────────────────────────────────────────────────────────────────────────────
class BiMPPI:
    """B independent rollouts stepped together; identical math to opt's GraphMPPI.

    Every buffer carries a leading (B,) axis and the per-iteration reduction is
    done per row, so the B rollouts never interact.  With B=1 this reproduces
    GraphMPPI exactly; with B=2 rows 0/1 are the forward and reverse plans.
    """

    def __init__(self, field, dim, steps, sample_num, horizon, device, b, use_graph):
        self.field, self.dim, self.steps = field, dim, steps
        self.B, self.S, self.H = b, sample_num, horizon
        self.use_graph = use_graph

        with torch.no_grad():
            h = field.embed(torch.zeros((1, dim), device=device)).shape[1]

        z = lambda *s, dt=torch.float32: torch.zeros(s, device=device, dtype=dt)
        self.st = {
            'start': z(b, dim),
            'goal': z(b, dim),
            'prior': z(b, dim),
            'emb_goal': z(b, h),
            'na': torch.empty((b, sample_num, 1, dim), device=device),
            'nb': torch.empty((b, sample_num, horizon, dim), device=device),
            'path': z(b, steps + 2, dim),
            'ctr': z(1, dt=torch.long),
            'dist': z(b),
        }

        if not use_graph:
            self.graph = None
            return

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
        st, dim, b, S = self.st, self.dim, self.B, self.S

        dP = 0.015 * st['na'].normal_() + 0.015 * st['nb'].normal_()   # (B,S,H,dim)
        dP = dP + 2 * st['prior'][:, None, None, :]
        dP = dP / (torch.clamp(torch.norm(dP, dim=3, keepdim=True), min=0.015) / 0.015)

        dPc = torch.cumsum(dP, dim=2)
        sel = torch.stack((dPc[:, :, 0], dPc[:, :, -1]), dim=2)        # (B,S,2,dim)
        cand = st['start'][:, None, None, :] + sel

        emb = self.field.embed(cand.reshape(b * S * 2, dim)).view(b, S * 2, -1)
        cost = tau_b(emb, st['emb_goal'][:, None, :]).view(b, S, 2)
        cost = 10 * cost[:, :, 0] + cost[:, :, 1]                      # (B,S)

        weight = torch.softmax(-50 * cost, dim=1)
        new_prior = torch.einsum('bs,bsd->bd', weight, dP[:, :, 0, :])

        st['prior'].copy_(new_prior)
        st['start'].add_(new_prior)
        st['path'].index_copy_(1, st['ctr'], st['start'][:, None, :])
        st['ctr'].add_(1)
        st['dist'].copy_(torch.norm(st['goal'] - st['start'], dim=1))

    def run(self, XP, wait_both):
        """XP: (B, 2*dim).  Returns (n_waypoints, iterations, reached_per_row)."""
        st, dim = self.st, self.dim

        st['start'].copy_(XP[:, :dim])
        st['goal'].copy_(XP[:, dim:])
        st['emb_goal'].copy_(self.field.embed(st['goal']))
        st['prior'].zero_()
        st['path'][:, 0].copy_(st['start'])
        st['ctr'].fill_(1)

        hit = [False] * self.B
        it = 0
        for it in range(self.steps):
            if self.graph is not None:
                self.graph.replay()
            else:
                self._step()
            d = st['dist'].tolist()
            for k in range(self.B):
                if d[k] < 0.01:
                    hit[k] = True
            if (all(hit) if wait_both else any(hit)):
                break

        n = int(st['ctr'].item())
        st['path'][:, n].copy_(st['goal'])
        return n + 1, it, hit


# ──────────────────────────────────────────────────────────────────────────────
# Endpoint collision screen (evaluate_training_3d_batched.py:143-188, 738-765)
# ──────────────────────────────────────────────────────────────────────────────
def _resolve(p, data_path):
    """meta.json stores absolute paths from the container that produced the data;
    fall back to the local datasets dir by basename when they do not resolve."""
    if os.path.exists(p):
        return p
    alt = os.path.join('./datasets/3dshape', os.path.basename(p))
    if os.path.exists(alt):
        return alt
    raise FileNotFoundError(f'{p} (and {alt})')


def build_screen(data_path):
    meta = json.load(open(os.path.join(data_path, 'meta.json')))
    env_scale = float(meta['env_scale'])
    env_center = np.asarray(meta['env_center'], dtype=np.float64)
    shape_scale = float(meta['shape_scale'])

    V_sh, F_sh, _ = load_obj(_resolve(meta['shape_obj'], data_path))
    shape_center = 0.5 * (V_sh.min(axis=0) + V_sh.max(axis=0))
    shape_V = np.ascontiguousarray((V_sh - shape_center) / env_scale * shape_scale,
                                   dtype=np.float64)
    shape_F = np.ascontiguousarray(F_sh, dtype=np.int64)
    shape_radius = float(np.linalg.norm(shape_V, axis=1).max())

    V_env, F_env, _ = load_obj(_resolve(meta['env_obj'], data_path))
    V_env_n = (V_env - env_center) / env_scale
    env_pts = np.ascontiguousarray(
        sample_surface_points(V_env_n, F_env, ENV_COLLISION_POINTS), dtype=np.float64)

    def in_collision(cfg):
        cfg = np.asarray(cfg).reshape(-1)
        t = cfg[0:3]
        near = env_pts[np.linalg.norm(env_pts - t, axis=1) <= shape_radius]
        if near.shape[0] == 0:
            return False
        R = _rotvec_to_matrix_np(cfg[3:6] * (2 * np.pi))
        near_local = np.ascontiguousarray((near - t) @ R)
        S = igl.signed_distance(near_local, shape_V, shape_F, _SDF_SIGN)[0]
        return bool(S.size and S.min() < 0.0)

    return in_collision


# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────
if args.checkpoint is not None:
    pt = args.checkpoint
else:
    latest = os.path.join(args.modelPath, 'latest.pt')
    pt = latest if os.path.exists(latest) else sorted(
        glob(os.path.join(args.modelPath, '*', 'Model_Epoch_*.pt')))[-1]

womodel = md.Model(args.modelPath, args.dataPath, DIM, [0.0] * DIM, device=args.device)
womodel.load(pt)
womodel.network.eval()
for p in womodel.network.parameters():
    p.requires_grad_(False)

print(f'checkpoint : {pt}')
print(f'data       : {args.dataPath}')
print(f'mode       : {args.mode} (B={B})'
      + ('  wait-both' if args.wait_both else '  stop-on-first (OR)'))

arr = np.load(os.path.join(args.dataPath, 'sampled_points.npy'))
n_walk = min(args.episodes, arr.shape[0])

valid = list(range(n_walk))
if args.screen:
    in_collision = build_screen(args.dataPath)
    valid, bad_s, bad_g = [], 0, 0
    for i in range(n_walk):
        s_bad = in_collision(arr[i][0:DIM])
        g_bad = in_collision(arr[i][DIM:2 * DIM])
        if s_bad or g_bad:
            bad_s += int(s_bad); bad_g += int(g_bad)
        else:
            valid.append(i)
    print(f'screen     : {len(valid)}/{n_walk} endpoint-valid '
          f'(bad_start={bad_s} bad_goal={bad_g})')

field = FastField(womodel.network)

if args.verify:
    with torch.no_grad():
        probe = torch.tensor(arr[:64], dtype=torch.float32, device=args.device)
        ref = womodel.function.TravelTimes(probe).detach()
        err = (ref - field.travel_times(probe)).abs()
        print(f'verify     : max abs err {err.max().item():.3e}')

use_graph = (not args.no_graph) and args.device.startswith('cuda')
mp = BiMPPI(field, DIM, args.steps, args.samples, args.horizon,
            args.device, B, use_graph)
print(f'graph      : {use_graph}')


# ──────────────────────────────────────────────────────────────────────────────
# Timed loop
# ──────────────────────────────────────────────────────────────────────────────
times, iters, ok_fwd, ok_rev, ok_or = [], [], 0, 0, 0
n_done = 0

with torch.no_grad():
    for pos, i in enumerate(valid):
        row = torch.tensor(arr[i], dtype=torch.float32, device=args.device)
        if B == 2:
            XP = torch.stack((row, torch.cat((row[DIM:], row[:DIM]))))
        else:
            XP = row.reshape(1, 2 * DIM)

        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        n, it, hit = mp.run(XP, args.wait_both)
        t1.record()
        torch.cuda.synchronize()
        dt = t0.elapsed_time(t1)

        tag = 'WARM' if pos < args.warmup else ('OK  ' if any(hit) else 'FAIL')
        if pos >= args.warmup:
            times.append(dt)
            iters.append(it + 1)
            ok_fwd += int(hit[0])
            ok_rev += int(hit[-1])
            ok_or += int(any(hit))
            n_done += 1
        print(f'[{i:03d}] {tag}  iters={it + 1:3d}  '
              f'fwd={"Y" if hit[0] else "n"} rev={"Y" if hit[-1] else "n"}  {dt:8.2f} ms')
        if n_done >= args.timed:
            break

if not times:
    raise SystemExit('No timed episodes.')

s = sorted(times)
pi = [a / b for a, b in zip(times, iters)]
print()
print(f'timed episodes   : {len(times)}')
print(f'MEAN             : {st.mean(times):8.2f} ms')
print(f'MEDIAN           : {st.median(times):8.2f} ms')
print(f'STDEV            : {st.stdev(times):8.2f} ms')
print(f'min / max        : {s[0]:8.2f} / {s[-1]:.2f} ms')
print(f'p90              : {s[int(0.9 * (len(s) - 1))]:8.2f} ms')
print(f'iterations       : {st.mean(iters):.1f} +/- {st.stdev(iters):.1f}')
print(f'ms per iteration : {st.mean(pi):.3f} +/- {st.stdev(pi):.3f}')
print(f'converged fwd    : {ok_fwd}/{len(times)}')
if B == 2:
    print(f'converged rev    : {ok_rev}/{len(times)}')
    print(f'converged OR     : {ok_or}/{len(times)}')
