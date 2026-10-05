"""Batched (GPU) evaluation for the 3-D shape planner.

Functionally identical to ``evaluate_training_3d.py`` -- same MPPI rollout, same
collision / convergence / linear-interp accounting, same success_rate.txt, same
plotly summaries, and the same interactive viser viewer -- but the MPPI rollouts
for many start/goal pairs are run *simultaneously* on the GPU instead of one
episode at a time.  This is the dominant cost (repeated ``TravelTimes`` network
evaluations), so batching gives a large speedup.

The per-waypoint collision test still runs on the CPU through ``igl`` (it is the
exact same point-in-solid check used by the non-batched script, so the pass/fail
labels are bit-for-bit the same); only the network-driven rollout is batched.

Episodes are processed in chunks of ``--batch`` at a time to bound GPU memory.

Before any rollout, every test case is screened with the same mesh collision
check at its START and GOAL pose.  A case whose start or goal already overlaps
the environment is unsolvable by construction, so it is discarded outright and
excluded from the denominator -- the reported success rate is therefore the
actual rate over solvable cases, not diluted by degenerate test data.

By default the console shows only the per-planner success rates
(``regular/forward``, ``flipped/reverse``, ``either/or``, ``alt/alternate``,
``locB/Alternative Bellman``, ``hlB/Alternative Bellman Horizon``).  Pass
``--verbose`` for the full diagnostic log -- per-case PASS/FAIL lines, the
failure breakdowns and the clearance frequency tables.  The file artifacts
(success_rate.txt, the summary HTML plots) are written either way.

Run from the nested ntrl-demo root:

    python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/rectangle_env1 --out ./results/output_3d/rectangle_env1 --batch 250
"""


import sys
sys.path.append('.')


import os
import io
import json
import importlib
import math
import time
import argparse
import contextlib
from glob import glob
from timeit import default_timer as timer


import numpy as np
import torch
import igl
import plotly.graph_objects as go
# ``viser`` is imported lazily inside launch_viser() so headless runs
# (``--no-viser``) work on machines where viser is not installed.


# The model package is selected by --models after argument parsing (see below);
# ``md`` is bound there so a checkpoint from a different package -- e.g. the
# single-route models/metric_june03 -- can be evaluated by this same script.
from dataprocessing.preprocess_obj import (
    load_obj, _rotvec_to_matrix_np, sample_surface_points)


# Sign convention for igl.signed_distance: the DEFAULT/WINDING_NUMBER types are
# unreliable in this binding, but FAST_WINDING_NUMBER returns negative-inside and
# is robust to per-face orientation (only requires a closed/watertight mesh).
_SDF_SIGN = igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
# Number of surface points sampled over the full environment mesh for collision.
ENV_COLLISION_POINTS = 50000
LOCAL_W = 0.03                # set from --local-weight; scales the step cost.
# 0.03 from a sweep on Ashape3d_env1 (200 cases): A+B 85.5% vs 76.6% at w=0.
# The usable window is narrow -- 0.15 collapses the planner to ~19% because the
# normalized slowness term then outweighs cost-to-go (its spread across samples
# is ~14x larger). Do not raise this much above 0.07.
HTTP_PORT = 8081
DIM = 6
# Largest number of config PAIRS handed to a single TravelTimes call inside
# cost_to_go_indep.  The split evaluates 4 legs per candidate, so at --batch 250
# with all_horizon it would otherwise push 250k pairs (= 500k rows, since the net
# stacks both endpoints) through the network at once.  Chunking bounds the
# transient activation memory without changing any result.
CTG_PAIR_CHUNK = 200000


parser = argparse.ArgumentParser(
    description='Batched (GPU) evaluation for the 3-D shape planner.')
parser.add_argument('--dataPath', default='./testing_data/3dshape/rectangle_env1',
                    help='Directory holding the test data (sampled_points.npy, '
                         'speed.npy, env.npy, meta.json) used as start/goal pairs.')
parser.add_argument('--out', default='./output_3d',
                    help='Output directory for the rendered HTML plots (also the '
                         'web root served over HTTP).')
parser.add_argument('--batch', type=int, default=250,
                    help='Number of episodes whose MPPI rollouts are run together '
                         'on the GPU per chunk (bounds GPU memory).')
parser.add_argument('--device', default='cuda',
                    help='Torch device the network and the batched rollouts run '
                         'on (e.g. cuda, cuda:2, cpu).')
parser.add_argument('--nbins_dist', type=int, default=8,
                    help='speed_dist bins (uniform over [0,1]) for the end-of-run '
                         'endpoint frequency tables.')
parser.add_argument('--nbins_angle', type=int, default=8,
                    help='speed_angle bins (uniform over [0,1]) for the end-of-run '
                         'endpoint frequency tables.')
parser.add_argument('--momentum', type=float, default=2.0,
                    help='Gain on the previous accepted step, added to every MPPI '
                         'sample before the magnitude clamp. 2.0 is the original '
                         'sampler; 0 disables momentum.')
parser.add_argument('--step', type=float, default=0.015,
                    help='Per-sample displacement cap in the normalized 6-D config '
                         'space (rotvec / 2*pi).  Note the convergence ball is 0.01, '
                         'so the default stride is LARGER than the target.')
parser.add_argument('--temp', type=float, default=50.0,
                    help='MPPI softmax temperature: sample weights are softmax(-temp * cost). '
                         'The default 50 is calibrated to fields with |grad tau| ~ 5; a field '
                         'that learned |grad tau| ~ 1 (lots of speed-1 data, e.g. the '
                         'drone_small_Aloha cell) needs a proportionally larger value or the '
                         'weights flatten and the rollout stalls.')
parser.add_argument('--taper', type=float, default=0.0,
                    help='If > 0, shrink the sampling radius and executed step to '
                         'taper*dist_to_goal on final approach (floor 0.1*step), so '
                         'the stride cannot overshoot the convergence ball. '
                         '0 = original fixed-stride behaviour.')
parser.add_argument('--2d', dest='two_d', action='store_true',
                    help='Planar mode: restrict the MPPI rollout to the same 2-D '
                         'sub-space preprocess_obj.py --2d samples in -- x, y and '
                         'rotation about z only, with z / rx / ry held at their '
                         'start values (0 in --2d data).  Must match how the test '
                         'set was generated.')
parser.add_argument('--cases', type=int, default=1000,
                    help='How many start/goal pairs from sampled_points.npy to '
                         'evaluate. 0 (the default) means ALL of them, which is '
                         'what the reported success rate should normally be over; '
                         'set a small number only for a quick smoke test.')
parser.add_argument('--local-weight', dest='local_weight', type=float, default=0.03,
                    help='Weight on the tau(current -> candidate) step cost used '
                         'by the local-step and horizon+local planners.')
parser.add_argument('--models', default='metric',
                    help='Model package under models/ holding the network the '
                         'checkpoint was trained with, e.g. metric (default) or '
                         'metric_june03. The June-3 packages are a different '
                         'architecture, so their checkpoints only load there.')
parser.add_argument('--modelPath', default='./Experiments/3dshape',
                    help='Experiment root searched for latest.pt (or the newest '
                         '*/Model_Epoch_*.pt) when --checkpoint is not given.')
parser.add_argument('--checkpoint', default=None,
                    help='Explicit .pt to evaluate.  Without this the script '
                         'silently takes whatever latest.pt happens to be in '
                         '--modelPath, which is usually the last model TRAINED, '
                         'not the one matching --dataPath.')
parser.add_argument('--no-viser', action='store_true',
                    help='Skip launching the interactive viser viewer (still '
                         'writes success_rate.txt and the summary HTML plots).')
parser.add_argument('--verbose', '-v', action='store_true',
                    help='Print the full diagnostic log: per-case PASS/FAIL lines, '
                         'endpoint-filter and timing lines, the failure breakdowns '
                         '(rescued / lost / collision / no-convergence, phi, closest '
                         'approach) and the clearance frequency tables.  Without it '
                         'only the per-planner success rates are printed; every file '
                         'artifact (success_rate.txt, the summary HTML plots) is '
                         'written either way.')
args = parser.parse_args()

md = importlib.import_module('models.{}.model_train_metric'.format(args.models))


VERBOSE = args.verbose


def _quiet():
    """Swallow stdout of an imported helper (load_obj's mesh listing) unless --verbose."""
    return (contextlib.nullcontext() if VERBOSE
            else contextlib.redirect_stdout(io.StringIO()))


def vprint(*a, **kw):
    """print() that is silenced unless --verbose was passed.

    Everything diagnostic goes through this; only the success-rate report and
    genuine warnings use bare print().
    """
    if VERBOSE:
        print(*a, **kw)


OUTPUT_DIR = args.out
LOCAL_W = args.local_weight
EPISODE_BATCH = args.batch
DEVICE = args.device
# MPPI sampler tunables, read as defaults inside MPPI_batched.
MOMENTUM = args.momentum
STEP = args.step
TAPER = args.taper
# Planar (--2d) mode.  preprocess_obj.py's ``_sample_configs(two_d=True)`` draws
# placements with z == 0 and a rotvec of (0, 0, rz), so the only free coordinates
# of the normalized 6-D config are x, y and rz -- indices 0, 1 and 5.  MPPI must
# sample in that same sub-space or it plans through configs the field never saw.
PLANAR = args.two_d
PLANAR_FREE_DIMS = (0, 1, 5)


os.makedirs(OUTPUT_DIR, exist_ok=True)


# Clear stale artifacts from any previous run so old episode_*.html (with a
# different success/fail suffix) don't linger alongside this run's output.
for _stale in (glob(os.path.join(OUTPUT_DIR, 'episode_*.html'))
               + glob(os.path.join(OUTPUT_DIR, 'summary_*.html'))
               + glob(os.path.join(OUTPUT_DIR, 'success_rate.txt'))):
    os.remove(_stale)




# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def check_trajectory_collision(traj, env_pts, shape_V, shape_F, shape_radius):
    """Return True if the placed shape mesh overlaps the environment at ANY waypoint.


    Collision uses the same point-in-solid test that labels the training data
    (preprocess_obj.points_inside_tets): an environment surface point lying
    *inside* the watertight shape mesh.  The full 6-DOF pose (position + rotvec)
    is applied at each waypoint, and ``env_pts`` densely covers the whole
    environment surface (obstacles *and* walls), so everything counts as an
    obstacle.  No interpolation between waypoints -- each is checked on its own.


    ``traj`` is a list of (1, DIM) configs with the rotvec stored normalized by
    2*pi (as written by preprocess_obj); ``shape_V``/``shape_F`` is the shape
    mesh in its local frame; ``shape_radius`` is max ||vertex|| about that frame.
    """
    two_pi = 2 * np.pi
    for cfg_t in traj:
        cfg = cfg_t.detach().cpu().numpy().reshape(-1)
        t = cfg[0:3]
        # Broad phase: a point inside the shape must lie within its bounding radius.
        near = env_pts[np.linalg.norm(env_pts - t, axis=1) <= shape_radius]
        if near.shape[0] == 0:
            continue
        # Map env points into the shape's local frame (inverse of _placed_mesh,
        # which does  world = local @ R.T + t  =>  local = (world - t) @ R).
        R = _rotvec_to_matrix_np(cfg[3:6] * two_pi)
        near_local = np.ascontiguousarray((near - t) @ R)
        S = igl.signed_distance(near_local, shape_V, shape_F, _SDF_SIGN)[0]
        if S.size and S.min() < 0.0:
            return True
    return False




def check_config_collision(cfg, env_pts, shape_V, shape_F, shape_radius):
    """Return True if the shape placed at a single config overlaps the environment.


    Thin wrapper around ``check_trajectory_collision`` for a one-waypoint
    "trajectory"; ``cfg`` is any tensor holding DIM normalized coordinates.
    """
    return check_trajectory_collision(
        [cfg.reshape(1, -1)], env_pts, shape_V, shape_F, shape_radius)




def linear_interp_traj(start_cfg, goal_cfg, n_steps=50):
    """Build a list of (1, DIM) configs linearly interpolating start -> goal.


    Configs are returned in the same normalized space (rotvec stored / 2*pi)
    that ``check_trajectory_collision`` consumes, so the straight-line path can
    be collision-checked exactly like an MPPI trajectory.  Used to measure
    whether the trivial linear interpolation from start to goal is itself a
    valid (collision-free) path.
    """
    start = start_cfg.reshape(1, -1)
    goal = goal_cfg.reshape(1, -1)
    alphas = torch.linspace(0.0, 1.0, n_steps, device=start.device).reshape(-1, 1)
    interp = (1.0 - alphas) * start + alphas * goal
    return [interp[i:i + 1] for i in range(n_steps)]




def MPPI_batched(womodel, XP, dim, steps=200,
                 momentum=None, step=None, taper=None, planar=None):
    """Batched MPPI: run ``B`` start/goal pairs' rollouts together on the GPU.


    ``XP`` is a (B, 2*dim) tensor on the run device (each row ``[start(dim) | goal(dim)]``,
    rotvec stored normalized by 2*pi).  This is the vectorized-over-episodes form
    of the single-episode ``MPPI`` in evaluate_training_3d.py: identical sampling,
    cost, softmax-weighting and convergence rule, just with a leading batch axis.


    Once an episode's current config reaches the goal (``dis < 0.01``) it is
    *frozen* (its config stops updating) exactly like the single-episode version
    breaks out of the loop; other episodes in the batch keep stepping.  Episodes
    are fully independent within the batch, so freezing one does not affect any
    other.


    ``steps`` caps the number of MPPI iterations (the loop still exits early once
    every episode has converged).


    ``momentum`` / ``step`` / ``taper`` default to the module-level MOMENTUM /
    STEP / TAPER (set from the CLI).  ``momentum`` is the gain on the previous
    accepted step that is added to every sample BEFORE the magnitude clamp, so it
    biases the whole 50-sample cloud in one direction rather than just the mean;
    0 disables it.  ``step`` is the per-sample displacement cap.  ``taper`` > 0
    shrinks both the sampling radius and the executed step to
    ``taper * dist_to_goal`` once that is below ``step``, so the stride cannot
    exceed the 0.01 convergence ball on the final approach (with a floor at
    0.1*step so a converging episode cannot stall).


    ``planar`` (defaults to the module-level PLANAR, set by ``--2d``) restricts
    sampling to the 2-D sub-space ``preprocess_obj.py --2d`` generates data in:
    the displacement is zeroed on every coordinate outside PLANAR_FREE_DIMS
    (x, y, rz), so z / rx / ry keep their start values for the whole rollout --
    exactly 0 for a ``--2d`` test set.  The magnitude clamp is applied after
    that projection, so the stride is measured in the free sub-space only and a
    planar rollout gets the same per-step displacement budget in x-y-rz that a
    full 6-D one gets across all six coordinates.

    Returns, per episode, the same objects the single-episode loop expects:
        points_list : list (len B) of lists of (1, dim) configs -- the recorded
                      trajectory up to convergence with the goal config appended
                      last (matches ``point0`` from the single-episode MPPI).
        iters       : list (len B) of the convergence iteration index.
        success     : list (len B) of bools.
        min_dis     : (B,) numpy array, the closest the rollout EVER came to the
                      goal.  For a non-converged episode this separates "circled
                      the goal but could not stick the landing" (min_dis just
                      above the 0.01 ball) from "never got near it".
    """
    B = XP.shape[0]
    sample_num = 50
    horizon = 5
    dev = XP.device
    momentum = MOMENTUM if momentum is None else momentum
    momentum = 0
    step = STEP if step is None else step
    taper = TAPER if taper is None else taper
    planar = PLANAR if planar is None else planar
    if planar and dim != 6:
        raise ValueError('planar (--2d) mode assumes the SE(3) layout '
                         '(x,y,z,rx,ry,rz); got dim={}'.format(dim))
    # (dim,) 1.0 on the free coordinates, 0.0 on the frozen ones.  Broadcasts over
    # the (B, sample_num, horizon, dim) displacement tensor below.
    free_mask = None
    if planar:
        free_mask = torch.zeros(dim, device=dev)
        free_mask[list(PLANAR_FREE_DIMS)] = 1.0


    dP_prior = torch.zeros((B, dim), device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    conv_step = torch.full((B,), steps - 1, dtype=torch.long, device=dev)
    min_dis = torch.norm(XP[:, dim:dim * 2] - XP[:, 0:dim], dim=1)   # (B,)


    # recorded[k] is the (B, dim) current config after k updates (recorded[0] = start).
    recorded = [XP[:, 0:dim].clone()]


    for it in range(steps):
        XP_tmp = XP.clone()
        # (B, sample_num, horizon, 2*dim)
        XP_tmp = XP_tmp[:, None, None, :].repeat(1, sample_num, horizon, 1)


        # Per-episode stride cap.  Constant `step` reproduces the original
        # sampler; with taper it closes down as the goal is approached.
        if taper > 0:
            dis_cur = torch.norm(XP[:, dim:dim * 2] - XP[:, 0:dim], dim=1)
            radius = torch.clamp(dis_cur * taper, max=step, min=step * 0.1)
        else:
            radius = torch.full((B,), step, device=dev)
        r = radius[:, None, None, None]


        dP = step * torch.normal(0, 1, size=(B, sample_num, 1, dim),
                                 dtype=torch.float32, device=dev) \
            + step * torch.normal(0, 1, size=(B, sample_num, horizon, dim),
                                  dtype=torch.float32, device=dev)
        if momentum:
            dP = dP + momentum * dP_prior[:, None, None, :]
        if free_mask is not None:
            # Project onto the planar sub-space BEFORE the magnitude clamp, so
            # the norm below is taken over the free coordinates alone.  Every
            # downstream quantity (dP_cumsum, step_prior, dP_prior) inherits the
            # zeros, hence z / rx / ry are never written during the rollout.
            dP = dP * free_mask
        dP_norm = torch.norm(dP, dim=3, keepdim=True)
        dP = dP / (torch.clamp(dP_norm, min=r) / r)
        dP_cumsum = torch.cumsum(dP, dim=2)
        XP_tmp[..., 0:dim] = XP_tmp[..., 0:dim] + dP_cumsum


        # First and last horizon step of every sample -> (B, sample_num, 2, 2*dim).
        endpoints = XP_tmp[:, :, [0, -1], :]
        cost = womodel.function.TravelTimes(endpoints.reshape(-1, dim * 2))
        cost = cost.reshape(B, sample_num, 2)
        cost = 10 * cost[:, :, 0] + cost[:, :, 1]           # (B, sample_num)


        weight = torch.softmax(-args.temp * cost, dim=1)           # (B, sample_num)
        # Weighted mean of the first-step displacement over samples -> (B, dim).
        step_prior = torch.bmm(weight.unsqueeze(1), dP[:, :, 0, :]).squeeze(1)
        dP_prior = step_prior


        # Freeze converged episodes: only advance those not yet done.
        XP[:, 0:dim] = XP[:, 0:dim] + step_prior * (~done).unsqueeze(1)


        dis = torch.norm(XP[:, dim:dim * 2] - XP[:, 0:dim], dim=1)   # (B,)
        min_dis = torch.minimum(min_dis, dis)
        recorded.append(XP[:, 0:dim].clone())


        newly = (dis < 0.01) & (~done)
        conv_step[newly] = it
        done = done | (dis < 0.01)
        if bool(done.all()):
            break


    success = done.detach().cpu().numpy()
    conv_step_cpu = conv_step.detach().cpu().numpy()
    n_recorded = len(recorded)
    goal = XP[:, dim:dim * 2]


    points_list = []
    iters = []
    for b in range(B):
        if success[b]:
            # start + (conv_step+1) updates  ->  recorded[0 .. conv_step+1].
            end = int(conv_step_cpu[b]) + 2
        else:
            end = n_recorded
        cfgs = [recorded[k][b:b + 1, :] for k in range(end)]
        cfgs.append(goal[b:b + 1, :])      # append goal config, like point0
        points_list.append(cfgs)
        iters.append(int(conv_step_cpu[b]))


    return points_list, iters, success.tolist(), min_dis.detach().cpu().numpy()


def cost_to_go_indep(womodel, pairs, dim):
    """Cost-to-go that times translation and rotation with their OWN speeds.

    ``pairs`` is (..., 2*dim) -- any leading shape, each row [a | b].  Returns the
    travel time with that leading shape.

    The plain cost-to-go is ``tau(a -> b)``: one scalar for a motion that mixes
    translating and rotating.  But the loss constrains the two blocks of the
    gradient SEPARATELY (``model_function_metric.Loss``):

        |grad_trans tau| * s_dist = 1        |grad_rot tau| * s_ang = 1

    so where both hold, |grad tau| = sqrt(s_dist^-2 + s_ang^-2) -- tau charges every
    direction of the 6-D config space the same ISOTROPIC slowness, the L2 sum of the
    two.  A candidate whose remaining motion is pure translation is therefore billed
    for rotational slowness it never incurs, and one that only has to rotate in place
    next to a wall is billed for the translational slowness of that wall.

    This splits the move into a pure-translation leg and a pure-rotation leg and
    reads tau on each:

        tau_t = tau( (p_a, r_a) -> (p_b, r_a) )     rotation held fixed
        tau_r = tau( (p_b, r_a) -> (p_b, r_b) )     translation held fixed

    A displacement inside one leg has no component in the other block, so that leg's
    travel time integrates only its own block of the gradient: tau_t is timed by
    s_dist alone and tau_r by s_ang alone, which is the whole point.  The legs are
    summed (translate, then rotate) and averaged over both L-corners, since neither
    ordering is privileged.

    tau is a metric by construction -- ``NN.out`` returns a logsumexp-aggregated
    distance between the two endpoint embeddings -- so the triangle inequality makes
    this cost >= the plain tau(a -> b) on every pair.  The gap is exactly the detour
    penalty of a move whose translation and rotation cannot be spent along one
    straight line in the learned metric, i.e. the signal the isotropic tau averages
    away.  No gradients are needed (the rollouts run under ``torch.no_grad``): all
    four legs go into ONE ``TravelTimes`` call, so this costs 4 forward pair
    evaluations instead of 1.
    """
    half = dim // 2
    lead = pairs.shape[:-1]
    flat = pairs.reshape(-1, 2 * dim)

    p_a, r_a = flat[:, :half], flat[:, half:dim]
    p_b, r_b = flat[:, dim:dim + half], flat[:, dim + half:]

    a = flat[:, :dim]
    b = flat[:, dim:]
    corner_tr = torch.cat([p_b, r_a], dim=1)     # translate first, then rotate
    corner_rt = torch.cat([p_a, r_b], dim=1)     # rotate first, then translate

    legs = torch.stack([
        torch.cat([a, corner_tr], dim=1),        # translation leg  (rot = r_a)
        torch.cat([corner_tr, b], dim=1),        # rotation leg     (pos = p_b)
        torch.cat([a, corner_rt], dim=1),        # rotation leg     (pos = p_a)
        torch.cat([corner_rt, b], dim=1),        # translation leg  (rot = r_b)
    ], dim=0)                                    # (4, N, 2*dim)

    legs = legs.reshape(-1, 2 * dim)
    if legs.shape[0] <= CTG_PAIR_CHUNK:
        t = womodel.function.TravelTimes(legs)
    else:
        t = torch.cat([womodel.function.TravelTimes(legs[i:i + CTG_PAIR_CHUNK])
                       for i in range(0, legs.shape[0], CTG_PAIR_CHUNK)], dim=0)
    t = t.reshape(4, -1)
    cost = 0.5 * (t[0] + t[1] + t[2] + t[3])     # mean of the two orderings
    return cost.reshape(lead)


def _candidate_cost(womodel, XP_tmp, cur, dim, local_step, all_horizon, local_w,
                    indep=False):
    """Score MPPI candidates, with the two optional cost repairs.

    ``XP_tmp`` is (B, S, H, 2*dim) -- every sample's rolled-out horizon, each row
    [candidate | target].  ``cur`` is (B, dim), the mover's current config.

    Baseline (both flags off) is the original objective, pure cost-to-go scored
    at horizon steps 0 and -1 only:

        cost = 10 * tau(cand_0 -> target) + tau(cand_last -> target)

    ``local_step`` (A) adds cost-so-far to each scored step, NORMALIZED by the
    length of that step: tau(cur -> cand) / ||cand - cur||.  Raw travel time
    mixes two things -- how long the step is and how slow the region is -- because
    the sampler only CAPS ||dP|| at the step radius rather than fixing it, and
    under all_horizon cand_k sits k steps out so the length term grows with k.
    Dividing it out leaves an estimate of local SLOWNESS (~1/speed) along the
    step, which is the part that tracks collision: on failing paths the learned
    speed is ~0.14 at the waypoints that collide against ~0.37 at the free ones.
    ``all_horizon`` (B) scores every horizon step instead of just the first and
    last, so a sample that sweeps through an obstacle mid-horizon is no longer
    invisible; the extra steps are averaged to keep the term on the original
    scale (the softmax temperature -50 is calibrated to it).

    ``indep`` (C) swaps the cost-to-go for ``cost_to_go_indep``, which times the
    translation and the rotation of the remaining move with their own speeds instead
    of charging both the same isotropic slowness.  It deliberately touches ONLY the
    cost-to-go: the local step term keeps its original form so that locB_indep vs
    locB (and hlB_indep vs hlB) isolates the cost-to-go change and nothing else.

    Shared by the one-way and the bidirectional planners so the two directions
    cannot drift apart.
    """
    B, S, H, _ = XP_tmp.shape
    idx = list(range(H)) if all_horizon else [0, -1]
    sel = XP_tmp[:, :, idx, :]
    K = sel.shape[2]

    if indep:
        f = cost_to_go_indep(womodel, sel, dim)                  # (B, S, K)
    else:
        f = womodel.function.TravelTimes(sel.reshape(-1, dim * 2)).reshape(B, S, K)

    if local_step:
        cand = sel[..., 0:dim]
        src = cur[:, None, None, :].expand_as(cand)
        local = womodel.function.TravelTimes(
            torch.cat([src, cand], dim=3).reshape(-1, dim * 2)).reshape(B, S, K)
        # Per-unit-length slowness, not raw travel time -- see the docstring.
        seg = torch.norm(cand - src, dim=3).clamp(min=1e-6)          # (B, S, K)
        f = f + local_w * (local / seg)

    if all_horizon:
        return 10 * f[:, :, 0] + f[:, :, 1:].mean(dim=2)
    return 10 * f[:, :, 0] + f[:, :, 1]


def MPPI_alternating_batched(womodel, XP, dim, steps=200, local_step=False,
                             all_horizon=False, local_w=None, indep=False,
                             momentum=None, step=None, taper=None, planar=None):
    """Alternating bidirectional MPPI: both ends march toward each other.

    ``MPPI_batched`` drives the start toward a goal that never moves.  Here the
    roles alternate every iteration:

        (s , e )  -- one MPPI step from s toward e  -->  (s1, e )
        swap      -->  (e , s1)
        (e , s1)  -- one MPPI step from e toward s1 -->  (e1, s1)
        swap      -->  (s1, e1)
        ...

    so each end advances one step toward the OTHER end's current position, and
    the two chains converge on a meeting point somewhere between them instead of
    one chain having to travel the whole way.  Convergence is the same rule as
    everywhere else in this file: the two live ends within 0.01 of each other.

    The sampler is byte-for-byte the one in ``MPPI_batched`` -- same 50 samples,
    same horizon 5, same STEP/TAPER/PLANAR handling, same 10*t0 + t1 cost, same
    softmax(-50 * cost) weighting -- so any difference in success rate is due to
    the alternation alone and not to a different integrator.  Converged episodes
    are frozen exactly as they are there.

    The delivered path is chain A (from the start) followed by chain B reversed,
    so it reads start -> ... -> meeting point -> ... -> goal and can be handed to
    ``check_trajectory_collision`` unchanged.  Like ``MPPI_batched`` the final
    config in the list is the original goal.

    ``indep`` is forwarded to ``_candidate_cost``: it replaces the cost-to-go with
    ``cost_to_go_indep`` (translation and rotation timed by their own speeds).

    Returns the same 4-tuple as ``MPPI_batched``.
    """
    B = XP.shape[0]
    sample_num = 50
    horizon = 5
    dev = XP.device
    local_w = LOCAL_W if local_w is None else local_w
    momentum = MOMENTUM if momentum is None else momentum
    momentum = 0
    step = STEP if step is None else step
    taper = TAPER if taper is None else taper
    planar = PLANAR if planar is None else planar
    if planar and dim != 6:
        raise ValueError('planar (--2d) mode assumes the SE(3) layout '
                         '(x,y,z,rx,ry,rz); got dim={}'.format(dim))
    free_mask = None
    if planar:
        free_mask = torch.zeros(dim, device=dev)
        free_mask[list(PLANAR_FREE_DIMS)] = 1.0

    XP = XP.clone()
    dP_prior = torch.zeros((B, dim), device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    conv_step = torch.full((B,), steps - 1, dtype=torch.long, device=dev)
    min_dis = torch.norm(XP[:, dim:dim * 2] - XP[:, 0:dim], dim=1)   # (B,)

    # chain_a grows from the original start, chain_b from the original goal.
    # n_a / n_b count how many entries of each are real for each episode (a
    # frozen episode keeps appending its unchanged config, which is dropped).
    chain_a = [XP[:, 0:dim].clone()]
    chain_b = [XP[:, dim:dim * 2].clone()]
    n_a = torch.ones(B, dtype=torch.long)
    n_b = torch.ones(B, dtype=torch.long)
    moving_a = True          # which chain currently occupies XP[:, 0:dim]

    for it in range(steps):
        XP_tmp = XP.clone()
        XP_tmp = XP_tmp[:, None, None, :].repeat(1, sample_num, horizon, 1)

        if taper > 0:
            dis_cur = torch.norm(XP[:, dim:dim * 2] - XP[:, 0:dim], dim=1)
            radius = torch.clamp(dis_cur * taper, max=step, min=step * 0.1)
        else:
            radius = torch.full((B,), step, device=dev)
        r = radius[:, None, None, None]

        dP = step * torch.normal(0, 1, size=(B, sample_num, 1, dim),
                                 dtype=torch.float32, device=dev)             + step * torch.normal(0, 1, size=(B, sample_num, horizon, dim),
                                  dtype=torch.float32, device=dev)
        if momentum:
            dP = dP + momentum * dP_prior[:, None, None, :]
        if free_mask is not None:
            dP = dP * free_mask
        dP_norm = torch.norm(dP, dim=3, keepdim=True)
        dP = dP / (torch.clamp(dP_norm, min=r) / r)
        dP_cumsum = torch.cumsum(dP, dim=2)
        XP_tmp[..., 0:dim] = XP_tmp[..., 0:dim] + dP_cumsum

        # The mover is always XP[:, 0:dim] -- after a swap that is the other end,
        # so the local step cost is measured from whichever end is stepping.
        cost = _candidate_cost(womodel, XP_tmp, XP[:, 0:dim], dim,
                               local_step, all_horizon, local_w, indep)

        weight = torch.softmax(-args.temp * cost, dim=1)
        step_prior = torch.bmm(weight.unsqueeze(1), dP[:, :, 0, :]).squeeze(1)
        dP_prior = step_prior

        live = (~done)
        XP[:, 0:dim] = XP[:, 0:dim] + step_prior * live.unsqueeze(1)

        dis = torch.norm(XP[:, dim:dim * 2] - XP[:, 0:dim], dim=1)
        min_dis = torch.minimum(min_dis, dis)

        live_cpu = live.detach().cpu().long()
        if moving_a:
            chain_a.append(XP[:, 0:dim].clone())
            n_a = n_a + live_cpu
        else:
            chain_b.append(XP[:, 0:dim].clone())
            n_b = n_b + live_cpu

        newly = (dis < 0.01) & (~done)
        conv_step[newly] = it
        done = done | (dis < 0.01)
        if bool(done.all()):
            break

        # Hand the step to the other end: swap the halves so the mover becomes
        # the target and vice versa.  The prior belongs to the end that was
        # moving, so it does not carry over.
        XP = torch.cat([XP[:, dim:dim * 2], XP[:, 0:dim]], dim=1)
        moving_a = not moving_a
        dP_prior = torch.zeros((B, dim), device=dev)

    success = done.detach().cpu().numpy()
    conv_step_cpu = conv_step.detach().cpu().numpy()

    points_list = []
    iters = []
    for b in range(B):
        a_len = int(n_a[b])
        b_len = int(n_b[b])
        cfgs = [chain_a[k][b:b + 1, :] for k in range(a_len)]
        cfgs += [chain_b[k][b:b + 1, :] for k in reversed(range(b_len))]
        points_list.append(cfgs)
        iters.append(int(conv_step_cpu[b]))

    return points_list, iters, success.tolist(), min_dis.detach().cpu().numpy()




def _placed_mesh(shape_V, cfg):
    """Transform the shape's local mesh by a config (x,y,z, rotvec in radians)."""
    R = _rotvec_to_matrix_np(cfg[3:6])
    return shape_V @ R.T + cfg[0:3]




def _progress_color(t):
    """Map a progress value t in [0, 1] to an RGB tuple of ints (viridis)."""
    try:
        import matplotlib.cm as cm
        r, g, b, _ = cm.get_cmap('viridis')(float(t))
    except Exception:
        # Fallback: simple blue -> yellow ramp if matplotlib is unavailable.
        r, g, b = float(t), float(t), 1.0 - float(t)
    return (int(r * 255), int(g * 255), int(b * 255))




def _to_waypoints(seq):
    """Turn a rollout (list of (1, DIM) tensors) into a (T, 6) pose array.

    The network works in the normalized frame (rotvec stored / 2*pi); the viewer
    wants the rotvec back in radians, so only the rotation block is rescaled.
    """
    if len(seq) == 0:
        return np.zeros((0, DIM))
    arr = torch.cat(list(seq), dim=0).detach().cpu().numpy()
    out = np.empty((arr.shape[0], DIM))
    out[:, 0:3] = arr[:, 0:3]
    out[:, 3:6] = arr[:, 3:6] * (2 * np.pi)
    return out




def add_environment(server, env_V, obst_F, wall_F):
    """Draw the environment as its actual triangle mesh (static scene).


    Obstacles are solid grey; walls are translucent light-blue.  This replaces
    the old sampled-point cloud rendering.
    """
    if len(obst_F) > 0:
        server.scene.add_mesh_simple(
            '/env/obstacles', vertices=env_V, faces=obst_F,
            color=(150, 150, 150), opacity=1.0, flat_shading=True, side='double')
    if len(wall_F) > 0:
        server.scene.add_mesh_simple(
            '/env/walls', vertices=env_V, faces=wall_F,
            color=(173, 216, 230), opacity=0.15, flat_shading=True, side='double')




def render_episode(server, ep, shape_V, shape_F, mode='regular'):
    """Add the moving-shape sweep + start/goal poses for one episode.


    The shape mesh is drawn at every waypoint, colored by PROGRESS along the path
    (viridis, dark=start .. bright=goal).  The start pose is red and the goal pose
    is green.  Returns the list of scene handles so the caller can remove them
    before rendering the next episode.


    ``mode`` selects which of the planners' paths to draw:
      'regular'  the single forward start->goal rollout.
      'flipped'  the backward goal->start rollout, reversed so it too reads
                 start->goal.  This is the path the 'rescued' cases are about.
      'both'     the two-path planner: whichever of the forward / backward
                 rollouts succeeded (forward preferred when both do).  The
                 backward path is reversed first so it still reads start->goal.


    Every path is stored start->goal, so the progress ramp is comparable across
    modes.  ``ep['waypoints_<mode>']`` is a (T, 6) array [x, y, z, rx, ry, rz]
    with the rotvec in radians.
    """
    handles = []
    waypoints = ep.get('waypoints_' + mode)
    if waypoints is None:
        waypoints = ep['waypoints_regular']
    T = len(waypoints)
    for t in range(T):
        Vp = _placed_mesh(shape_V, waypoints[t])
        handles.append(server.scene.add_mesh_simple(
            f'/episode/traj/{t:04d}', vertices=Vp, faces=shape_F,
            color=_progress_color(t / max(T - 1, 1)), opacity=0.5,
            flat_shading=True, side='double'))


    markers = [(ep['begin_cfg'], (220, 30, 30), 'start'),
               (ep['end_cfg'], (30, 180, 30), 'goal')]
    for cfg, col, nm in markers:
        if cfg is None:
            continue
        Vp = _placed_mesh(shape_V, cfg)
        handles.append(server.scene.add_mesh_simple(
            f'/episode/{nm}', vertices=Vp, faces=shape_F,
            color=col, opacity=0.9, flat_shading=True, side='double'))
    return handles




def launch_viser(episodes, shape_V, shape_F, env_V, obst_F, wall_F):
    """Serve an interactive viser scene with tabs + a dropdown to browse episodes."""
    import threading
    import viser
    server = viser.ViserServer(host='0.0.0.0', port=HTTP_PORT)
    server.scene.set_up_direction('+y')

    add_environment(server, env_V, obst_F, wall_F)

    MODES = ['regular', 'flipped', 'both']
    mode_sel = server.gui.add_dropdown('Mode', options=MODES)
    detail = server.gui.add_text('Outcome', initial_value='', disabled=True)

    # -- Case sets, one per tab --
    # The two directional tabs are the whole point of the flip experiment: a case
    # only carries directional information when exactly ONE of the two rollouts
    # solved it.  'Fwd fail / rev ok' is the rescued set (switch Mode to 'flipped'
    # to watch the path that actually worked); 'Fwd ok / rev fail' is its mirror.
    # Cases solved -- or failed -- both ways appear only under 'All'.
    TAB_SPECS = [
        ('All', lambda ep: True),
        ('Fwd fail / rev ok',
         lambda ep: ep['status_regular'] == 'fail' and ep['status_flipped'] == 'success'),
        ('Fwd ok / rev fail',
         lambda ep: ep['status_regular'] == 'success' and ep['status_flipped'] == 'fail'),
    ]
    NONE = '(none)'          # placeholder for an empty tab: viser needs an option

    current = []
    sel = {'i': 0}           # episode index, kept across mode / tab switches
    # Writing dd.value from code fires that dropdown's on_update on viser's thread
    # pool, indistinguishable from a real click.  The mode switch below rewrites
    # EVERY tab's dropdown, so unguarded it races one extra show() per tab against
    # the one the handler itself issues -- and whichever thread lands last decides
    # what is on screen.  That is how an episode from 'Fwd ok / rev fail' ends up
    # displayed while 'Fwd fail / rev ok' is the open tab.  Count each programmatic
    # write and let its own echo cancel it; a plain suspend flag would not survive
    # the (unbounded) thread-pool delay.
    echo = {}                # id(dropdown) -> programmatic writes not yet echoed
    echo_lock = threading.Lock()

    def show(i):
        for h in current:
            h.remove()
        current.clear()
        sel['i'] = i
        ep = episodes[i]
        mode = mode_sel.value
        current.extend(render_episode(server, ep, shape_V, shape_F, mode=mode))
        detail.value = 'regular {} | flipped {} | both {} ({})'.format(
            ep['status_regular'], ep['status_flipped'], ep['status_both'],
            ep['both_source'])

    # Episode labels carry the outcome of the SELECTED mode, so every dropdown
    # doubles as the list of that planner's failures within that tab's case set.
    def labels_for(subset, mode):
        return [f"{episodes[i]['idx']:03d}_{episodes[i]['status_' + mode]}"
                for i in subset] or [NONE]

    # viser grew add_tab_group early on, but fall back to plain stacked dropdowns
    # (one per case set) if this install predates it.
    has_tabs = hasattr(server.gui, 'add_tab_group')
    tab_group = server.gui.add_tab_group() if has_tabs else None
    tabs = []                # (subset, dropdown) per tab

    for title, pred in TAB_SPECS:
        subset = [k for k, ep in enumerate(episodes) if pred(ep)]
        label = f'{title} ({len(subset)})'
        if has_tabs:
            with tab_group.add_tab(label):
                dd = server.gui.add_dropdown('Episode',
                                             options=labels_for(subset, MODES[0]))
        else:
            dd = server.gui.add_dropdown(f'Episode [{label}]',
                                         options=labels_for(subset, MODES[0]))
        tabs.append((subset, dd))

        @dd.on_update
        def _(_, subset=subset, dd=dd):
            with echo_lock:
                if echo.get(id(dd), 0) > 0:
                    echo[id(dd)] -= 1       # our own relabel, not a user click
                    return
            if not subset or dd.value == NONE:
                return
            show(subset[list(dd.options).index(dd.value)])

    @mode_sel.on_update
    def _(_):
        # Relabel every tab's episodes for the new mode, holding each selection.
        for subset, dd in tabs:
            if not subset:
                continue
            k = list(dd.options).index(dd.value) if dd.value in dd.options else 0
            new_options = labels_for(subset, mode_sel.value)
            dd.options = new_options            # options setter fires no callback
            with echo_lock:                     # ...but the value setter does
                echo[id(dd)] = echo.get(id(dd), 0) + 1
            dd.value = new_options[k]
        show(sel['i'])

    if episodes:
        show(0)

    print(f"\nServing viser at http://0.0.0.0:{HTTP_PORT}  \u2014  open this on your host PC")
    print("Use 'Mode' to switch planner (regular / flipped / both), the tabs "
          "to pick a case set, and 'Episode' to browse.")
    print("Picking a tab does not move the scene on its own (viser exposes no "
          "tab-change callback) -- choose an episode from that tab's 'Episode' "
          "dropdown to display it.")
    print("Tabs: 'All' | 'Fwd fail / rev ok' (rescued by the flip) | "
          "'Fwd ok / rev fail' (the mirror).  View those two in Mode 'flipped' to "
          "see the reversed rollout itself.")
    print("Path is colored by progress (dark=start .. bright=goal); start pose red, "
          "goal green.")
    print("Press Ctrl-C to stop.\n")
    while True:
        time.sleep(10)


# ──────────────────────────────────────────────────────────────────────────────
# Model & data setup
# ──────────────────────────────────────────────────────────────────────────────
modelPath = args.modelPath
dataPath = args.dataPath


womodel = md.Model(modelPath, dataPath, DIM, [0.0] * DIM, device=DEVICE)


# Prefer the always-current latest.pt written every epoch by training; fall back
# to the most recent timestamped Model_Epoch_*.pt checkpoint.
if args.checkpoint is not None:
    pt = args.checkpoint
    if not os.path.exists(pt):
        raise FileNotFoundError(pt)
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

vprint(f'Loading checkpoint: {pt}')

#pt = './Experiments/3dshape/3dshape_08_15_20_19/latest.pt'
womodel.load(pt)
womodel.network.eval()


arr = np.load(os.path.join(dataPath, 'sampled_points.npy'))       # (N, 12)
arr_speeds = np.load(os.path.join(dataPath, 'speed.npy'))         # (N, 2)


# Per-endpoint clearance proxies, used only for the end-of-run frequency tables.
# Column 0 is the START (x0), column 1 the GOAL (x1).  Optional: older datasets
# may predate them, in which case the tables are skipped.
def _try_load(name):
    p = os.path.join(dataPath, name)
    return np.load(p) if os.path.exists(p) else None


arr_speed_dists = _try_load('speed_dists.npy')                    # (N, 2) or None
arr_speed_angles = _try_load('speed_angles.npy')                  # (N, 2) or None


# Reconstruct the shape + wall meshes in the normalized frame from meta.json
# (written by preprocess_obj.py) so the visualization matches the sampling frame.
with open(os.path.join(dataPath, 'meta.json')) as f:
    meta = json.load(f)
# The test set records how it was sampled; --2d must agree with it or the
# rollout plans in a different sub-space than the one the field was fit on.
meta_two_d = bool(meta.get('two_d', False))
if meta_two_d and not PLANAR:
    print('WARNING: meta.json says this test set was generated with --2d '
          '(planar, z=0, z-rotation only) but --2d was NOT passed; MPPI will '
          'sample all 6 DOF and leave the plane.')
elif PLANAR and not meta_two_d:
    print('WARNING: --2d was passed but meta.json says this test set is full '
          '6-DOF; z / rx / ry will be frozen at their start values, so goals '
          'that differ in those coordinates can never converge.')

env_scale = float(meta['env_scale'])
env_center = np.asarray(meta['env_center'], dtype=np.float64)
shape_scale = float(meta['shape_scale'])


def _resolve_mesh(path):
    """meta.json stores absolute paths from the container that generated the
    data (e.g. /workspace/ntrl-demo/datasets/...).  A container that mounts the
    repo at a different depth cannot resolve those, so fall back to the local
    datasets dir by basename."""
    if os.path.exists(path):
        return path
    alt = os.path.join('./datasets/3dshape', os.path.basename(path))
    if os.path.exists(alt):
        vprint(f'[meta] {path} not found; using {alt}')
        return alt
    raise FileNotFoundError(f'{path} (and {alt})')


with _quiet():
    V_sh, F_sh, _ = load_obj(_resolve_mesh(meta['shape_obj']))
shape_center = 0.5 * (V_sh.min(axis=0) + V_sh.max(axis=0))
shape_V = np.ascontiguousarray((V_sh - shape_center) / env_scale * shape_scale,
                               dtype=np.float64)
shape_F = np.ascontiguousarray(F_sh, dtype=np.int64)
# Bounding radius of the shape about its local origin (used for collision broad phase).
shape_radius = float(np.linalg.norm(shape_V, axis=1).max())


with _quiet():
    V_env, F_env, names_env = load_obj(_resolve_mesh(meta['env_obj']))
V_env_n = (V_env - env_center) / env_scale
wall_mask = np.array(['wall' in str(n).lower() for n in names_env])
wall_F = F_env[wall_mask]
obst_F = F_env[~wall_mask]


# Collision point cloud: env.npy is obstacles-only, so resample densely over the
# FULL environment mesh (obstacles + walls) -- everything counts as an obstacle.
env_collision_pts = np.ascontiguousarray(
    sample_surface_points(V_env_n, F_env, ENV_COLLISION_POINTS), dtype=np.float64)


# How many of the file's pairs to evaluate.  This used to be a hardcoded count
# that did not track the size of the test set, so a 1000-pair test set was
# silently scored on a fraction of itself; --cases 0 now means "all of them".
n_requested = len(arr) if args.cases <= 0 else min(args.cases, len(arr))
if args.cases > len(arr):
    print(f'WARNING: --cases {args.cases} exceeds the {len(arr)} pairs in '
          f'{os.path.join(dataPath, "sampled_points.npy")}; using all {len(arr)}.')
vprint(f'[cases] evaluating {n_requested} of {len(arr)} start/goal pairs'
      f'{" (ALL)" if n_requested == len(arr) else ""}')

test_list = []
test_list_speed = []
for i in range(n_requested):
    curr = torch.tensor(arr[i]).to(DEVICE)
    test_list.append(curr)
    test_list_speed.append(min(arr_speeds[i]))

if PLANAR:
    # MPPI never writes the frozen coordinates, and convergence is measured by a
    # full 6-D norm, so any start/goal disagreement on z / rx / ry is an
    # unreachable offset that silently caps the success rate at 0.
    frozen = [d for d in range(DIM) if d not in PLANAR_FREE_DIMS]
    _pts = np.stack([c.detach().cpu().numpy() for c in test_list], axis=0)
    _off = np.abs(_pts[:, [DIM + d for d in frozen]] - _pts[:, frozen]).max()
    vprint(f'[--2d] free dims {PLANAR_FREE_DIMS} (x, y, rz); '
          f'frozen dims {tuple(frozen)} (z, rx, ry) held at their start values.  '
          f'Max |goal - start| over the frozen dims across {len(test_list)} '
          f'cases: {_off:.3e}')
    if _off > 1e-6:
        print('WARNING: the test set moves in the frozen coordinates -- those '
              'cases cannot converge under --2d.')


# ──────────────────────────────────────────────────────────────────────────────
# Endpoint validity filter
# ──────────────────────────────────────────────────────────────────────────────
# A test case is only meaningful if the shape is collision-free at BOTH the start
# and the goal pose -- if either endpoint is already inside the environment the
# case is unsolvable by construction and would otherwise be charged to the
# planner as a failure.  Those cases are dropped here (before the rollouts, so no
# GPU time is spent on them) and excluded from every statistic below, making the
# reported success rate the actual rate over solvable cases.
n_cases = len(test_list)
eval_list = []
eval_idx = []          # original index in test_list, for logging / speed lookup
n_invalid_endpoints = 0
n_invalid_start = 0
n_invalid_goal = 0

filter_start = timer()
for i, curr in enumerate(test_list):
    XP = curr.reshape(1, 2 * DIM)
    start_bad = check_config_collision(
        XP[0, 0:DIM], env_collision_pts, shape_V, shape_F, shape_radius)
    goal_bad = check_config_collision(
        XP[0, DIM:2 * DIM], env_collision_pts, shape_V, shape_F, shape_radius)
    if start_bad or goal_bad:
        n_invalid_endpoints += 1
        n_invalid_start += int(start_bad)
        n_invalid_goal += int(goal_bad)
        vprint(f"[{i:03d}] SKIP  start_collision={start_bad}  goal_collision={goal_bad}")
        continue
    eval_list.append(curr)
    eval_idx.append(i)

vprint(f"\nEndpoint check took {timer() - filter_start:.1f}s: "
      f"{n_invalid_endpoints} / {n_cases} cases discarded "
      f"(start in collision: {n_invalid_start}, goal in collision: {n_invalid_goal}); "
      f"{len(eval_list)} remain.\n")


# ──────────────────────────────────────────────────────────────────────────────
# Batched evaluation loop (MPPI rollouts run EPISODE_BATCH at a time on the GPU)
# ──────────────────────────────────────────────────────────────────────────────
total = 0
total_flip = 0         # successes of the flipped (goal -> start) attempt alone
total_either = 0       # successes of EITHER attempt -- the 2-path planner
n_rescued = 0          # regular failed but flipped succeeded
n_flip_only = 0        # regular succeeded but flipped failed (the mirror image)
n_collision = 0
n_no_conv = 0
n_collision_flip = 0
n_no_conv_flip = 0
total_lb = 0           # local step, bidirectional (A + alternating)
n_collision_lb = 0
n_no_conv_lb = 0
n_lb_rescued = 0
n_lb_only = 0
total_hb = 0           # horizon + local, bidirectional (A + B + alternating)
n_collision_hb = 0
n_no_conv_hb = 0
n_hb_rescued = 0
n_hb_only = 0
# # Same two planners again, but scoring candidates with cost_to_go_indep (the
# # translation / rotation split cost-to-go) instead of the plain isotropic tau.
# total_lbi = 0          # locB_indep
# n_collision_lbi = 0
# n_no_conv_lbi = 0
# n_lbi_rescued = 0      # vs regular
# n_lbi_only = 0
# n_lbi_over_lb = 0      # locB failed, locB_indep passed  -- the split's own gain
# n_lb_over_lbi = 0      # locB passed, locB_indep failed
# total_hbi = 0          # hlB_indep
# n_collision_hbi = 0
# n_no_conv_hbi = 0
# n_hbi_rescued = 0
# n_hbi_only = 0
# n_hbi_over_hb = 0
# n_hb_over_hbi = 0
total_alt = 0          # successes of the alternating bidirectional planner
n_collision_alt = 0
n_no_conv_alt = 0
n_alt_rescued = 0      # regular failed but alternating succeeded
n_alt_only = 0         # regular succeeded but alternating failed
alt_rescued_idx = []
no_conv_min_dis = []   # closest approach to the goal, for NON-CONVERGED regular runs
n_lin_valid = 0       # cases where the straight-line start->goal path is collision-free
n_succ_lin_valid = 0  # planner successes among cases where linear interp is valid
n_succ_lin_invalid = 0  # planner successes among cases where linear interp is invalid
success_list = []
fail_list = []
failed_idx = []        # original test-case index of every FAILED episode
failed_both_idx = []   # index of every case where BOTH regular AND flipped failed
rescued_idx = []       # index of every case the flip rescued (regular fail, flip pass)
# The mirror set: the forward rollout solved it and the reversed one did not.
# Read together with rescued_idx these two sets are the whole directional signal
# -- everything else is solved (or failed) both ways and says nothing about
# direction.  If the two sets are the same size and sit in the same place in
# clearance space, the flip is just re-rolling MPPI's dice.
flip_only_idx = []     # index of every case where regular passed but the flip failed
test_list_speed_failed = []
episodes = []          # per-episode data for the interactive viser viewer


two_pi = 2 * np.pi
n_total = len(eval_list)          # solvable cases only (invalid endpoints dropped)
eval_start = timer()
n_done = 0                        # cases scored so far, for the progress report


# Cadence of the running progress report.  That report is deliberately NOT gated
# by --verbose: on a full 1000-case run the rollouts take long enough that it is
# the only sign of life, so it prints either way.
PROGRESS_EVERY = 50


def planner_rows(denom):
    """(reported name, successes, rate) for every planner at the current tallies.

    ``denom`` is what the rates are taken over -- the number of cases scored so
    far for a progress report, ``n_total`` for the final one.  The names are the
    code's label and the reporting name, so both reports read identically.
    """
    rows = [
        ('regular/forward', total),
        ('flipped/reverse', total_flip),
        ('either/or', total_either),
        ('alt/alternate', total_alt),
        ('locB/Alternative Bellman', total_lb),
        ('hlB/Alternative Bellman Horizon', total_hb),
    ]
    return [(name, n_ok, n_ok / denom if denom else 0.0) for name, n_ok in rows]


REPORT_W = max(len(name) for name, _, _ in planner_rows(1))

for chunk_start in range(0, n_total, EPISODE_BATCH):
    chunk = eval_list[chunk_start:chunk_start + EPISODE_BATCH]
    XPb = torch.stack([c.reshape(2 * DIM) for c in chunk], dim=0)   # (B, 2*DIM)
    # Second attempt at the same case, planned BACKWARDS (goal -> start).  Rows are
    # [start | goal], so flipping is a half-swap.  tau is symmetric by construction
    # but the MPPI rollout is not: it is stochastic and descends the field from a
    # different end, so a case whose forward rollout gets trapped can still be
    # solvable in reverse.  Two paths are generated per case and the case counts as
    # solved if EITHER succeeds.
    XPb_flip = torch.cat([XPb[:, DIM:2 * DIM], XPb[:, 0:DIM]], dim=1)


    # Batched MPPI rollout for the whole chunk (network-heavy part, on GPU).
    with torch.no_grad():
        points_list, iters, successes, min_dis = MPPI_batched(
            womodel, XPb.clone(), dim=DIM)
        points_list_f, iters_f, successes_f, _ = MPPI_batched(
            womodel, XPb_flip.clone(), dim=DIM)
        # Third planner: the two ends alternate steps toward each other.
        points_list_a, iters_a, successes_a, _ = MPPI_alternating_batched(
            womodel, XPb.clone(), dim=DIM)
        # Fourth / fifth: the two cost repairs, with the ends alternating steps
        # toward each other (A, and A+B).
        points_list_lb, iters_lb, successes_lb, _ = MPPI_alternating_batched(
            womodel, XPb.clone(), dim=DIM, local_step=True, all_horizon=False)
        points_list_hb, iters_hb, successes_hb, _ = MPPI_alternating_batched(
            womodel, XPb.clone(), dim=DIM, local_step=True, all_horizon=True)
        # locB_indep / hlB_indep are DISABLED.  ``cost_to_go_indep`` and the
        # ``indep`` flag on _candidate_cost / MPPI_alternating_batched are still
        # live, so re-enabling is just uncommenting this block and the four
        # blocks below it (scoring, tallies, viewer payload, report).
        # # Sixth / seventh: the same two, with the cost-to-go split into a
        # # translation leg (timed by s_dist) and a rotation leg (timed by s_ang).
        # # Everything else -- sampler, local step term, alternation -- is identical
        # # to locB / hlB, so the delta is the split cost-to-go alone.
        # points_list_lbi, iters_lbi, successes_lbi, _ = MPPI_alternating_batched(
            # womodel, XPb.clone(), dim=DIM, local_step=True, all_horizon=False,
            # indep=True)
        # points_list_hbi, iters_hbi, successes_hbi, _ = MPPI_alternating_batched(
            # womodel, XPb.clone(), dim=DIM, local_step=True, all_horizon=True,
            # indep=True)


    for b, (point, iter, success) in enumerate(zip(points_list, iters, successes)):
        cnt2 = eval_idx[chunk_start + b]     # original test-case index
        XP = chunk[b].reshape(1, 2 * DIM)
        # displacement goal - start over all 6 DOF (position part used for scatter)
        diff = (XP[0, DIM:2 * DIM] - XP[0, 0:DIM]).detach().cpu().numpy().tolist()


        # (T, 6) waypoints; rotvec de-normalized to radians (stored / 2*pi).
        waypoints = _to_waypoints(point)


        begin_cfg = waypoints[0].copy()
        end_cfg = np.array([XP[0][DIM + 0].item(), XP[0][DIM + 1].item(), XP[0][DIM + 2].item(),
                            XP[0][DIM + 3].item() * two_pi, XP[0][DIM + 4].item() * two_pi,
                            XP[0][DIM + 5].item() * two_pi])


        collision = check_trajectory_collision(
            point, env_collision_pts, shape_V, shape_F, shape_radius)
        did_not_converge = not success


        # A case can independently collide AND fail to converge -- count both.
        if collision:
            n_collision += 1
        if did_not_converge:
            n_no_conv += 1
            # How close did it ever get?  Just above the 0.01 ball means the
            # rollout found the goal but overshot it; far means it never arrived.
            no_conv_min_dis.append(float(min_dis[b]))


        # ── Flipped attempt (goal -> start), scored by the same rules ──
        collision_f = check_trajectory_collision(
            points_list_f[b], env_collision_pts, shape_V, shape_F, shape_radius)
        did_not_converge_f = not successes_f[b]
        if collision_f:
            n_collision_flip += 1
        if did_not_converge_f:
            n_no_conv_flip += 1
        ok_flip = successes_f[b] and not collision_f

        # -- Alternating bidirectional attempt, scored by the same rules --
        collision_a = check_trajectory_collision(
            points_list_a[b], env_collision_pts, shape_V, shape_F, shape_radius)
        did_not_converge_a = not successes_a[b]
        if collision_a:
            n_collision_alt += 1
        if did_not_converge_a:
            n_no_conv_alt += 1
        ok_alt = successes_a[b] and not collision_a

        # -- bidirectional variants of the same two cost repairs --
        collision_lb = check_trajectory_collision(
            points_list_lb[b], env_collision_pts, shape_V, shape_F, shape_radius)
        if collision_lb:
            n_collision_lb += 1
        if not successes_lb[b]:
            n_no_conv_lb += 1
        ok_lb = successes_lb[b] and not collision_lb

        collision_hb = check_trajectory_collision(
            points_list_hb[b], env_collision_pts, shape_V, shape_F, shape_radius)
        if collision_hb:
            n_collision_hb += 1
        if not successes_hb[b]:
            n_no_conv_hb += 1
        ok_hb = successes_hb[b] and not collision_hb

        # # -- split (indep) cost-to-go variants of the same two planners --
        # collision_lbi = check_trajectory_collision(
            # points_list_lbi[b], env_collision_pts, shape_V, shape_F, shape_radius)
        # if collision_lbi:
            # n_collision_lbi += 1
        # if not successes_lbi[b]:
            # n_no_conv_lbi += 1
        # ok_lbi = successes_lbi[b] and not collision_lbi

        # collision_hbi = check_trajectory_collision(
            # points_list_hbi[b], env_collision_pts, shape_V, shape_F, shape_radius)
        # if collision_hbi:
            # n_collision_hbi += 1
        # if not successes_hbi[b]:
            # n_no_conv_hbi += 1
        # ok_hbi = successes_hbi[b] and not collision_hbi


        # Linear-interpolation baseline: is the straight line from start to goal a
        # collision-free path on its own?  (Independent of what the planner found.)
        lin_traj = linear_interp_traj(XP[0, 0:DIM], XP[0, DIM:2 * DIM])
        lin_collision = check_trajectory_collision(
            lin_traj, env_collision_pts, shape_V, shape_F, shape_radius)
        lin_valid = not lin_collision
        if lin_valid:
            n_lin_valid += 1


        ok = success and not collision
        ok_either = ok or ok_flip
        if ok_alt:
            total_alt += 1
        if ok_alt and not ok:
            n_alt_rescued += 1
            alt_rescued_idx.append(cnt2)
        if ok and not ok_alt:
            n_alt_only += 1
        if ok_lb:
            total_lb += 1
        if ok_lb and not ok:
            n_lb_rescued += 1
        if ok and not ok_lb:
            n_lb_only += 1
        if ok_hb:
            total_hb += 1
        if ok_hb and not ok:
            n_hb_rescued += 1
        if ok and not ok_hb:
            n_hb_only += 1
        # if ok_lbi:
            # total_lbi += 1
        # if ok_lbi and not ok:
            # n_lbi_rescued += 1
        # if ok and not ok_lbi:
            # n_lbi_only += 1
        # # Head-to-head against the same planner on the plain cost-to-go: this is
        # # the only pairing that isolates what the split cost bought.
        # if ok_lbi and not ok_lb:
            # n_lbi_over_lb += 1
        # if ok_lb and not ok_lbi:
            # n_lb_over_lbi += 1
        # if ok_hbi:
            # total_hbi += 1
        # if ok_hbi and not ok:
            # n_hbi_rescued += 1
        # if ok and not ok_hbi:
            # n_hbi_only += 1
        # if ok_hbi and not ok_hb:
            # n_hbi_over_hb += 1
        # if ok_hb and not ok_hbi:
            # n_hb_over_hbi += 1
        if ok_flip:
            total_flip += 1
        if ok_either:
            total_either += 1
        if ok_either and not ok:
            n_rescued += 1
            rescued_idx.append(cnt2)
        if ok and not ok_flip:
            # Mirror of a rescue: the forward direction was the one that worked.
            n_flip_only += 1
            flip_only_idx.append(cnt2)
        if not ok_either:
            # Hard failure: the flip did not rescue it either.
            failed_both_idx.append(cnt2)
        status = 'success' if ok else 'fail'


        # ── Viewer payload: one path per planner, all stored start -> goal ──
        # 'both' is the two-path planner's DELIVERED path: the forward rollout
        # when it worked, otherwise the backward one played in reverse so it
        # still reads start -> goal.  When neither worked there is nothing to
        # deliver, so it falls back to the forward path to show the failure.
        # The backward rollout played in reverse, so it too reads start -> goal.
        flip_wp = _to_waypoints(list(reversed(points_list_f[b])))
        if ok:
            both_wp, both_src = waypoints, 'regular'
        elif ok_flip:
            both_wp, both_src = flip_wp, 'flipped'
        else:
            both_wp, both_src = waypoints, 'neither'
        episodes.append({
            'idx': cnt2,
            'status_regular': status,
            'status_flipped': 'success' if ok_flip else 'fail',
            'status_both': 'success' if ok_either else 'fail',
            'status_alternating': 'success' if ok_alt else 'fail',
            'status_local_bidir': 'success' if ok_lb else 'fail',
            'status_horizon_local_bidir': 'success' if ok_hb else 'fail',
            # 'status_local_bidir_indep': 'success' if ok_lbi else 'fail',
            # 'status_horizon_local_bidir_indep': 'success' if ok_hbi else 'fail',
            'waypoints_local_bidir': _to_waypoints(points_list_lb[b]),
            'waypoints_horizon_local_bidir': _to_waypoints(points_list_hb[b]),
            # 'waypoints_local_bidir_indep': _to_waypoints(points_list_lbi[b]),
            # 'waypoints_horizon_local_bidir_indep': _to_waypoints(points_list_hbi[b]),
            'waypoints_alternating': _to_waypoints(points_list_a[b]),
            'both_source': both_src,
            'waypoints_regular': waypoints,
            'waypoints_flipped': flip_wp,
            'waypoints_both': both_wp,
            'begin_cfg': begin_cfg,
            'end_cfg': end_cfg,
        })


        vprint(
            f"[{cnt2:03d}] {'PASS' if ok else 'FAIL'}  "
            f"flipped={'PASS' if ok_flip else 'FAIL'}  "
            f"either={'PASS' if ok_either else 'FAIL'}  "
            f"alt={'PASS' if ok_alt else 'FAIL'}  "
            f"locB={'PASS' if ok_lb else 'FAIL'}  "
            f"hlB={'PASS' if ok_hb else 'FAIL'}  "
            # f"locB_i={'PASS' if ok_lbi else 'FAIL'}  "
            # f"hlB_i={'PASS' if ok_hbi else 'FAIL'}  "
            f"did_not_converge={did_not_converge}  "
            f"collision={collision}  "
            f"linear_interp_valid={lin_valid}")


        if ok:
            success_list.append(diff[:3])
            total += 1
            if lin_valid:
                n_succ_lin_valid += 1
            else:
                n_succ_lin_invalid += 1
        else:
            fail_list.append(diff[:3])
            failed_idx.append(cnt2)
            test_list_speed_failed.append(test_list_speed[cnt2])


        # -- Running progress report, every PROGRESS_EVERY cases (always printed) --
        # Rates are over the cases scored SO FAR, so they are the running estimate
        # of the final numbers, not a fraction of n_total.
        n_done += 1
        if n_done % PROGRESS_EVERY == 0:
            elapsed = timer() - eval_start
            eta = elapsed / n_done * (n_total - n_done)
            print(f"\n[progress] {n_done}/{n_total} cases  "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
            for name, n_ok, rate in planner_rows(n_done):
                print(f"  {name:<{REPORT_W}} : {rate:.4f}  ({rate:.1%})  "
                      f"[{n_ok}/{n_done}]")


vprint(f"\nEvaluation (rollouts) took {timer() - eval_start:.1f}s "
      f"for {n_total} episodes in chunks of {EPISODE_BATCH}.")


success_rate = total / n_total if n_total else 0.0
# Conditional planner success rates: how often the planner succeeds among cases
# where the trivial straight-line (linear interp) path is valid vs. invalid.
n_lin_invalid = n_total - n_lin_valid
lin_valid_success_rate = n_succ_lin_valid / n_lin_valid if n_lin_valid else 0.0
lin_invalid_success_rate = n_succ_lin_invalid / n_lin_invalid if n_lin_invalid else 0.0
flip_rate = total_flip / n_total if n_total else 0.0
either_rate = total_either / n_total if n_total else 0.0
alt_rate = total_alt / n_total if n_total else 0.0
lb_rate = total_lb / n_total if n_total else 0.0
hb_rate = total_hb / n_total if n_total else 0.0
# lbi_rate = total_lbi / n_total if n_total else 0.0
# hbi_rate = total_hbi / n_total if n_total else 0.0


def _phi(p_a, p_b, p_both):
    """Phi (mean-square-contingency) correlation of two binary FAILURE events.

    0 => the two attempts fail independently, so pairing them buys the full
    1 - p_a*p_b; 1 => they fail on exactly the same cases and the second attempt
    is worthless.  Undefined (returned as 0) if either attempt never fails.
    """
    denom = math.sqrt(p_a * (1 - p_a) * p_b * (1 - p_b))
    return (p_both - p_a * p_b) / denom if denom > 0 else 0.0


# Failure-side view of each 2-path pairing: marginals, the observed joint, and
# the joint an independence assumption would predict.
f_reg = 1 - success_rate
f_flip = 1 - flip_rate
both_flip_rate = 1 - either_rate
indep_flip = f_reg * f_flip
phi_flip = _phi(f_reg, f_flip, both_flip_rate)
# ── Why did the non-converged runs fail? ──
# Bin the closest approach against the 0.01 convergence ball.  A rollout that got
# to within a couple of ball radii and still never landed is an overshoot: the
# stride (STEP, default 0.015) exceeds the ball, and momentum resists the
# reversal needed to correct it.  One that never got close is a descent failure
# and has nothing to do with the integrator.
_md = np.asarray(no_conv_min_dis, dtype=float)
_md_bins = [(0.01, 0.02), (0.02, 0.04), (0.04, 0.10), (0.10, np.inf)]
vprint(f"\n--- sampler: momentum={MOMENTUM}  step={STEP}  taper={TAPER}"
      f"  (convergence ball 0.01) ---")
vprint(f"discarded (start/goal in collision): {n_invalid_endpoints} / {n_cases}")
vprint(f"total: {total} / {n_total}  ({success_rate:.1%})")
vprint(f"\n--- closest approach of the {len(_md)} NON-CONVERGED regular runs ---")
if _md.size:
    vprint(f"  median {np.median(_md):.4f}   mean {np.mean(_md):.4f}   "
          f"min {np.min(_md):.4f}")
    for lo, hi in _md_bins:
        k = int(((_md >= lo) & (_md < hi)).sum())
        vprint(f"  min_dis in [{lo:.2f}, {hi if hi != np.inf else 999:>5.2f}): "
              f"{k:4d}  ({k / _md.size:.1%})"
              + ('   <- overshoot: reached the goal, could not stick it'
                 if hi <= 0.04 else ''))
else:
    vprint("  (every regular run converged)")
vprint(f"\n--- two-path planner (regular + flipped) ---")
vprint(f"  step 1  regular (start->goal): {total} / {n_total}  ({success_rate:.1%})")
vprint(f"          flipped (goal->start): {total_flip} / {n_total}  ({flip_rate:.1%})")
vprint(f"  step 2  EITHER succeeded     : {total_either} / {n_total}  ({either_rate:.1%})")
vprint(f"          rescued by the flip  : {n_rescued}  "
      f"(+{either_rate - success_rate:.1%} over regular alone)")
vprint(f"          lost by the flip     : {n_flip_only}  "
      f"[regular passed, flipped failed -- the mirror of a rescue]")
vprint(f"          failed BOTH ways     : {len(failed_both_idx)} / {n_total}  "
      f"({(1 - either_rate):.1%})")
vprint(f"\n--- do the two directions fail independently? ---")
vprint(f"  P(both fail)  flip           : {both_flip_rate:.4f}   "
      f"(independent would be {indep_flip:.4f})   phi={phi_flip:+.3f}")
vprint(f"  [phi ~ 0 => the two attempts fail independently, so the gain from")
vprint(f"   pairing them is variance reduction, not a directional effect.]")
vprint(f"\n--- alternating bidirectional planner (ends step toward each other) ---")
vprint(f"  alternating                  : {total_alt} / {n_total}  ({alt_rate:.1%})")
vprint(f"  regular (start->goal)        : {total} / {n_total}  ({success_rate:.1%})")
vprint(f"  delta vs regular             : {alt_rate - success_rate:+.1%}")
vprint(f"  rescued by alternating       : {n_alt_rescued}  "
      f"[regular failed, alternating passed]")
vprint(f"  lost by alternating          : {n_alt_only}  "
      f"[regular passed, alternating failed]")
vprint(f"  collision                    : {n_collision_alt} / {n_total}")
vprint(f"  did not converge             : {n_no_conv_alt} / {n_total}")
vprint(f"\n--- cost repairs, BIDIRECTIONAL (ends alternate steps) ---")
vprint(f"  local step weight            : {LOCAL_W}")
vprint(f"  regular (start->goal)        : {total} / {n_total}  ({success_rate:.1%})")
vprint(f"  alternating (plain cost)     : {total_alt} / {n_total}  ({alt_rate:.1%})")
vprint(f"  local step        (A)   bidir: {total_lb} / {n_total}  ({lb_rate:.1%})"
      f"   [{lb_rate - alt_rate:+.1%} vs plain alternating,"
      f" {lb_rate - success_rate:+.1%} vs regular]")
vprint(f"  horizon + local   (A+B) bidir: {total_hb} / {n_total}  ({hb_rate:.1%})"
      f"   [{hb_rate - alt_rate:+.1%} vs plain alternating,"
      f" {hb_rate - success_rate:+.1%} vs regular]")
vprint(f"  local bidir  : rescued {n_lb_rescued}  lost {n_lb_only}  "
      f"collision {n_collision_lb}  no_conv {n_no_conv_lb}   [vs regular]")
vprint(f"  hor+loc bidir: rescued {n_hb_rescued}  lost {n_hb_only}  "
      f"collision {n_collision_hb}  no_conv {n_no_conv_hb}   [vs regular]")
# print(f"\n--- split (indep) cost-to-go: translation and rotation timed separately ---")
# print(f"  [cost_to_go_indep replaces tau(cand->target) by tau over a pure-translation")
# print(f"   leg + a pure-rotation leg, so each block is timed by its own speed; the")
# print(f"   sampler and the local step term are untouched.]")
# print(f"  local step        (A)   bidir: {total_lb} / {n_total}  ({lb_rate:.1%})"
      # f"   -> indep: {total_lbi} / {n_total}  ({lbi_rate:.1%})"
      # f"   [{lbi_rate - lb_rate:+.1%}]")
# print(f"  horizon + local   (A+B) bidir: {total_hb} / {n_total}  ({hb_rate:.1%})"
      # f"   -> indep: {total_hbi} / {n_total}  ({hbi_rate:.1%})"
      # f"   [{hbi_rate - hb_rate:+.1%}]")
# print(f"  locB_indep   : won {n_lbi_over_lb}  lost {n_lb_over_lbi}   [head-to-head vs locB]"
      # f"   collision {n_collision_lbi}  no_conv {n_no_conv_lbi}")
# print(f"  hlB_indep    : won {n_hbi_over_hb}  lost {n_hb_over_hbi}   [head-to-head vs hlB]"
      # f"   collision {n_collision_hbi}  no_conv {n_no_conv_hbi}")
# print(f"  [won ~ lost with the rate unchanged => the split only re-rolled MPPI's")
# print(f"   dice; a real gain shows up as won >> lost.]")
vprint(f"success | linear interp valid  : {n_succ_lin_valid} / {n_lin_valid}  "
      f"({lin_valid_success_rate:.1%})")
vprint(f"success | linear interp invalid: {n_succ_lin_invalid} / {n_lin_invalid}  "
      f"({lin_invalid_success_rate:.1%})")


# ──────────────────────────────────────────────────────────────────────────────
# Success-rate report (always printed, --verbose or not)
# ──────────────────────────────────────────────────────────────────────────────
# One line per planner: the code's label, a slash, and the reporting name.  This
# is the whole of the non-verbose output -- everything else the script prints is
# diagnostic and lives behind --verbose.
PLANNER_REPORT = planner_rows(n_total)
_w = REPORT_W
print(f"\n=== success rate over {n_total} episodes ===")
for name, n_ok, rate in PLANNER_REPORT:
    print(f"{name:<{_w}} : {rate:.4f}  ({rate:.1%})  [{n_ok}/{n_total}]")


# Success-rate summary written into the output directory.
summary_lines = [
    "=== success rate by planner ===",
] + [
    f"{name:<{_w}} : {rate:.4f}  ({rate:.1%})  [{n_ok}/{n_total}]"
    for name, n_ok, rate in PLANNER_REPORT
] + [
    "",
    f"data_path                    : {dataPath}",
    f"model_path                   : {modelPath}",
    f"checkpoint                   : {pt}",
    f"test_cases                   : {n_cases}",
    f"mppi_momentum                : {MOMENTUM}",
    f"mppi_step                    : {STEP}   [convergence ball is 0.01]",
    f"mppi_taper                   : {TAPER}",
    f"mppi_temp                    : {args.temp}   [softmax(-temp * cost); default 50]",
    f"mppi_planar (--2d)           : {PLANAR}"
    + (f"   [free dims {PLANAR_FREE_DIMS} = x, y, rz]" if PLANAR else ""),
    f"no_conv_min_dis_median       : "
    + (f"{np.median(_md):.4f}" if _md.size else "n/a"),
    f"no_conv_min_dis_within_2ball : "
    + (f"{int((_md < 0.02).sum())} / {_md.size}"
       f"   [overshoot: got inside 0.02 but never inside 0.01]"
       if _md.size else "n/a"),
    f"discarded_invalid_endpoints  : {n_invalid_endpoints}"
    f"  (start: {n_invalid_start}, goal: {n_invalid_goal})",
    f"episodes                     : {n_total}   [evaluated = test_cases - discarded]",
    f"successes                    : {total}",
    f"failures                     : {n_total - total}",
    f"  collision                  : {n_collision}",
    f"  no_converge                : {n_no_conv}",
    f"success_rate                 : {success_rate:.4f}  ({success_rate:.1%})",
    f"successes_flipped            : {total_flip}"
    f"  (collision: {n_collision_flip}, no_converge: {n_no_conv_flip})",
    f"success_rate_flipped         : {flip_rate:.4f}  ({flip_rate:.1%})",
    f"successes_either             : {total_either}   [regular OR flipped]",
    f"success_rate_either          : {either_rate:.4f}  ({either_rate:.1%})",
    f"success_rate_alternating     : {alt_rate:.4f}  ({alt_rate:.1%})"
    f"  [{total_alt}/{n_total}]",
    f"alternating_rescued          : {n_alt_rescued}"
    f"   [regular failed, alternating passed]",
    f"alternating_lost             : {n_alt_only}"
    f"   [regular passed, alternating failed]",
    f"alternating_collision        : {n_collision_alt} / {n_total}",
    f"alternating_no_convergence   : {n_no_conv_alt} / {n_total}",
    f"local_step_weight            : {LOCAL_W}",
    f"success_rate_local_bidir     : {lb_rate:.4f}  ({lb_rate:.1%})"
    f"  [{total_lb}/{n_total}]",
    f"local_bidir_rescued          : {n_lb_rescued}   [regular failed, local-bidir passed]",
    f"local_bidir_lost             : {n_lb_only}   [regular passed, local-bidir failed]",
    f"local_bidir_collision        : {n_collision_lb} / {n_total}",
    f"local_bidir_no_convergence   : {n_no_conv_lb} / {n_total}",
    f"success_rate_hor_local_bidir : {hb_rate:.4f}  ({hb_rate:.1%})"
    f"  [{total_hb}/{n_total}]",
    f"hor_local_bidir_rescued      : {n_hb_rescued}   [regular failed, hor+local-bidir passed]",
    f"hor_local_bidir_lost         : {n_hb_only}   [regular passed, hor+local-bidir failed]",
    f"hor_local_bidir_collision    : {n_collision_hb} / {n_total}",
    f"hor_local_bidir_no_convergence: {n_no_conv_hb} / {n_total}",
    # f"success_rate_local_bidir_indep: {lbi_rate:.4f}  ({lbi_rate:.1%})"
    # f"  [{total_lbi}/{n_total}]   [{lbi_rate - lb_rate:+.1%} vs local_bidir]",
    # f"local_bidir_indep_vs_local_bidir: won {n_lbi_over_lb}, lost {n_lb_over_lbi}"
    # f"   [head-to-head; the split cost-to-go is the only difference]",
    # f"local_bidir_indep_rescued    : {n_lbi_rescued}   [regular failed, locB_indep passed]",
    # f"local_bidir_indep_lost       : {n_lbi_only}   [regular passed, locB_indep failed]",
    # f"local_bidir_indep_collision  : {n_collision_lbi} / {n_total}",
    # f"local_bidir_indep_no_convergence: {n_no_conv_lbi} / {n_total}",
    # f"success_rate_hor_local_bidir_indep: {hbi_rate:.4f}  ({hbi_rate:.1%})"
    # f"  [{total_hbi}/{n_total}]   [{hbi_rate - hb_rate:+.1%} vs hor_local_bidir]",
    # f"hor_local_bidir_indep_vs_hor_local_bidir: won {n_hbi_over_hb}, lost {n_hb_over_hbi}"
    # f"   [head-to-head; the split cost-to-go is the only difference]",
    # f"hor_local_bidir_indep_rescued: {n_hbi_rescued}   [regular failed, hlB_indep passed]",
    # f"hor_local_bidir_indep_lost   : {n_hbi_only}   [regular passed, hlB_indep failed]",
    # f"hor_local_bidir_indep_collision: {n_collision_hbi} / {n_total}",
    # f"hor_local_bidir_indep_no_convergence: {n_no_conv_hbi} / {n_total}",
    f"rescued_by_flip              : {n_rescued}",
    f"lost_by_flip                 : {n_flip_only}"
    f"   [regular passed but flipped failed; symmetric with rescued_by_flip"
    f" iff the flip carries no directional signal]",
    f"failures_both                : {len(failed_both_idx)}"
    f"   [regular AND flipped both failed]",
    f"failure_rate_both            : {1 - either_rate:.4f}  ({1 - either_rate:.1%})",
    f"failure_corr_phi_flip        : {phi_flip:+.4f}"
    f"   [P(both fail)={both_flip_rate:.4f}, independent={indep_flip:.4f}]",
    f"linear_interp_valid          : {n_lin_valid} / {n_total}",
    f"success_rate|lin_interp_valid: {lin_valid_success_rate:.4f}  "
    f"({lin_valid_success_rate:.1%})  [{n_succ_lin_valid}/{n_lin_valid}]",
    f"success_rate|lin_interp_fail : {lin_invalid_success_rate:.4f}  "
    f"({lin_invalid_success_rate:.1%})  [{n_succ_lin_invalid}/{n_lin_invalid}]",
]
with open(os.path.join(OUTPUT_DIR, 'success_rate.txt'), 'w') as f:
    f.write('\n'.join(summary_lines) + '\n')
vprint('Wrote ' + os.path.join(OUTPUT_DIR, 'success_rate.txt'))


# ──────────────────────────────────────────────────────────────────────────────
# Summary plots
# ──────────────────────────────────────────────────────────────────────────────
success_list = np.array(success_list)
fail_list = np.array(fail_list)


if success_list.size:
    fig = go.Figure(go.Scatter3d(
        x=success_list[:, 0], y=success_list[:, 1], z=success_list[:, 2],
        mode='markers', marker=dict(size=4, color='green', opacity=0.8)))
    fig.update_layout(title='Successes', scene=dict(
        xaxis_title='dx', yaxis_title='dy', zaxis_title='dz',
        camera=dict(up=dict(x=0, y=1, z=0))))
    fig.write_html(os.path.join(OUTPUT_DIR, 'summary_success.html'), include_plotlyjs='cdn')


if fail_list.size:
    fig = go.Figure(go.Scatter3d(
        x=fail_list[:, 0], y=fail_list[:, 1], z=fail_list[:, 2],
        mode='markers', marker=dict(size=4, color='red', opacity=0.8)))
    fig.update_layout(title='Failures', scene=dict(
        xaxis_title='dx', yaxis_title='dy', zaxis_title='dz',
        camera=dict(up=dict(x=0, y=1, z=0))))
    fig.write_html(os.path.join(OUTPUT_DIR, 'summary_failures.html'), include_plotlyjs='cdn')


if test_list_speed_failed:
    fig = go.Figure(go.Histogram(
        x=test_list_speed_failed, nbinsx=10,
        marker=dict(color='skyblue', line=dict(color='black', width=1)), opacity=0.7))
    fig.update_layout(title='Distribution of Speed (Failed Cases)',
                      xaxis_title='Speed Value', yaxis_title='Frequency', bargap=0.05)
    fig.write_html(os.path.join(OUTPUT_DIR, 'summary_speed_failed.html'), include_plotlyjs='cdn')


# ──────────────────────────────────────────────────────────────────────────────
# Failure frequency tables over the (speed_dist x speed_angle) grid
# ──────────────────────────────────────────────────────────────────────────────
# Where the FAILED episodes sit in clearance space -- one table binned by the
# start's clearance, one by the goal's.  Both cover the same set of failed cases
# (an episode fails as a whole), so the two grand totals are equal; what differs
# is which endpoint's clearance is used as the coordinate.  Bins are uniform over
# [0,1], the range both proxies are already normalized to.  Cases discarded by
# the endpoint filter never ran, so they are absent from both.
def print_speed_freq_table(title, rows, sd_col, sa_col, html_name=None):
    """Frequency of `rows` over the speed_dist x speed_angle grid, with margins.

    If `html_name` is given, the same grid is also written to OUTPUT_DIR as a
    plotly heatmap (rows = angular-speed bin, cols = translational-speed bin).
    """
    if arr_speed_dists is None or arr_speed_angles is None:
        vprint(f'\n{title}: skipped (speed_dists.npy / speed_angles.npy not in {dataPath})')
        return
    if not rows:
        vprint(f'\n{title}: none')
        return
    idx = np.asarray(rows)
    sd = np.asarray(arr_speed_dists)[idx, sd_col].astype(np.float64)
    sa = np.asarray(arr_speed_angles)[idx, sa_col].astype(np.float64)

    nb_d, nb_a = args.nbins_dist, args.nbins_angle
    di = np.clip((sd * nb_d).astype(np.int64), 0, nb_d - 1)
    ai = np.clip((sa * nb_a).astype(np.int64), 0, nb_a - 1)
    grid = np.zeros((nb_a, nb_d), dtype=np.int64)
    np.add.at(grid, (ai, di), 1)

    vprint('\n' + '=' * (13 + 8 * nb_d + 9))
    vprint(f'{title}   N = {len(idx)}')
    vprint('rows = speed_angle bin, cols = speed_dist bin  (0 = tight -> 1 = open)')
    vprint('=' * (13 + 8 * nb_d + 9))
    hdr = ' angle\\dist '
    for d in range(nb_d):
        hdr += '%8s' % ('%.2f' % ((d + 0.5) / nb_d))
    vprint(hdr + '  |   total')
    for a in range(nb_a - 1, -1, -1):                 # high angle on top
        row = '%11s ' % ('%.2f' % ((a + 0.5) / nb_a))
        for d in range(nb_d):
            row += '%8d' % grid[a, d]
        vprint(row + '  |%8d' % grid[a].sum())
    vprint(' ' * 12 + '-' * (8 * nb_d + 11))
    row = '%11s ' % 'total'
    for d in range(nb_d):
        row += '%8d' % grid[:, d].sum()
    vprint(row + '  |%8d' % grid.sum())

    if html_name is not None:
        d_centers = [(d + 0.5) / nb_d for d in range(nb_d)]
        a_centers = [(a + 0.5) / nb_a for a in range(nb_a)]
        fig = go.Figure(go.Heatmap(
            z=grid, x=d_centers, y=a_centers, colorscale='Reds',
            text=grid, texttemplate='%{text}', hovertemplate=(
                'trans speed %{x:.2f}<br>angular speed %{y:.2f}'
                '<br>count %{z}<extra></extra>'),
            colorbar=dict(title='count')))
        fig.update_layout(title=f'{title}   (N = {len(idx)})',
                          xaxis_title='speed_dist bin (0 = tight -> 1 = open)',
                          yaxis_title='speed_angle bin (0 = tight -> 1 = open)')
        path = os.path.join(OUTPUT_DIR, html_name)
        fig.write_html(path, include_plotlyjs='cdn')
        vprint('Wrote ' + path)


print_speed_freq_table('FREQUENCY: FAILED episodes, binned by START clearance',
                       failed_idx, 0, 0)
print_speed_freq_table('FREQUENCY: FAILED episodes, binned by END POINT clearance',
                       failed_idx, 1, 1)


# ── Hard failures: neither the regular nor the flipped rollout solved the case ──
# These are the cases the flip could not rescue, so they isolate genuinely hard
# geometry rather than an unlucky rollout direction.  Same grid as above, one
# table/heatmap per endpoint.
print_speed_freq_table(
    'FREQUENCY: FAILED BOTH WAYS (regular AND flipped), binned by START clearance',
    failed_both_idx, 0, 0, html_name='summary_failed_both_start.html')
print_speed_freq_table(
    'FREQUENCY: FAILED BOTH WAYS (regular AND flipped), binned by END POINT clearance',
    failed_both_idx, 1, 1, html_name='summary_failed_both_end.html')


# ── Rescued: the regular run failed but the flipped run solved it ──
# These isolate the cases where the planning DIRECTION is what mattered.  The
# test pairs are symmetric by construction (preprocess_obj --testing_data draws
# both endpoints as independent uniform collision-free placements), so a global
# directional preference cannot show up in the marginal rates -- it can only show
# up here, in WHERE the rescued cases sit in clearance space.
#
# Read the two tables against each other: if the flip helps because entering a
# tight goal is harder than leaving a tight start, the rescued mass concentrates
# in LOW bins of the END table and HIGH bins of the START table.  If instead the
# two tables look alike (and alike to the overall clearance distribution), the
# rescues are just MPPI's per-rollout randomness and carry no directional signal.
#
# The baseline below is what makes the rescued tables readable: it is the same
# grid over EVERY evaluated case, so "the rescues sit in the low-clearance bins"
# can be told apart from "most cases sit in the low-clearance bins".  Compare
# rescued-cell / baseline-cell, not the raw rescued counts.
print_speed_freq_table(
    'BASELINE: ALL evaluated episodes, binned by START clearance',
    eval_idx, 0, 0)
print_speed_freq_table(
    'BASELINE: ALL evaluated episodes, binned by END POINT clearance',
    eval_idx, 1, 1)


print_speed_freq_table(
    'FREQUENCY: RESCUED BY FLIP (regular fail, flipped pass), binned by START clearance',
    rescued_idx, 0, 0, html_name='summary_rescued_start.html')
print_speed_freq_table(
    'FREQUENCY: RESCUED BY FLIP (regular fail, flipped pass), binned by END POINT clearance',
    rescued_idx, 1, 1, html_name='summary_rescued_end.html')


# -- The mirror: the regular run solved it and the flipped run did not --
# Same directional question, opposite sign.  Compare these two tables cell by
# cell against the RESCUED ones: a real directional preference makes the two sets
# differ in size AND sit in different clearance regions.  If they are the same
# size and shaped alike, both sets are just MPPI's per-rollout randomness showing
# up twice, and the flip's headline gain is a second-sample effect.
print_speed_freq_table(
    'FREQUENCY: LOST BY FLIP (regular pass, flipped fail), binned by START clearance',
    flip_only_idx, 0, 0, html_name='summary_flip_only_start.html')
print_speed_freq_table(
    'FREQUENCY: LOST BY FLIP (regular pass, flipped fail), binned by END POINT clearance',
    flip_only_idx, 1, 1, html_name='summary_flip_only_end.html')


# ──────────────────────────────────────────────────────────────────────────────
# Interactive viser viewer — browse to http://<server-ip>:8080 from your host PC
# ──────────────────────────────────────────────────────────────────────────────
vprint(f"\nSummary plots saved to {OUTPUT_DIR}/")
if args.no_viser:
    vprint('--no-viser set; skipping the interactive viewer.')
else:
    launch_viser(episodes, shape_V, shape_F, V_env_n, obst_F, wall_F)
