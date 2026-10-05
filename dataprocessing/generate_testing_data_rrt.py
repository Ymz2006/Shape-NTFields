"""Generate a testing set whose every pair is collision-free *and* known solvable.

``preprocess_obj.py --testing_data`` samples start/goal pairs that are merely
collision-free: nothing checks that a path between them exists, so a test set
built that way can contain pairs no planner could ever solve, and a success rate
computed over it has an unknown ceiling below 100%.

This script closes that gap.  Each pair must clear three independent hurdles
before it is written:

1. **Sampled clearance** -- ``preprocess_obj.generate_valid_pairs(testing=True)``
   accepts a pair only if both placements are collision-free with clearance
   greater than ``--offset``, against the ``--num_env_points`` cloud the dataset
   itself carries.  ``--offset`` is therefore the closest a start or goal is ever
   allowed to sit to an obstacle.
2. **Re-checked collision** -- both placements are re-tested from scratch against
   a *denser* ``--verify_env_points`` cloud (50 000 by default), the same one
   ``evaluate_training_3d_batched.py`` resamples at evaluation time.  This is an
   independent audit of hurdle 1 rather than a restatement of it: it uses a
   different point cloud, and it is the cloud the pair will actually be judged
   against later.
3. **Demonstrated path** -- RRT-Connect (OMPL) must return an exact,
   collision-free path between the two placements within ``--rrt_time`` seconds.
   A pair it cannot solve in that budget is discarded and resampled.

Sampling and verification interleave: candidates are drawn in rounds, verified in
a process pool, and the survivors accumulate until ``--num_samples`` pairs are
banked.  The observed acceptance rate sizes the next round, so a cell where most
pairs are solvable draws barely more than it needs and a hard one keeps going.

The collision model and the planner are imported from
``baselines/baseline_ompl/rrt_connect_eval.py`` rather than reimplemented, so
"a path exists" here means exactly what it means in the RRT-Connect baseline
table -- same point-in-tet test, same SE(3)/SE(2) state space, same resolution.

Output layout matches ``testing_data/3dshape/<cell>/`` exactly (same .npy names,
shapes and dtypes, same ``meta.json`` keys), so the result is a drop-in
``--dataPath`` for the evaluator.  One extra file, ``rrt_verification.json``,
records the audit trail: per-pair solve times and path lengths, the rejection
counts, and the settings that produced them.

Run from the main package (``ntrl-demo/ntrl-demo``), in a container that has both
torch and the OMPL bindings:

    python dataprocessing/generate_testing_data_rrt.py \
        --shape  datasets/3dshape/Lshape3d.obj \
        --env    datasets/3dshape/env1.obj \
        --offset 0.02 \
        --out    testing_data_1k_complete/3dshape/Lshape3d_env1

and for a planar cell (both meshes z-up, as in the 2-D pipeline):

    python dataprocessing/generate_testing_data_rrt.py --2d \
        --shape  datasets/3dshape/Lshape3d_zup.obj \
        --env    datasets/3dshape/2denv1_zup.obj \
        --offset 0.02 \
        --out    testing_data_1k_complete/3dshape/Lshape3d_2denv1
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch

# The main package (for ``dataprocessing.*``) and the OMPL baseline (for the
# collision model + planner) both have to be importable.  This file lives in
# <pkg>/dataprocessing/, and the baseline in <repo>/baselines/baseline_ompl/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
_REPO = os.path.abspath(os.path.join(_PKG, os.pardir, os.pardir))
_OMPL_DIR = os.path.join(_REPO, 'baselines', 'baseline_ompl')
for _p in (_PKG, _OMPL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataprocessing.preprocess_obj import (            # noqa: E402
    DEFAULT_ENV_POINTS, DEFAULT_TET_SWITCHES, TWOD_SHAPE_THICKNESS,
    generate_radius_surface_points, generate_valid_pairs, load_obj,
    sample_surface_points, tetrahedralize_shape)

TWO_PI = 2.0 * np.pi
# Points sampled over the FULL environment mesh for the re-check and for the
# planner -- the count evaluate_training_3d_batched.py uses at eval time.
VERIFY_ENV_POINTS = 50000


# ──────────────────────────────────────────────────────────────────────────────
# Worker: one OMPL scene per process, reused across every pair it verifies
# ──────────────────────────────────────────────────────────────────────────────
_W = {}


def _init_worker(payload):
    """Build this process's collision checker and planner once.

    Imported here rather than at module scope so the parent never loads OMPL --
    it only needs torch, and mixing a CUDA context with process startup is worth
    avoiding.
    """
    import rrt_connect_eval as rce

    # Seed before anything allocates an RNG -- OMPL refuses a reseed once its
    # global stream has started, and make_setup builds a planner that starts it.
    # Each worker gets its own offset so they do not explore in lockstep.
    rce.ou.setLogLevel(rce.ou.LogLevel.LOG_ERROR)
    rce.ou.RNG.setSeed(payload['seed'] + (os.getpid() % 100000))

    # One checker per independently drawn cloud.  A point-sampled collision test
    # is a Monte Carlo approximation of the real mesh-mesh one: a placement that
    # grazes an obstacle can come back free simply because no sampled point
    # happened to land inside it, and which placements those are changes with the
    # draw.  Requiring every cloud to agree makes a grazing pose very unlikely to
    # survive; the first cloud is also the one the planner itself uses.
    checkers = [rce.ShapeCollisionChecker(payload['tets'], ep, payload['radius'])
                for ep in payload['env_pts']]
    ns = argparse.Namespace(two_d=payload['two_d'], range=0.0,
                            resolution=payload['resolution'])
    space, ss = rce.make_setup(checkers[0], payload['bounds'], ns)

    _W.update(rce=rce, checker=checkers[0], checkers=checkers, space=space, ss=ss,
              two_d=payload['two_d'], rrt_time=payload['rrt_time'],
              to_state=rce.cfg_to_state_2d if payload['two_d'] else rce.cfg_to_state,
              to_pose=rce.state_to_pose_2d if payload['two_d'] else rce.state_to_pose)


def _verify_pair(job):
    """Re-check collision, then try to connect the pair with RRT-Connect.

    ``job`` is ``(idx, cfg0, cfg1)`` with the configs in stored form (rotvec
    divided by 2*pi).  Returns a dict the parent uses to accept or drop the pair;
    ``endpoints_free`` False means hurdle 2 rejected it before the planner ran.
    """
    idx, cfg0, cfg1 = job
    rce, checker, space, ss = _W['rce'], _W['checker'], _W['space'], _W['ss']
    checkers = _W['checkers']
    to_state, to_pose = _W['to_state'], _W['to_pose']
    tlimit = _W['rrt_time']

    s0 = to_state(space, cfg0)
    s1 = to_state(space, cfg1)
    p0, p1 = to_pose(s0), to_pose(s1)
    free = all(not c.in_collision(*p) for c in checkers for p in (p0, p1))
    if not free:
        return dict(idx=idx, endpoints_free=False, solved=False,
                    time_s=0.0, length=float('nan'), path_collision=False)

    ss.clear()
    ss.setStartAndGoalStates(s0, s1)
    t0 = time.perf_counter()
    ss.solve(tlimit)
    elapsed = time.perf_counter() - t0

    exact = bool(ss.haveExactSolutionPath())
    length, path_collision = float('nan'), False
    if exact:
        path = ss.getSolutionPath()
        length = float(path.length())
        # A returned path is still re-walked waypoint by waypoint: the pair is
        # only "solvable" if the path it comes with is actually clean.
        path_collision = rce.path_in_collision(path, checker, to_pose)

    return dict(idx=idx, endpoints_free=True,
                solved=bool(exact and not path_collision),
                time_s=elapsed, length=length, path_collision=path_collision)


# ──────────────────────────────────────────────────────────────────────────────
# Scene construction -- mirrors preprocess_obj.main() so the frames agree
# ──────────────────────────────────────────────────────────────────────────────
def build_scene(args):
    """Load and normalize both meshes exactly the way the preprocessor does."""
    V_env, F_env, names_env = load_obj(args.env)
    if args.two_d:
        # Planar mode collapses the environment onto z=0 *before* the bbox, so
        # normalization is driven by the x-y footprint alone.
        V_env[:, 2] = 0.0
        print('[--2d] environment flattened onto z=0')

    bb_min, bb_max = V_env.min(axis=0), V_env.max(axis=0)
    center_env = 0.5 * (bb_min + bb_max)
    scale = float((bb_max - bb_min).max())
    V_env_n = (V_env - center_env) / scale

    # 'null'-named groups are excluded from the dataset cloud, as in preprocess.
    sample_mask = np.array(['null' not in str(n).lower() for n in names_env])
    env_points = sample_surface_points(
        V_env_n, F_env[sample_mask], args.num_env_points).astype(np.float32)

    ranges = (bb_max - bb_min) / scale
    half_extent = ranges * 0.5 - 0.01
    if args.two_d:
        half_extent[2] = 0.0

    V_sh, F_sh, _ = load_obj(args.shape)
    shape_center = 0.5 * (V_sh.min(axis=0) + V_sh.max(axis=0))
    V_sh_local = (V_sh - shape_center) / scale * args.shape_scale
    if args.two_d:
        z = V_sh_local[:, 2]
        z_ext = float(z.max() - z.min())
        if z_ext > 1e-12:
            V_sh_local[:, 2] = ((z - 0.5 * (z.max() + z.min()))
                                * (TWOD_SHAPE_THICKNESS / z_ext))
        else:
            V_sh_local[:, 2] = 0.0
        print('[--2d] shape z extent {:.4f} -> {:.4f}'.format(
            z_ext, TWOD_SHAPE_THICKNESS))

    # 'Q' keeps tetgen's statistics page out of the log.
    TV, TT, TF = tetrahedralize_shape(V_sh_local, F_sh,
                                      switches=args.tet_switches + 'Q')
    tet_verts_local = torch.tensor(TV[TT], dtype=torch.float32)
    face_verts_local = torch.tensor(TV[TF], dtype=torch.float32)

    rad_points, rad_bins = generate_radius_surface_points(
        V_sh_local, F_sh, args.num_radius_points, args.radius_bins)

    # The verification cloud is denser and covers the FULL mesh (walls too) --
    # it is what the evaluator will judge these pairs against.
    verify_pts = [np.ascontiguousarray(
        sample_surface_points(V_env_n, F_env, args.verify_env_points),
        dtype=np.float64) for _ in range(args.verify_clouds)]
    tets_local = np.asarray(TV, dtype=np.float64)[TT]
    radius = float(np.linalg.norm(np.asarray(TV, dtype=np.float64), axis=1).max())

    bounds = (V_env_n.min(axis=0), V_env_n.max(axis=0))
    if args.two_d:
        bounds = (bounds[0][:2], bounds[1][:2])

    print('shape          : {}  ({} verts, {} tets, radius {:.4f})'.format(
        args.shape, len(V_sh_local), TT.shape[0], radius))
    print('environment    : {}  ({} tris, {} dataset pts, {} x {} verify pts)'.format(
        args.env, len(F_env), len(env_points), args.verify_clouds,
        len(verify_pts[0])))
    print('bounds         : low {}  high {}'.format(
        np.round(bounds[0], 4).tolist(), np.round(bounds[1], 4).tolist()))

    return dict(env_points=env_points, half_extent=half_extent,
                tet_verts_local=tet_verts_local, face_verts_local=face_verts_local,
                rad_points=rad_points, rad_bins=rad_bins,
                verify_pts=verify_pts, tets_local=tets_local, radius=radius,
                bounds=bounds, scale=scale, center_env=center_env,
                num_tets=int(TT.shape[0]))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Generate collision-free, RRT-verified testing pairs.')
    parser.add_argument('--shape', required=True,
                        help='Shape OBJ (watertight; tetrahedralized for queries).')
    parser.add_argument('--env', required=True,
                        help='Environment OBJ (static obstacles; surface only).')
    parser.add_argument('--out', required=True,
                        help='Output directory for the .npy testing data.')
    parser.add_argument('--offset', type=float, default=0.02,
                        help='Minimum clearance, in env-normalized units, that a '
                             'start or goal placement must keep from any obstacle. '
                             'This is the closest a pair is allowed to sit to the '
                             'environment.')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of verified pairs to bank.')
    parser.add_argument('--rrt_time', type=float, default=180.0,
                        help='Per-pair RRT-Connect budget in seconds; a pair not '
                             'solved within it is discarded and resampled.')
    parser.add_argument('--workers', type=int, default=8,
                        help='Verification processes to run in parallel.')
    parser.add_argument('--margin', type=float, default=0.05,
                        help='Clearance that maps to speed=1 (sets the speed '
                             'arrays; matches preprocess_obj.py).')
    parser.add_argument('--shape_scale', type=float, default=1.0,
                        help='Uniform scale applied to the shape.')
    parser.add_argument('--num_env_points', type=int, default=DEFAULT_ENV_POINTS,
                        help='Points in the dataset env cloud (written to env.npy).')
    parser.add_argument('--verify_env_points', type=int, default=VERIFY_ENV_POINTS,
                        help='Points in the denser cloud used for the collision '
                             're-check and the planner.')
    parser.add_argument('--verify_clouds', type=int, default=3,
                        help='How many independently drawn dense clouds must all '
                             'agree a placement is collision-free. 1 reproduces a '
                             'single-draw check, which lets the occasional grazing '
                             'pose through; 3 is the default.')
    parser.add_argument('--num_radius_points', type=int, default=1000,
                        help='Shape-surface points binned by radius from origin.')
    parser.add_argument('--radius_bins', type=int, default=10,
                        help='Number of radial bins for the shape-surface points.')
    parser.add_argument('--tet_switches', default=DEFAULT_TET_SWITCHES,
                        help='tetgen switches used to tetrahedralize the shape.')
    parser.add_argument('--resolution', type=float, default=0.005,
                        help='OMPL motion-validation resolution, fraction of extent.')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Sampling batch size (each batch evaluates 2x configs). '
                             'Keep this small in --2d: flattening the env defeats '
                             'the broad-phase cull and the clearance tensor stays '
                             'dense.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=1,
                        help='Seed for sampling and for OMPL.')
    parser.add_argument('--max_rounds', type=int, default=40,
                        help='Give up after this many sample/verify rounds.')
    parser.add_argument('--2d', dest='two_d', action='store_true',
                        help='Planar mode: flatten the environment onto z=0, squash '
                             'the shape in z, and sample only the (x, y, rz) slice. '
                             'Both meshes must be z-up.')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    t_start = time.time()
    scene = build_scene(args)

    payload = dict(tets=scene['tets_local'], env_pts=scene['verify_pts'],
                   radius=scene['radius'], bounds=scene['bounds'],
                   two_d=args.two_d, resolution=args.resolution,
                   rrt_time=args.rrt_time, seed=args.seed)

    want = args.num_samples
    print('\ntarget         : {} verified pairs  (offset {}, RRT budget {:.0f}s, '
          '{} workers)\n'.format(want, args.offset, args.rrt_time, args.workers))

    # Banked, verified pairs -- one entry per accepted candidate.
    keep = dict(pairs=[], dists=[], angles=[], normals=[], trans=[], rot=[],
                solve_time=[], path_len=[])
    n_sampled = n_verified = n_coll_rejected = n_rrt_rejected = n_path_coll = 0

    # ``spawn`` keeps the parent's CUDA context out of the workers, which only
    # ever touch numpy and OMPL.
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(processes=args.workers, initializer=_init_worker,
                    initargs=(payload,))
    try:
        for rnd in range(1, args.max_rounds + 1):
            have = len(keep['pairs'])
            if have >= want:
                break

            # Size the round by the acceptance rate seen so far, with a floor so
            # a bad early streak cannot stall progress.
            rate = (have / n_verified) if n_verified and have else 0.5
            rate = min(max(rate, 0.05), 1.0)
            need = want - have
            n_draw = int(min(max(int(need / rate * 1.15) + 8, 32), 4 * want))

            print('[round {}] sampling {} candidates '
                  '({}/{} banked, acceptance {:.1%})'.format(
                      rnd, n_draw, have, want, rate), flush=True)

            t0 = time.time()
            pairs, dists, angles, normals, trans_n, rot_n, _, _ = generate_valid_pairs(
                n_draw, scene['tet_verts_local'], scene['face_verts_local'],
                scene['env_points'], scene['half_extent'],
                margin=args.margin, offset=args.offset,
                rad_points=scene['rad_points'], rad_bins=scene['rad_bins'],
                batch_size=args.batch_size, device=args.device,
                testing=True, yrot=False, two_d=args.two_d,
                track_angle_pts=False)
            t_sample = time.time() - t0

            pairs = pairs.cpu().numpy()
            dists = dists.cpu().numpy()
            angles = angles.cpu().numpy()
            normals = normals.cpu().numpy()
            trans_n = trans_n.cpu().numpy()
            rot_n = rot_n.cpu().numpy()
            # Store the rotvec 2*pi-normalized, as preprocess_obj does; the
            # planner reads the pairs in exactly this form.
            pairs[:, 3:6] /= TWO_PI
            pairs[:, 9:12] /= TWO_PI
            n_sampled += len(pairs)

            jobs = [(i, pairs[i, 0:6].astype(np.float64),
                     pairs[i, 6:12].astype(np.float64)) for i in range(len(pairs))]

            t0 = time.time()
            done = 0
            for res in pool.imap_unordered(_verify_pair, jobs, chunksize=1):
                done += 1
                n_verified += 1
                i = res['idx']
                if not res['endpoints_free']:
                    n_coll_rejected += 1
                elif not res['solved']:
                    n_rrt_rejected += 1
                    if res['path_collision']:
                        n_path_coll += 1
                elif len(keep['pairs']) < want:
                    keep['pairs'].append(pairs[i])
                    keep['dists'].append(dists[i])
                    keep['angles'].append(angles[i])
                    keep['normals'].append(normals[i])
                    keep['trans'].append(trans_n[i])
                    keep['rot'].append(rot_n[i])
                    keep['solve_time'].append(res['time_s'])
                    keep['path_len'].append(res['length'])
                if done % 100 == 0 or done == len(jobs):
                    print('    verified {}/{}  banked {}/{}'.format(
                        done, len(jobs), len(keep['pairs']), want), flush=True)
                if len(keep['pairs']) >= want:
                    break
            t_verify = time.time() - t0

            print('[round {}] sample {:.1f}s  verify {:.1f}s  '
                  'banked {}/{}  (collision-rejected {}, unsolved {})'.format(
                      rnd, t_sample, t_verify, len(keep['pairs']), want,
                      n_coll_rejected, n_rrt_rejected), flush=True)
    finally:
        pool.terminate()
        pool.join()

    n_final = len(keep['pairs'])
    if n_final < want:
        print('\nWARNING: only {} of {} pairs verified after {} rounds; writing '
              'what was banked.'.format(n_final, want, args.max_rounds))

    # ── assemble and write, in the testing_data layout ───────────────────────
    pairs = np.asarray(keep['pairs'], dtype=np.float32)
    dists = np.asarray(keep['dists'], dtype=np.float32)
    angles = np.asarray(keep['angles'], dtype=np.float32)
    normals = np.asarray(keep['normals'], dtype=np.float32)
    trans_n = np.asarray(keep['trans'], dtype=np.float32)
    rot_n = np.asarray(keep['rot'], dtype=np.float32)

    speed_dists = np.clip(dists / args.margin,
                          a_min=args.offset / args.margin, a_max=1.0)
    speed_angles = angles / np.pi
    speed_pairs = (speed_dists + speed_angles) / 2

    np.save(os.path.join(args.out, 'sampled_points'), pairs)
    np.save(os.path.join(args.out, 'speed'), speed_pairs)
    np.save(os.path.join(args.out, 'speed_angles'), speed_angles)
    np.save(os.path.join(args.out, 'speed_dists'), speed_dists)
    np.save(os.path.join(args.out, 'normal'), normals)
    np.save(os.path.join(args.out, 'trans_n'), trans_n)
    np.save(os.path.join(args.out, 'rot_n'), rot_n)
    np.save(os.path.join(args.out, 'env'), scene['env_points'])

    meta = {
        'shape_scale': float(args.shape_scale),
        'shape_obj': os.path.abspath(args.shape),
        'env_obj': os.path.abspath(args.env),
        'margin': float(args.margin),
        'offset': float(args.offset),
        'rot_norm': float(TWO_PI),
        'testing_data': True,
        'yrot': False,
        'two_d': bool(args.two_d),
        'shape_z_thickness': float(TWOD_SHAPE_THICKNESS) if args.two_d else None,
        'env_scale': scene['scale'],
        'env_center': scene['center_env'].tolist(),
        'num_tets': scene['num_tets'],
        # Beyond the preprocessor's keys: what makes this set different.
        'rrt_verified': True,
        'rrt_time_limit': float(args.rrt_time),
        'verify_env_points': int(args.verify_env_points),
        'verify_clouds': int(args.verify_clouds),
    }
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    solve_time = np.asarray(keep['solve_time'], dtype=np.float64)
    path_len = np.asarray(keep['path_len'], dtype=np.float64)
    audit = {
        'pairs_written': int(n_final),
        'pairs_requested': int(want),
        'candidates_sampled': int(n_sampled),
        'candidates_verified': int(n_verified),
        'rejected_collision_recheck': int(n_coll_rejected),
        'rejected_no_rrt_path': int(n_rrt_rejected),
        'rejected_path_in_collision': int(n_path_coll),
        'acceptance_rate': float(n_final / n_verified) if n_verified else 0.0,
        'rrt_time_limit_s': float(args.rrt_time),
        'rrt_resolution': float(args.resolution),
        'verify_env_points': int(args.verify_env_points),
        'verify_clouds': int(args.verify_clouds),
        'offset': float(args.offset),
        'seed': int(args.seed),
        'wall_clock_s': float(time.time() - t_start),
        'solve_time_s': {
            'mean': float(solve_time.mean()) if n_final else float('nan'),
            'median': float(np.median(solve_time)) if n_final else float('nan'),
            'max': float(solve_time.max()) if n_final else float('nan'),
        },
        'path_length': {
            'mean': float(path_len.mean()) if n_final else float('nan'),
            'max': float(path_len.max()) if n_final else float('nan'),
        },
        'per_pair_solve_time_s': [round(float(t), 4) for t in solve_time],
        'per_pair_path_length': [round(float(x), 4) for x in path_len],
    }
    with open(os.path.join(args.out, 'rrt_verification.json'), 'w') as f:
        json.dump(audit, f, indent=2)

    print('\nWrote {} verified pairs to {}'.format(n_final, args.out))
    print('  candidates sampled        : {}  (verified {})'.format(
        n_sampled, n_verified))
    print('  rejected (collision)      : {}'.format(n_coll_rejected))
    print('  rejected (no path in {:.0f}s): {}'.format(args.rrt_time, n_rrt_rejected))
    print('  acceptance rate           : {:.1%}  [of verified]'.format(
        n_final / n_verified if n_verified else 0.0))
    print('  solve time  mean/median/max: {:.2f} / {:.2f} / {:.2f} s'.format(
        audit['solve_time_s']['mean'], audit['solve_time_s']['median'],
        audit['solve_time_s']['max']))
    print('  wall clock                : {:.1f} s'.format(audit['wall_clock_s']))


if __name__ == '__main__':
    main()
