"""Give an environment OBJ the enclosing wall box the pipeline expects.

``preprocess_obj.py`` normalizes an environment by ``scale = largest bbox extent``
and samples start/goal placements inside that bbox.  When the OBJ contains an
enclosing wall (a group whose name contains ``wall``, as ``env1.obj`` has) the
bbox is the ROOM, so placements are drawn throughout the room and the wall
rejects any that leave it.  When there is no wall -- ``env2.obj`` -- the bbox
collapses onto the obstacle cluster, ``env_scale`` drops, every normalized length
(the shape, ``margin``, ``offset``) inflates by the same factor, and nothing stops
a placement from hanging out in the void beyond the obstacles.

This script copies an environment OBJ and appends a wall box to it, either taken
verbatim from a reference environment (``--like env1.obj``, which keeps the two
environments in the SAME normalized frame -- same room, different obstacle
layout) or built as a padded box around the obstacles.  Non-destructive: it
writes a new file and never touches the input.

    python dataprocessing/add_env_wall.py \
        --env datasets/3dshape/env2.obj \
        --like datasets/3dshape/env1.obj \
        --out datasets/3dshape/env2_walled.obj
"""

import argparse
import os
import sys

import numpy as np

sys.path.append('.')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataprocessing.preprocess_obj import load_obj

# The 12 triangles of a box, as 0-based indices into the 8 corners emitted by
# ``box_corners`` below.  Winding is irrelevant to the pipeline (the collision
# test is a winding-number query, and the renderer draws walls double-sided), but
# they are kept consistent so the box is a clean closed solid.
BOX_TRIS = np.array([
    [0, 1, 2], [0, 2, 3],      # -x
    [4, 6, 5], [4, 7, 6],      # +x
    [0, 4, 5], [0, 5, 1],      # -y
    [3, 2, 6], [3, 6, 7],      # +y
    [0, 3, 7], [0, 7, 4],      # -z
    [1, 5, 6], [1, 6, 2],      # +z
], dtype=np.int64)


def box_corners(lo, hi):
    """The 8 corners of an axis-aligned box, ordered to match ``BOX_TRIS``."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    return np.array([
        [x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0],
        [x1, y0, z0], [x1, y0, z1], [x1, y1, z1], [x1, y1, z0],
    ], dtype=np.float64)


def wall_of(path):
    """Bounding box of the wall group in ``path``, or None if it has no wall."""
    V, F, names = load_obj(path)
    mask = np.array(['wall' in str(n).lower() for n in names])
    if not mask.any():
        return None
    W = V[np.unique(F[mask])]
    return W.min(axis=0), W.max(axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--env', required=True, help='environment OBJ to add a wall to')
    ap.add_argument('--like', default=None,
                    help='reference environment whose wall box is copied verbatim, so '
                         'both environments share one normalized frame')
    ap.add_argument('--pad', type=float, default=None,
                    help='instead of --like, build the wall as the obstacle bbox grown '
                         'by this many raw units on every side')
    ap.add_argument('--name', default='wall (2)',
                    help='group name for the wall; must contain "wall" or the pipeline '
                         'will treat it as a solid obstacle')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    assert (args.like is None) != (args.pad is None), 'pass exactly one of --like / --pad'
    assert 'wall' in args.name.lower(), (
        f'--name {args.name!r} must contain "wall"; preprocess_obj and the eval '
        f'viewer both key off that substring')

    V, F, names = load_obj(args.env)
    if any('wall' in str(n).lower() for n in names):
        print(f'WARNING: {args.env} already has a wall group; appending another.')
    obst_lo, obst_hi = V.min(axis=0), V.max(axis=0)

    if args.like:
        wall = wall_of(args.like)
        assert wall is not None, f'{args.like} has no wall group to copy'
        lo, hi = wall
        src = f'copied from {os.path.basename(args.like)}'
        outside = (obst_lo < lo - 1e-6).any() or (obst_hi > hi + 1e-6).any()
        if outside:
            print(f'WARNING: the obstacles of {args.env} stick out of '
                  f'{args.like}\'s wall box (obstacles {np.round(obst_lo, 2)} .. '
                  f'{np.round(obst_hi, 2)} vs wall {np.round(lo, 2)} .. '
                  f'{np.round(hi, 2)}); the room does not contain them.')
    else:
        lo, hi = obst_lo - args.pad, obst_hi + args.pad
        src = f'obstacle bbox padded by {args.pad}'

    corners = box_corners(lo, hi)
    base = len(V)                                    # OBJ vertex indices are 1-based

    with open(args.env) as fh:
        body = fh.read()
    if not body.endswith('\n'):
        body += '\n'
    lines = [body, f'\ng {args.name}\n\n']
    lines += [f'v {x:.6f} {y:.6f} {z:.6f}\n' for x, y, z in corners]
    lines.append('\n')
    lines += [f'f {base + a + 1} {base + b + 1} {base + c + 1}\n' for a, b, c in BOX_TRIS]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as fh:
        fh.writelines(lines)

    old_scale = float((obst_hi - obst_lo).max())
    new_lo = np.minimum(obst_lo, lo)
    new_hi = np.maximum(obst_hi, hi)
    new_scale = float((new_hi - new_lo).max())
    print(f'[wall] {src}: {np.round(lo, 2)} .. {np.round(hi, 2)}')
    print(f'[wall] wrote {args.out}')
    print(f'[wall] env_scale {old_scale:.2f} -> {new_scale:.2f} '
          f'(every normalized length shrinks by {new_scale / old_scale:.3f}x)')


if __name__ == '__main__':
    main()
