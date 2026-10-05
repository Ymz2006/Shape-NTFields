"""OBJ based preprocessing for 3-D shape path planning.

This is the 3-D counterpart of ``dataprocessing/preprocess_dxf.py``.  Where the
DXF pipeline moves a rigid 2-D shape (the F-shape) through a 2-D environment in
SE(2), this script moves a rigid 3-D shape (a watertight OBJ mesh) through a 3-D
environment in SE(3).  It consumes two Wavefront OBJ files:

    * an environment OBJ  -- the static obstacles / walls (a triangle soup, need
                             not be watertight; only its *surface* is used)
    * a shape OBJ         -- the moving body (a closed / watertight mesh)

Just like the 2-D code triangulates the F-shape (``FSHAPE_TRIANGLES``) and runs
all collision / distance queries on that triangle soup, here we **tetrahedralize
the shape OBJ** (via libigl's tetgen wrapper).  The resulting tetrahedra are used
for the point-in-solid collision test and their boundary triangles are used for
the clearance-distance query.

The configuration space is SE(3): every sample is ``(x, y, z, rx, ry, rz)`` where
``(x, y, z)`` is the position of the shape centre and ``(rx, ry, rz)`` is a
rotation vector (axis-angle -- the direct generalization of the 2-D ``theta``,
which is just a 1-D rotation vector).  The three rotation coordinates are stored
normalized by ``2*pi`` so that all six coordinates live on a comparable scale
(the network treats them identically through its Fourier features), exactly like
``theta`` in the 2-D pipeline.

Sampling mirrors ``preprocess_dxf.py`` (and the gibson sampler it is based on):
each training row is a correlated pair ``(x0, x1)`` where

    * ``x0`` is drawn in the **narrow band** ``offset < clearance < margin`` --
      only there does the eikonal / normal loss carry useful gradient signal;
    * ``x1`` is a random SE(3) displacement away from ``x0`` (uniform direction,
      length ~ U[0, sqrt(6)] in the 2*pi-normalized SE(3) metric) and is kept iff
      it is collision-free with clearance ``> offset``.

For each endpoint we record

    * ``speed``  -- ``clip(clearance / margin, offset/margin, 1)``  (same formula
                    as the gibson sampler)
    * ``normal`` -- unit gradient of the clearance field in configuration space.
                    Built from the closest point between the placed shape and the
                    environment and converting that workspace contact normal into
                    an SE(3) direction:
                        d(clearance)/d(x,y,z) = n_ws            (contact normal)
                        d(clearance)/d(rotvec) = (s* - centre) x n_ws  (moment)
                    This generalizes the 2-D ``cross(r, n_ws)`` theta term.

Outputs (written to ``--out``):
    sampled_points.npy  (N, 12)  -- pair per row [x0(6), x1(6)]
    speed.npy           (N, 2)   -- [speed0, speed1]
    normal.npy          (N, 12)  -- [n0(6) | n1(6)]
    env.npy             (M, 3)   -- environment surface points
    meta.json                    -- shape_scale, margin, offset, ...

Usage:
    python dataprocessing/preprocess_obj.py \
        --env   datasets/3dshape/env1.obj \
        --shape datasets/3dshape/rectangle.obj \
        --out   datasets/3dshape/rectangle_env1 \
        --num_samples 400000 \
        --margin 0.1  --offset 0.01  --shape_scale 1.0
"""

import sys
sys.path.append('.')

import os
import json
import time
import argparse

import numpy as np
import torch

import plotly.graph_objects as go


# Target number of points sampled (area-weighted) on the environment surface.
DEFAULT_ENV_POINTS = 8000
# Default tetgen switches: piecewise-linear-complex, quality, preserve surface.
DEFAULT_TET_SWITCHES = "pq1.414Y"
EPS = 1e-12


# ---------------------------------------------------------------------------
# Minimal Wavefront OBJ reader (vertices + triangle faces only)
# ---------------------------------------------------------------------------
def load_obj(path):
    """Read an OBJ file into (V, F).

    Only ``v`` (vertex) and ``f`` (face) records are used; texture / normal
    indices on the face lines are ignored.  Polygonal faces with more than three
    vertices are fan-triangulated.

    Returns
    -------
    V : (n, 3) float64
    F : (m, 3) int64       (0-based triangle vertex indices)
    face_names : (m,) object   object/group name owning each triangle ('' if none)
    """
    verts = []
    faces = []
    face_names = []            # object/group name for each triangle in ``faces``
    # DEBUG: track object / group names and how many triangles each contributes.
    object_faces = []          # list of (name, face_count)
    cur_name = ''
    cur_count = 0
    with open(path, 'r') as fh:
        for line in fh:
            if line.startswith('v '):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.split()[1:]
                # each token is  v / vt / vn  -- take the (1-based) vertex index
                idx = [int(p.split('/')[0]) for p in parts]
                # OBJ allows negative (relative) indices
                idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
                # fan-triangulate any n-gon
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k + 1]])
                    face_names.append(cur_name)
                    cur_count += 1
            elif line.startswith('o ') or line.startswith('g '):
                # new object/group: flush the previous one
                if cur_count > 0 or cur_name:
                    object_faces.append((cur_name, cur_count))
                cur_name = line[2:].strip()
                cur_count = 0
    if cur_count > 0 or cur_name:
        object_faces.append((cur_name, cur_count))

    # DEBUG: report the objects found in this OBJ file.
    print("[load_obj] '{}': {} objects/groups".format(path, len(object_faces)))
    for i, (name, cnt) in enumerate(object_faces):
        print('  [{}] {!r}  ({} triangles)'.format(i, name, cnt))

    if len(verts) == 0 or len(faces) == 0:
        raise ValueError("OBJ file '{}' has no vertices or faces".format(path))
    return (np.asarray(verts, dtype=np.float64),
            np.asarray(faces, dtype=np.int64),
            np.asarray(face_names, dtype=object))


# ---------------------------------------------------------------------------
# OBJ -> environment surface point cloud  (analog of DXF sample_points)
# ---------------------------------------------------------------------------
def sample_surface_points(V, F, num_points):
    """Area-weighted uniform sampling of ``num_points`` points on a mesh surface.

    Only the vertices actually referenced by ``F`` are included as seed points
    (so corners / edges are well covered, mirroring how the DXF sampler hits
    every polyline vertex) -- vertices belonging to faces that were filtered out
    upstream are not added.
    """
    used = np.unique(F.reshape(-1))
    seed = V[used]                                    # corners of the sampled faces

    a = V[F[:, 0]]
    b = V[F[:, 1]]
    c = V[F[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = areas.sum()
    if total <= 0:
        return seed.copy()

    probs = areas / total
    n = max(int(num_points) - len(seed), 0)
    tri = np.random.choice(len(F), size=n, p=probs)

    r1 = np.sqrt(np.random.rand(n, 1))
    r2 = np.random.rand(n, 1)
    pa, pb, pc = a[tri], b[tri], c[tri]
    pts = (1 - r1) * pa + r1 * (1 - r2) * pb + r1 * r2 * pc

    return np.concatenate([seed, pts], axis=0)


# ---------------------------------------------------------------------------
# Shape OBJ -> tetrahedral mesh  (analog of FSHAPE_TRIANGLES)
# ---------------------------------------------------------------------------
def tetrahedralize_shape(V, F, switches=DEFAULT_TET_SWITCHES):
    """Tetrahedralize a watertight surface mesh.

    Tries libigl's tetgen wrapper first (the only meshing dependency installed
    in the Docker image), then falls back to the standalone ``tetgen`` package.

    Returns
    -------
    TV : (nv, 3) float32   tet-mesh vertices
    TT : (k, 4)  int64     tetrahedra (vertex indices into TV)
    TF : (mf, 3) int64     boundary surface triangles (vertex indices into TV)
    """
    V = np.ascontiguousarray(V, dtype=np.float64)
    F = np.ascontiguousarray(F, dtype=np.int64)
    try:
        import igl
        tetfn = getattr(igl, 'tetrahedralize', None)
        if tetfn is None:
            # newer layouts expose it under the copyleft sub-package
            from igl.copyleft.tetgen import tetrahedralize as tetfn  # noqa: F401

        try:
            # Newer (nanobind) binding: flags is a keyword and the call returns
            # a long tuple (TV, TT, TF, ..., status) with the status LAST.
            out = tetfn(V, F, flags=switches)
            status, TV, TT, TF = int(out[-1]), out[0], out[1], out[2]
        except TypeError:
            # Older binding: positional switches, returns (status, TV, TT, TF).
            status, TV, TT, TF = tetfn(V, F, switches)

        if status != 0:
            raise RuntimeError(
                'igl.tetrahedralize failed (status={}). Is the shape OBJ a '
                'closed, self-intersection-free mesh?'.format(status))
    except ImportError:
        import tetgen
        tet = tetgen.TetGen(V, F)
        TV, TT = tet.tetrahedralize(switches=switches)
        # boundary faces of the resulting tet mesh
        TF = tet.grid.extract_surface().faces.reshape(-1, 4)[:, 1:]

    TV = np.asarray(TV, dtype=np.float32)
    TT = np.asarray(TT, dtype=np.int64)
    TF = np.asarray(TF, dtype=np.int64)
    return TV, TT, TF


# ---------------------------------------------------------------------------
# Batched rotation-vector (axis-angle) -> rotation matrix  (Rodrigues)
# ---------------------------------------------------------------------------
def rotvec_to_matrix(rotvec):
    """(B, 3) rotation vectors -> (B, 3, 3) rotation matrices."""
    theta = torch.linalg.norm(rotvec, dim=1, keepdim=True)          # (B,1)
    axis = rotvec / (theta + EPS)                                   # (B,3)
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zero = torch.zeros_like(x)
    K = torch.stack([zero, -z, y,
                     z, zero, -x,
                     -y, x, zero], dim=1).reshape(-1, 3, 3)         # (B,3,3)
    I = torch.eye(3, device=rotvec.device).unsqueeze(0)
    s = torch.sin(theta).unsqueeze(-1)
    c = torch.cos(theta).unsqueeze(-1)
    R = I + s * K + (1 - c) * torch.matmul(K, K)
    return R


def wrap_rotvec(rotvec):
    """Wrap rotation-vector magnitudes into (-pi, pi] (equivalent rotation).

    Rotation by ``angle`` about an axis equals rotation by ``angle - 2*pi`` about
    the same axis, so this keeps the stored coordinates on a comparable scale --
    the 3-D analog of wrapping ``theta`` into ``[-pi, pi]`` in the 2-D pipeline.
    """
    ang = torch.linalg.norm(rotvec, dim=1, keepdim=True)
    axis = rotvec / (ang + EPS)
    wrapped = ((ang + np.pi) % (2 * np.pi)) - np.pi
    out = axis * wrapped
    return torch.where(ang < 1e-8, rotvec, out)


# ---------------------------------------------------------------------------
# Batched closest point on a triangle (Ericson, Real-Time Collision Detection)
# ---------------------------------------------------------------------------
def closest_point_on_triangle(p, a, b, c):
    """Closest point to ``p`` on triangle (a, b, c). All inputs broadcast to
    (..., 3); returns (..., 3)."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = (ab * ap).sum(-1)
    d2 = (ac * ap).sum(-1)

    bp = p - b
    d3 = (ab * bp).sum(-1)
    d4 = (ac * bp).sum(-1)

    cp = p - c
    d5 = (ab * cp).sum(-1)
    d6 = (ac * cp).sum(-1)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    denom = 1.0 / (va + vb + vc + EPS)
    v = (vb * denom).unsqueeze(-1)
    w = (vc * denom).unsqueeze(-1)
    res = a + v * ab + w * ac                              # interior (face) point

    # edge AB
    mAB = ((vc <= 0) & (d1 >= 0) & (d3 <= 0)).unsqueeze(-1)
    res = torch.where(mAB, a + (d1 / (d1 - d3 + EPS)).unsqueeze(-1) * ab, res)
    # edge AC
    mAC = ((vb <= 0) & (d2 >= 0) & (d6 <= 0)).unsqueeze(-1)
    res = torch.where(mAC, a + (d2 / (d2 - d6 + EPS)).unsqueeze(-1) * ac, res)
    # edge BC
    mBC = ((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)).unsqueeze(-1)
    res = torch.where(
        mBC, b + ((d4 - d3) / ((d4 - d3) + (d5 - d6) + EPS)).unsqueeze(-1) * (c - b), res)
    # vertices (most specific -> applied last so they win)
    mA = ((d1 <= 0) & (d2 <= 0)).unsqueeze(-1)
    res = torch.where(mA, a.expand_as(res), res)
    mB = ((d3 >= 0) & (d4 <= d3)).unsqueeze(-1)
    res = torch.where(mB, b.expand_as(res), res)
    mC = ((d6 >= 0) & (d5 <= d6)).unsqueeze(-1)
    res = torch.where(mC, c.expand_as(res), res)
    return res


# ---------------------------------------------------------------------------
# Closest-point clearance distance + SE(3) clearance normal
# ---------------------------------------------------------------------------
def calculate_dist(face_verts, env_points, centers):
    """Clearance of a batch of shape placements and its configuration-space normal.

    Parameters
    ----------
    face_verts : (B, F, 3, 3)  transformed shape boundary-triangle vertices
    env_points : (E, 3)        environment surface point cloud
    centers    : (B, 3)        position (centre) of each placement

    Returns
    -------
    min_dist : (B,)     minimum distance between the placed shape and the env
    normal   : (B, 6)   unit (d/dx, d/dy, d/dz, d/drx, d/dry, d/drz) direction
                        that increases clearance; rotation part chain-ruled to
                        the 2*pi-normalized coordinates used elsewhere.
    """
    B, F = face_verts.shape[0], face_verts.shape[1]
    device = face_verts.device

    a = face_verts[:, :, 0, :].view(B, 1, F, 3)        # (B,1,F,3)
    b = face_verts[:, :, 1, :].view(B, 1, F, 3)
    c = face_verts[:, :, 2, :].view(B, 1, F, 3)
    p = env_points.view(1, -1, 1, 3).to(device)        # (1,E,1,3)

    closest = closest_point_on_triangle(p, a, b, c)    # (B,E,F,3)  s* candidates
    diff = closest - p                                 # (B,E,F,3)  s* - e*
    dist = torch.linalg.norm(diff, dim=-1)             # (B,E,F)

    flat = dist.reshape(B, -1)
    min_dist, min_idx = flat.min(dim=1)                # (B,), (B,)

    flat_closest = closest.reshape(B, -1, 3)
    flat_diff = diff.reshape(B, -1, 3)
    gather = min_idx.view(B, 1, 1).expand(-1, 1, 3)
    s_star = torch.gather(flat_closest, 1, gather).squeeze(1)   # (B,3)  closest shape pt
    sel_diff = torch.gather(flat_diff, 1, gather).squeeze(1)    # (B,3)  s* - e*

    # Workspace contact normal: unit vector from the closest obstacle point
    # towards the shape -- the direction the shape moves to gain clearance.
    n_ws = sel_diff / (min_dist.unsqueeze(1) + EPS)            # (B,3)

    # Configuration-space gradient of the clearance field:
    #   d(clearance)/d(x,y,z)   = n_ws
    #   d(clearance)/d(rotvec)  = (s* - centre) x n_ws        (moment of n_ws)
    r = s_star - centers.to(device)                            # (B,3)
    moment = torch.linalg.cross(r, n_ws, dim=1)                # (B,3)

    # rotvec is stored normalized by 2*pi, so d/d(rotvec_norm) = 2*pi * d/d(rotvec)
    normal = torch.cat([n_ws, 2.0 * np.pi * moment], dim=1)    # (B,6)
    normal = normal / (torch.linalg.norm(normal, dim=1, keepdim=True) + EPS)
    return min_dist, normal


# ---------------------------------------------------------------------------
# Point-in-tetrahedron collision test (analog of point-in-triangle)
# ---------------------------------------------------------------------------
def points_inside_tets(tet_verts, env_points):
    """Does any env point fall inside any tetrahedron of each placement?

    Parameters
    ----------
    tet_verts  : (B, K, 4, 3)  transformed tetrahedron vertices
    env_points : (E, 3)        environment surface point cloud

    Returns
    -------
    is_free : (B,) bool   True iff no env point lies inside any tet.
    """
    B, K = tet_verts.shape[0], tet_verts.shape[1]
    device = tet_verts.device

    v0 = tet_verts[:, :, 0, :]                                 # (B,K,3)
    e1 = tet_verts[:, :, 1, :] - v0
    e2 = tet_verts[:, :, 2, :] - v0
    e3 = tet_verts[:, :, 3, :] - v0
    M = torch.stack([e1, e2, e3], dim=-1)                      # (B,K,3,3) columns
    Minv = torch.linalg.inv(M + EPS * torch.eye(3, device=device))

    p = env_points.view(1, 1, -1, 3).to(device)               # (1,1,E,3)
    rhs = p - v0.unsqueeze(2)                                  # (B,K,E,3)
    bary = torch.einsum('bkij,bkej->bkei', Minv, rhs)          # (B,K,E,3) = (l1,l2,l3)
    l0 = 1.0 - bary.sum(-1)                                    # (B,K,E)

    tol = -1e-6
    inside = (bary[..., 0] >= tol) & (bary[..., 1] >= tol) & \
             (bary[..., 2] >= tol) & (l0 >= tol)               # (B,K,E)
    inside_any = inside.reshape(B, -1).any(dim=1)              # (B,)
    return ~inside_any


# ---------------------------------------------------------------------------
# Per-batch placement evaluation: collision-free mask + clearance + normal
# ---------------------------------------------------------------------------
def evaluate_placements(configs, tet_verts_local, face_verts_local,
                        env_points, device):
    """Evaluate a batch of SE(3) placements.

    Parameters
    ----------
    configs          : (B, 6)      raw (x,y,z, rx,ry,rz) on CPU (rotvec in radians)
    tet_verts_local  : (K, 4, 3)   shape tetrahedra in local frame
    face_verts_local : (F, 3, 3)   shape boundary triangles in local frame
    env_points       : (E, 3)      environment surface points (CPU)
    device           : torch device

    Returns
    -------
    is_free : (B,) bool, CPU
    dist    : (B,) float, CPU      clearance distance (shape face -> nearest env pt)
    normal  : (B, 6) float, CPU    unit SE(3) clearance normal (rotvec in /2pi units)
    """
    B = configs.shape[0]
    t = configs[:, 0:3].to(device)                             # (B,3)
    R = rotvec_to_matrix(configs[:, 3:6].to(device))           # (B,3,3)

    tets = tet_verts_local.to(device).reshape(1, -1, 3)        # (1, K*4, 3)
    tets = torch.einsum('bij,nj->bni', R, tets.squeeze(0)) + t.unsqueeze(1)
    tets = tets.reshape(B, -1, 4, 3)                           # (B,K,4,3)

    faces = face_verts_local.to(device).reshape(-1, 3)         # (F*3, 3)
    faces = torch.einsum('bij,nj->bni', R, faces) + t.unsqueeze(1)
    faces = faces.reshape(B, -1, 3, 3)                         # (B,F,3,3)

    is_free = points_inside_tets(tets, env_points)
    dist, normal = calculate_dist(faces, env_points, configs[:, 0:3])
    return is_free.cpu(), dist.cpu(), normal.cpu()


# ---------------------------------------------------------------------------
# Correlated-pair rejection sampling -- mirrors preprocess_dxf.generate_valid_pairs
#
#   x0 : drawn in the narrow band  offset < dist(x0) < margin
#   x1 : x0 + random SE(3) displacement, kept iff collision-free and dist > offset
# ---------------------------------------------------------------------------
def _sample_configs(n, hx, hy, hz):
    """Sample ``n`` uniformly-random SE(3) placements (position in bbox, random
    rotation as axis-angle with angle in [0, pi])."""
    c = torch.empty(n, 6)
    c[:, 0].uniform_(-hx, hx)
    c[:, 1].uniform_(-hy, hy)
    c[:, 2].uniform_(-hz, hz)
    axis = torch.randn(n, 3)
    axis = axis / (torch.linalg.norm(axis, dim=1, keepdim=True) + EPS)
    ang = torch.rand(n, 1) * np.pi
    c[:, 3:6] = axis * ang
    return c


def generate_valid_pairs(number_pairs, tet_verts_local, face_verts_local,
                         env_points, half_extent,
                         margin, offset, batch_size=256, device='cuda',
                         testing=False, yrot=False):
    """Sample ``number_pairs`` SE(3) placement pairs.

    Training mode (default): correlated pairs -- ``x0`` is drawn in the narrow
    band ``offset < clearance < margin`` and ``x1`` is a random SE(3) displacement
    away from it, kept iff collision-free with clearance > offset.

    Testing mode (``testing=True``): ``x0`` and ``x1`` are *independent*,
    uniformly-random placements, kept iff *both* are simply collision-free (no
    clearance band, no correlation).  Use this to generate start/goal test pairs.

    Y-rotation-only mode (``yrot=True``): the x- and z-rotation components of the
    rotvec are zeroed so the shape can only rotate about the y axis.

    Returns
    -------
    pairs   : (number_pairs, 12)  raw [x0(6), x1(6)] (rotvec in radians)
    dists   : (number_pairs, 2)   raw clearance distances [d0, d1]
    normals : (number_pairs, 12)  SE(3) clearance normals [n0(6) | n1(6)]
    """
    env_t = torch.tensor(env_points, dtype=torch.float32)
    hx, hy, hz = float(half_extent[0]), float(half_extent[1]), float(half_extent[2])

    pairs = torch.zeros(number_pairs, 12)
    dists = torch.zeros(number_pairs, 2)
    normals = torch.zeros(number_pairs, 12)
    count = 0
    sqrt6 = float(np.sqrt(6.0))
    two_pi = 2.0 * np.pi

    while count < number_pairs:
        # ── Sample x0: position uniform in the env bbox; rotation uniform ──
        x0 = _sample_configs(batch_size, hx, hy, hz)

        # Y-rotation only: zero the x- and z-rotation components.
        if yrot:
            x0[:, 3] = 0
            x0[:, 5] = 0

        if testing:
            # ── x1: a second, *independent* uniformly-random placement ──
            x1 = _sample_configs(batch_size, hx, hy, hz)

            # Y-rotation only: zero the x- and z-rotation components.
            if yrot:
                x1[:, 3] = 0
                x1[:, 5] = 0
        else:
            # ── Correlated displacement to x1 in 2pi-normalized SE(3) ──
            d = torch.rand(batch_size, 6) - 0.5

            # Y-rotation only: zero the x- and z-rotation displacements.
            if yrot:
                d[:, 3] = 0
                d[:, 5] = 0

            d = d / (torch.linalg.norm(d, dim=1, keepdim=True) + EPS)
            rL = torch.rand(batch_size, 1) * sqrt6
            delta = d * rL                                      # normalized units

            x1 = x0.clone()
            x1[:, 0:3] = x0[:, 0:3] + delta[:, 0:3]
            x1[:, 3:6] = wrap_rotvec(x0[:, 3:6] + delta[:, 3:6] * two_pi)

        # ── Both endpoints must be inside the position bbox (rotation wraps) ──
        in_bbox = (x1[:, 0].abs() <= hx) & (x1[:, 1].abs() <= hy) & (x1[:, 2].abs() <= hz)
        if int(in_bbox.sum()) == 0:
            continue
        x0 = x0[in_bbox]
        x1 = x1[in_bbox]

        # ── Evaluate both endpoints ──
        free0, dist0, normal0 = evaluate_placements(
            x0, tet_verts_local, face_verts_local, env_t, device)
        free1, dist1, normal1 = evaluate_placements(
            x1, tet_verts_local, face_verts_local, env_t, device)

        if testing:
            # ── Testing data: independent points, only require no collision ──
            keep = free0 & free1 & (dist1 > offset)  & (dist0 > offset) 
        else:
            # ── Asymmetric filtering (matches the 2-D / gibson sampler) ──
            keep_x0 = free0 & (dist0 > offset) & (dist0 < margin)   # narrow band
            keep_x1 = free1 & (dist1 > offset)                      # loose: reachable
            keep = keep_x0 & keep_x1

        nk = int(keep.sum())
        if nk == 0:
            continue

        take = min(nk, number_pairs - count)
        idx = torch.nonzero(keep, as_tuple=False).squeeze(1)[:take]

        pairs[count:count + take, 0:6] = x0[idx]
        pairs[count:count + take, 6:12] = x1[idx]
        dists[count:count + take, 0] = dist0[idx]
        dists[count:count + take, 1] = dist1[idx]
        normals[count:count + take, 0:6] = normal0[idx]
        normals[count:count + take, 6:12] = normal1[idx]
        count += take
        print('generated: {} / {}'.format(count, number_pairs))

    return pairs, dists, normals


# ---------------------------------------------------------------------------
# Optional debug visualization of the sampled data
# ---------------------------------------------------------------------------
def _rotvec_to_matrix_np(rv):
    """Single rotation vector (3,) -> rotation matrix (3, 3) (Rodrigues, numpy)."""
    theta = float(np.linalg.norm(rv))
    if theta < 1e-8:
        return np.eye(3)
    k = rv / theta
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def visual_training(configs, speed, env_points, shape_V, shape_F, save_path,
                    wall_V=None, wall_F=None, cnt=20):
    """Interactive 3-D plotly view of the sampled placements.

    Mirrors the plotly style used in ``evaluate_training.py``: the environment is
    drawn as sampled grey points (with the walls shown as a translucent mesh) and
    a subset of the sampled shape placements are drawn as their actual transformed
    ``Mesh3d`` (one combined mesh, vertices colored by clearance speed).  Written
    to a self-contained HTML file.

    Parameters
    ----------
    configs : (N, 6)  raw placements [x, y, z, rx, ry, rz] (rotvec in radians)
    speed   : (N,)    per-placement clearance speed (for coloring)
    shape_V : (nv, 3) shape mesh vertices in local frame
    shape_F : (nf, 3) shape mesh triangles
    wall_V  : (nv, 3) env vertices (normalized) for the translucent wall mesh
    wall_F  : (nf, 3) wall triangles (the faces excluded from point sampling)
    """
    traces = []

    # ── Walls: translucent mesh (bound the workspace; not sampled as obstacles) ──
    if wall_V is not None and wall_F is not None and len(wall_F) > 0:
        traces.append(go.Mesh3d(
            x=wall_V[:, 0], y=wall_V[:, 1], z=wall_V[:, 2],
            i=wall_F[:, 0], j=wall_F[:, 1], k=wall_F[:, 2],
            color='lightblue', opacity=0.15, name='walls',
            flatshading=True,
        ))

    # ── Environment: sampled surface points ──
    if len(env_points) > 0:
        sub = env_points[np.random.choice(
            len(env_points), size=min(len(env_points), 8000), replace=False)]
        traces.append(go.Scatter3d(
            x=sub[:, 0], y=sub[:, 1], z=sub[:, 2],
            mode='markers', name='environment',
            marker=dict(size=1.5, color='grey', opacity=0.3),
        ))

    # ── Shape: a subset of placements as transformed meshes ──
    m = min(cnt, len(configs))
    nv = shape_V.shape[0]
    verts_all, faces_all, intensity_all = [], [], []
    for i in range(m):
        R = _rotvec_to_matrix_np(configs[i, 3:6])
        Vi = shape_V @ R.T + configs[i, 0:3]
        verts_all.append(Vi)
        faces_all.append(shape_F + i * nv)
        intensity_all.append(np.full(nv, speed[i]))
    if verts_all:
        V = np.concatenate(verts_all, axis=0)
        Fc = np.concatenate(faces_all, axis=0)
        inten = np.concatenate(intensity_all, axis=0)
        traces.append(go.Mesh3d(
            x=V[:, 0], y=V[:, 1], z=V[:, 2],
            i=Fc[:, 0], j=Fc[:, 1], k=Fc[:, 2],
            intensity=inten, colorscale='Viridis',
            cmin=float(speed.min()), cmax=float(speed.max()),
            opacity=1.0, name='shape placements', flatshading=True,
            colorbar=dict(title='clearance speed'),
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title='Sampled placements (shape mesh colored by clearance speed)',
        scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z',
                   aspectmode='data',
                   camera=dict(up=dict(x=0, y=1, z=0))))
    fig.write_html(save_path, include_plotlyjs='cdn')
    print('Saved sample visualization: ' + save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='OBJ-based preprocessing for 3-D shape path planning.')
    parser.add_argument('--env', default='datasets/3dshape/env1.obj',
                        help='Environment OBJ (static obstacles; surface only).')
    parser.add_argument('--shape', default='datasets/3dshape/rectangle.obj',
                        help='Shape OBJ (watertight; tetrahedralized for queries).')
    parser.add_argument('--out', default='datasets/3dshape/rectangle_env1',
                        help='Output directory for the .npy training data.')
    parser.add_argument('--num_samples', type=int, default=400000,
                        help='Total number of sampled configs (split into pairs).')
    parser.add_argument('--shape_scale', type=float, default=1.0,
                        help='Uniform scale applied to the shape (1.0 = baseline '
                             'size in env-normalized units, <1 shrinks).')
    parser.add_argument('--margin', type=float, default=0.1,
                        help='Upper band: x0 must have clearance < margin; maps to speed=1.')
    parser.add_argument('--offset', type=float, default=0.01,
                        help='Lower band: x0 must have clearance > offset, and x1 '
                             '(paired goal) must also have clearance > offset. '
                             'Maps to the minimum speed value offset/margin.')
    parser.add_argument('--num_env_points', type=int, default=DEFAULT_ENV_POINTS,
                        help='Target number of points sampled on the env surface.')
    parser.add_argument('--tet_switches', default=DEFAULT_TET_SWITCHES,
                        help='tetgen switches used to tetrahedralize the shape.')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Sampling batch size (each batch evaluates 2x configs).')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--visualize', action='store_true',
                        help='Also save a debug plot of the sampled placements.')
    parser.add_argument('--testing_data', action='store_true',
                        help='Generate TEST pairs: x0 and x1 are independent, '
                             'uniformly-random placements kept only if both are '
                             'collision-free (no narrow-band / clearance filtering).')
    parser.add_argument('--yrot', action='store_true',
                        help='Restrict rotation to the y axis only: zero the x- '
                             'and z-rotation components of every sampled rotvec.')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ── Environment: load, normalize to a unit box, sample its surface ──
    V_env, F_env, names_env = load_obj(args.env)
    # Bounding box uses *all* geometry (walls included) so the normalized frame
    # matches the true extent of the environment.
    bb_min = V_env.min(axis=0)
    bb_max = V_env.max(axis=0)
    center_env = 0.5 * (bb_min + bb_max)
    scale = float((bb_max - bb_min).max())            # largest extent -> 1.0
    V_env_n = (V_env - center_env) / scale

    # Do NOT sample surface points on objects whose name contains 'wall' -- they
    # bound the workspace but should not act as collision obstacles to plan around.
    sample_mask = np.array(['null' not in str(n).lower() for n in names_env])
    n_wall = int((~sample_mask).sum())
    print('[env] {} / {} triangles excluded from sampling (wall objects)'.format(
        n_wall, len(F_env)))
    F_sample = F_env[sample_mask]

    env_points = sample_surface_points(V_env_n, F_sample, args.num_env_points)
    env_points = env_points.astype(np.float32)

    ranges = (bb_max - bb_min) / scale
    half_extent = ranges * 0.5 - 0.01                 # small inset, like the 2-D code

    # ── Shape: load, normalize into the same units, tetrahedralize ──
    V_sh, F_sh, _ = load_obj(args.shape)
    shape_center = 0.5 * (V_sh.min(axis=0) + V_sh.max(axis=0))
    V_sh_local = (V_sh - shape_center) / scale * args.shape_scale

    TV, TT, TF = tetrahedralize_shape(V_sh_local, F_sh, switches=args.tet_switches)
    tet_verts_local = torch.tensor(TV[TT], dtype=torch.float32)     # (K,4,3)
    face_verts_local = torch.tensor(TV[TF], dtype=torch.float32)    # (F,3,3)
    print('Tetrahedralized shape: {} tets, {} boundary triangles, {} verts'.format(
        TT.shape[0], TF.shape[0], TV.shape[0]))

    num_pairs = int(args.num_samples)
    mode = 'TESTING (independent collision-free pairs)' if args.testing_data \
        else 'TRAINING (narrow-band correlated pairs)'
    print('Sampling {} (x0, x1) pairs [{}]   margin={}  offset={}  ...'.format(
        num_pairs, mode, args.margin, args.offset))
    t0 = time.time()
    pairs, dists, normals = generate_valid_pairs(
        num_pairs, tet_verts_local, face_verts_local, env_points, half_extent,
        margin=args.margin, offset=args.offset,
        batch_size=args.batch_size, device=args.device,
        testing=args.testing_data, yrot=args.yrot)
    print('Sampling done in {:.1f}s'.format(time.time() - t0))

    pairs = pairs.cpu().numpy()
    dists = dists.cpu().numpy()           # (N, 2)
    normals = normals.cpu().numpy()       # (N, 12)

    # Eikonal speed term: clearance clipped into [offset/margin, 1].
    speed_pairs = np.clip(dists / args.margin,
                          a_min=args.offset / args.margin, a_max=1.0)

    if args.visualize:
        visual_training(pairs[:, 0:6].copy(), speed_pairs[:, 0], env_points,
                        V_sh_local, F_sh,
                        save_path=os.path.join(args.out, 'sampled_placements.html'),
                        wall_V=V_env_n, wall_F=F_env[~sample_mask])

    # rotvec -> normalized by 2*pi so all six coords share a comparable scale.
    pairs[:, 3:6] /= (2 * np.pi)
    pairs[:, 9:12] /= (2 * np.pi)

    print('x0 pos    : x=[{:.3f},{:.3f}]  y=[{:.3f},{:.3f}]  z=[{:.3f},{:.3f}]'.format(
        pairs[:, 0].min(), pairs[:, 0].max(),
        pairs[:, 1].min(), pairs[:, 1].max(),
        pairs[:, 2].min(), pairs[:, 2].max()))
    print('speed x0  : [{:.3f}, {:.3f}]   (band [{:.3f}, 1.0])'.format(
        speed_pairs[:, 0].min(), speed_pairs[:, 0].max(),
        args.offset / args.margin))
    print('speed x1  : [{:.3f}, {:.3f}]'.format(
        speed_pairs[:, 1].min(), speed_pairs[:, 1].max()))

    np.save(os.path.join(args.out, 'sampled_points'), pairs)
    np.save(os.path.join(args.out, 'speed'), speed_pairs)
    np.save(os.path.join(args.out, 'normal'), normals)
    np.save(os.path.join(args.out, 'env'), env_points)

    meta = {
        'shape_scale': float(args.shape_scale),
        'shape_obj':   os.path.abspath(args.shape),
        'env_obj':     os.path.abspath(args.env),
        'margin':      float(args.margin),
        'offset':      float(args.offset),
        'rot_norm':    float(2 * np.pi),
        'testing_data': bool(args.testing_data),
        'yrot':        bool(args.yrot),
        'env_scale':   scale,
        'env_center':  center_env.tolist(),
        'num_tets':    int(TT.shape[0]),
    }
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print('Wrote {} training pairs to {}'.format(pairs.shape[0], args.out))
    print('  sampled_points.npy', pairs.shape)
    print('  speed.npy         ', speed_pairs.shape)
    print('  normal.npy        ', normals.shape)
    print('  env.npy           ', env_points.shape)


if __name__ == '__main__':
    main()
