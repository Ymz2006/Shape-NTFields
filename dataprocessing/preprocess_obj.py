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

A planar ``--2d`` mode is available for problems that are really SE(2): it
flattens the environment onto ``z = 0``, squashes the shape to a very thin slab
in z (``TWOD_SHAPE_THICKNESS``, so tetrahedralization still succeeds), and samples
only the ``(x, y, rz)`` slice -- ``z`` is pinned to 0 and the rotvec is
``(0, 0, rz)``.  Everything else (clearance, angle clearance, normals, speeds,
output files) is produced exactly as in the full SE(3) pipeline.

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
    trans_n.npy         (N, 12)  -- [trans_n0(6) | trans_n1(6)]  unit translation
                                    direction that reduces obstacle distance most
                                    (translation half non-zero, rotation half 0)
    rot_n.npy           (N, 12)  -- [rot_n0(6) | rot_n1(6)]  unit rotation axis that
                                    reduces rotational clearance most (rotation half
                                    non-zero, translation half 0)
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
DEFAULT_ENV_POINTS = 10000
# Default tetgen switches: piecewise-linear-complex, quality, preserve surface.
DEFAULT_TET_SWITCHES = "pq1.414Y"
EPS = 1e-12
# --2d mode: thickness (in env-normalized units) the shape's z extent is squashed
# to.  Kept non-zero so the mesh stays a solid and tetgen can still tetrahedralize
# it, but small enough that the body behaves like a flat 2-D footprint.
TWOD_SHAPE_THICKNESS = 0.01


# ---------------------------------------------------------------------------
# Lightweight block profiler (enabled by --profile)
# ---------------------------------------------------------------------------
# Accumulates wall-clock time per named block across all batches.  CUDA work is
# async, so we synchronize around each block to attribute time correctly; that
# sync has a cost, hence the whole thing is gated off unless --profile is set.
_PROFILE = False
_TIMERS = {}
_TIC = {}


def _prof_tic(name, device):
    if not _PROFILE:
        return
    if 'cuda' in str(device):
        torch.cuda.synchronize()
    _TIC[name] = time.perf_counter()


def _prof_toc(name, device):
    if not _PROFILE:
        return
    if 'cuda' in str(device):
        torch.cuda.synchronize()
    _TIMERS[name] = _TIMERS.get(name, 0.0) + (time.perf_counter() - _TIC[name])


def _prof_report():
    if not _TIMERS:
        return
    total = sum(_TIMERS.values())
    print('\n── evaluate_placements block timing (cumulative across batches) ──')
    for name, sec in sorted(_TIMERS.items(), key=lambda kv: -kv[1]):
        print('  {:16s} {:9.2f}s  ({:5.1f}%)'.format(name, sec, 100.0 * sec / total))
    print('  {:16s} {:9.2f}s'.format('TOTAL (measured)', total))


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


def generate_radius_surface_points(V, F, num_points, bins):
    """Sample surface points and group them into equal-size radial bins.

    ``num_points`` points are sampled area-uniformly on the surface, sorted by
    their distance from the origin, and split into ``bins`` consecutive groups
    of equal size: bin ``0`` holds the ``num_points // bins`` points closest to
    the origin, the final bin holds the farthest.  Any remainder points (when
    ``num_points`` is not divisible by ``bins``) are dropped.

    Parameters
    ----------
    V, F      : mesh vertices ``(nv, 3)`` and triangles ``(nf, 3)``.
    num_points : total number of points to sample on the surface.
    bins       : number of radial bins to split the sorted points into.

    Returns
    -------
    binned : (bins, num, 3) float32   surface points grouped by radial bin,
                                       where ``num = num_points // bins``.
    edges  : (bins + 1,)   float32    radial bin boundaries; bin ``i`` holds
                                       points with ``edges[i] <= r < edges[i+1]``
                                       (``edges[0]`` = smallest radius,
                                       ``edges[bins]`` = largest radius).
    """
    a = V[F[:, 0]]
    b = V[F[:, 1]]
    c = V[F[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = areas.sum()
    if total <= 0:
        raise ValueError("mesh has zero surface area; cannot sample points")

    n = int(num_points)
    probs = areas / total
    tri = np.random.choice(len(F), size=n, p=probs)

    # Uniform barycentric sample within each chosen triangle.
    r1 = np.sqrt(np.random.rand(n, 1))
    r2 = np.random.rand(n, 1)
    pa, pb, pc = a[tri], b[tri], c[tri]
    pts = (1 - r1) * pa + r1 * (1 - r2) * pb + r1 * r2 * pc        # (n, 3)

    pts = torch.tensor(pts, dtype=torch.float32)
    radius = torch.linalg.norm(pts, dim=1)                        # (n,)

    # Sort by distance from origin (ascending) so bin 0 is the closest shell.
    order = torch.argsort(radius)
    pts = pts[order]
    radius = radius[order]

    num = n // bins
    keep = num * bins                                             # drop remainder
    pts = pts[:keep]
    radius = radius[:keep]

    binned = pts.reshape(bins, num, 3)                            # (bins, num, 3)

    # Bin boundaries: lower-edge radius of each bin plus the global max radius.
    radius_per_bin = radius.reshape(bins, num)
    edges = torch.empty(bins + 1, dtype=torch.float32)
    edges[:bins] = radius_per_bin[:, 0]
    edges[bins] = radius[-1]

    return binned, edges


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
    env_points : (B, M, 3)     per-placement env points (broad-phase culled set)
    centers    : (B, 3)        position (centre) of each placement

    Returns
    -------
    min_dist : (B,)     minimum distance between the placed shape and the env
    normal   : (B, 6)   unit (d/dx, d/dy, d/dz, d/drx, d/dry, d/drz) direction
                        that increases clearance; rotation part chain-ruled to
                        the 2*pi-normalized coordinates used elsewhere.
    n_ws     : (B, 3)   unit workspace contact normal (direction the shape moves
                        to gain translational clearance).
    """
    B, F = face_verts.shape[0], face_verts.shape[1]
    device = face_verts.device

    a = face_verts[:, :, 0, :].view(B, 1, F, 3)        # (B,1,F,3)
    b = face_verts[:, :, 1, :].view(B, 1, F, 3)
    c = face_verts[:, :, 2, :].view(B, 1, F, 3)
    p = env_points.to(device).unsqueeze(2)             # (B,M,1,3)

    closest = closest_point_on_triangle(p, a, b, c)    # (B,M,F,3)  s* candidates
    diff = closest - p                                 # (B,M,F,3)  s* - e*
    dist = torch.linalg.norm(diff, dim=-1)             # (B,M,F)

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
    return min_dist, normal, n_ws


# ---------------------------------------------------------------------------
# Point-in-tetrahedron collision test (analog of point-in-triangle)
# ---------------------------------------------------------------------------
def points_inside_tets(tet_verts, env_points):
    """Does any env point fall inside any tetrahedron of each placement?

    Parameters
    ----------
    tet_verts  : (B, K, 4, 3)  transformed tetrahedron vertices
    env_points : (B, M, 3)     per-placement env points (broad-phase culled set)

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

    p = env_points.to(device).unsqueeze(1)                    # (B,1,M,3)
    rhs = p - v0.unsqueeze(2)                                  # (B,K,M,3)
    bary = torch.einsum('bkij,bkej->bkei', Minv, rhs)          # (B,K,M,3) = (l1,l2,l3)
    l0 = 1.0 - bary.sum(-1)                                    # (B,K,M)

    tol = -1e-6
    inside = (bary[..., 0] >= tol) & (bary[..., 1] >= tol) & \
             (bary[..., 2] >= tol) & (l0 >= tol)               # (B,K,E)
    inside_any = inside.reshape(B, -1).any(dim=1)              # (B,)
    return ~inside_any


# ---------------------------------------------------------------------------
# Per-batch placement evaluation: collision-free mask + clearance + normal
# ---------------------------------------------------------------------------
def evaluate_placements(configs, tet_verts_local, face_verts_local,
                        env_points, rad_points, rad_bins, device,
                        return_angle_pts=False, two_d=False):
    """Evaluate a batch of SE(3) placements.

    Parameters
    ----------
    configs          : (B, 6)      raw (x,y,z, rx,ry,rz) on CPU (rotvec in radians)
    tet_verts_local  : (K, 4, 3)   shape tetrahedra in local frame
    face_verts_local : (F, 3, 3)   shape boundary triangles in local frame
    env_points       : (E, 3)      environment surface points (CPU)
    device           : torch device
    return_angle_pts : bool        if True, also return the shape-surface point and
                                    env point that realize the min-angle clearance
                                    (debug/visualization only; adds an argmax).
    two_d            : bool        planar (``--2d``) mode: project every returned
                                    SE(3) direction onto the (x, y, rz) sub-space
                                    before renormalizing.

    Returns
    -------
    is_free : (B,) bool, CPU
    dist    : (B,) float, CPU      clearance distance (shape face -> nearest env pt)
    min_angle : (B,) float, CPU    smallest shape-dir / env-dir angle (rotational clearance)
    normal  : (B, 6) float, CPU    unit SE(3) clearance normal (rotvec in /2pi units)
    trans_n : (B, 6) float, CPU    unit translation direction that *reduces* the
                                   distance to the obstacle the most. Only the
                                   first 3 dims are non-zero (= -n_ws); last 3 = 0.
    rot_n   : (B, 6) float, CPU    unit rotation axis that *reduces* the rotational
                                   clearance (min angle) the most. Only the last 3
                                   dims are non-zero (= unit(u_s x u_e)); first 3 = 0.

    If ``return_angle_pts`` is True, two extra tensors follow:
    shape_pt : (B, 3) float, CPU   winning shape-surface point (world frame)
    env_pt   : (B, 3) float, CPU   winning env point (world frame)
    """
    B = configs.shape[0]
    # Move everything to the device once, up front; all uses below stay on-device.
    # generate_valid_pairs already hoists the static geometry, so those .to() calls
    # are no-ops there -- but doing it here too keeps the function self-contained
    # (correct when called directly with host tensors).
    configs = configs.to(device)
    env_points = env_points.to(device)
    tet_verts_local = tet_verts_local.to(device)
    face_verts_local = face_verts_local.to(device)
    rad_points = rad_points.to(device)
    rad_bins = rad_bins.to(device)
    t = configs[:, 0:3]                                        # (B,3)
    R = rotvec_to_matrix(configs[:, 3:6])                      # (B,3,3)

    _prof_tic('transform', device)
    tets = tet_verts_local.reshape(1, -1, 3)                   # (1, K*4, 3)
    tets = torch.einsum('bij,nj->bni', R, tets.squeeze(0)) + t.unsqueeze(1)
    tets = tets.reshape(B, -1, 4, 3)                           # (B,K,4,3)

    faces = face_verts_local.reshape(-1, 3)                    # (F*3, 3)
    faces = torch.einsum('bij,nj->bni', R, faces) + t.unsqueeze(1)
    faces = faces.reshape(B, -1, 3, 3)                         # (B,F,3,3)

    bins, num = rad_points.shape[0], rad_points.shape[1]
    rad = rad_points.reshape(-1, 3)                            # (bins*num, 3)
    translated_rad_points = torch.einsum('bij,nj->bni', R, rad) + t.unsqueeze(1)
    translated_rad_points = translated_rad_points.reshape(B, bins, num, 3)   # (B,bins,num,3)
    _prof_toc('transform', device)

    _prof_tic('cull', device)
    # Vector from each placement centre to every env point, and its length.
    to_env = env_points.unsqueeze(0) - t.unsqueeze(1)              # (B,E,3)
    to_env_dist = torch.linalg.norm(to_env, dim=-1)                # (B,E)

    # ── Broad-phase env cull (correctness-preserving) ──────────────────────
    # d(e, shape) is bounded by |to_env_dist(e) - R_shape| .. to_env_dist(e)+R,
    # so the true nearest-to-surface env point always has
    #   to_env_dist <= min_center_dist + 2*R_shape.
    # Keep the nearest such points per placement (sized to the batch max so it
    # stays a dense tensor) and run the expensive clearance / collision queries
    # on this subset only -- the min distance, contact normal, and collision
    # result are all unchanged, but the (B,E,F) / (B,K,E) work shrinks with E.
    R_shape = face_verts_local.reshape(-1, 3).norm(dim=1).max()
    sorted_d, sort_idx = torch.sort(to_env_dist, dim=1)            # (B,E) ascending
    within = sorted_d <= (sorted_d[:, 0:1] + 2.0 * R_shape)       # (B,E)
    num_keep = int(within.sum(dim=1).max().clamp(min=1))
    keep_idx = sort_idx[:, :num_keep]                             # (B,num_keep) nearest
    env_kept = env_points[keep_idx]                              # (B,num_keep,3)
    _prof_toc('cull', device)

    _prof_tic('angle_binning', device)
    # Group env points into the shape's radial shells.  Bin i spans
    # (rad_bins[i], rad_bins[i+1]]; points below the smallest / above the largest
    # shell radius can never be reached by rotating the shape and are dropped.
    #
    # The cull above already produced a by-distance sort (sorted_d, sort_idx).
    # Because the shells are radial, that single sort *also* orders points by bin
    # (each shell is a contiguous run in sorted order), so we slice the bins out
    # with searchsorted + one gather -- no per-bin sort.  right=True reproduces
    # bucketize(to_env_dist) - 1 exactly: bin i == positions [pos[i], pos[i+1]).
    E = to_env.shape[1]
    edges = rad_bins
    pos = torch.searchsorted(sorted_d, edges.expand(B, -1).contiguous(), right=True)  # (B,bins+1)
    counts = pos[:, 1:] - pos[:, :bins]                            # (B,bins) per-shell occupancy
    num_env = int(counts.max().clamp(min=1))                       # pad width = largest shell
    empty_bins = counts == 0                                       # (B,bins)

    # Position of each (bin, slot) in the by-distance order.  Invalid/padding
    # slots point at the bin start (a real in-bin point -> can't beat the bin's
    # true max), clamped to stay in range; empty shells are overwritten below.
    slot = torch.arange(num_env, device=device)
    starts = pos[:, :bins].unsqueeze(-1)                           # (B,bins,1)
    cols = starts + slot                                           # (B,bins,num_env)
    valid = slot < counts.unsqueeze(-1)                            # (B,bins,num_env)
    cols = torch.where(valid, cols, starts).clamp(max=E - 1)
    flat_cols = cols.reshape(B, bins * num_env)                    # (B, bins*num_env)

    # binned_env : (B,bins,num_env,3) env->centre vectors grouped by shell.
    to_env_sorted = torch.gather(to_env, 1, sort_idx.unsqueeze(-1).expand(-1, -1, 3))  # (B,E,3)
    binned_env = torch.gather(
        to_env_sorted, 1, flat_cols.unsqueeze(-1).expand(-1, -1, 3)).reshape(B, bins, num_env, 3)
    filler = torch.tensor([1.0, 0.0, 0.0], device=device)
    binned_env[empty_bins] = filler                               # shells with no env point
    # Map each binned slot back to its original env-point index. Needed to recover
    # the winning env point for the rotational-clearance direction (rot_n) below,
    # and for the optional min-angle visualization.
    sel_all = torch.gather(sort_idx, 1, flat_cols).reshape(B, bins, num_env)
    _prof_toc('angle_binning', device)






    # Smallest angle between any shape-surface direction and any env direction
    # (both measured from the placement centre), reduced to one value per
    # placement.  min(angle) == arccos(max cosine), so we carry a running MAX
    # COSINE and arccos it once at the end -- and we fuse the reduction into the
    # loop so the full (B,bins,num,num_env) pairwise tensor is never allocated.
    # The per-bin pairwise block (B,num,num_env) is still large, so we chunk it
    # over the env axis to bound peak memory.
    _prof_tic('angle_cosine', device)
    ANGLE_CHUNK = 128
    rad_vec = translated_rad_points - t.view(B, 1, 1, 3)          # (B,bins,num,3)
    rad_u = rad_vec / (rad_vec.norm(dim=-1, keepdim=True) + EPS)  # unit dirs (B,bins,num,3)
    env_u = binned_env / (binned_env.norm(dim=-1, keepdim=True) + EPS)  # (B,bins,num_env,3)

    best_cos = torch.full((B,), -1.0, device=device)             # running max cosine
    # Track which (bin, shape-dir, env-point) realizes the running max cosine. This
    # winning pair defines the rotational-clearance direction (rot_n) and is also
    # used for the optional min-angle visualization.
    best_bin   = torch.zeros(B, dtype=torch.long, device=device)
    best_j     = torch.zeros(B, dtype=torch.long, device=device)  # shape rad-pt idx
    best_eorig = torch.zeros(B, dtype=torch.long, device=device)  # original env idx
    for bi in range(bins):

        rb = rad_u[:, bi]                                         # (B,num,3)
        for s in range(0, num_env, ANGLE_CHUNK):
            eb = env_u[:, bi, s:s + ANGLE_CHUNK]                  # (B,c,3)
            cos = torch.einsum('bjk,bmk->bjm', rb, eb)           # (B,num,c)
            c = cos.shape[2]
            blk_cos, blk_arg = cos.reshape(B, -1).max(dim=1)     # (B,), (B,)
            # ignore empty (placeholder) shells for this bin
            blk_cos = torch.where(empty_bins[:, bi], best_cos, blk_cos)
            win = blk_cos > best_cos
            best_cos = torch.where(win, blk_cos, best_cos)
            j_idx = blk_arg // c                                 # shape-dir index
            m_idx = blk_arg %  c + s                             # binned env slot
            best_bin = torch.where(win, torch.full_like(best_bin, bi), best_bin)
            best_j   = torch.where(win, j_idx, best_j)
            # binned slot -> original env-point index
            eorig = torch.gather(sel_all[:, bi], 1, m_idx.unsqueeze(1)).squeeze(1)
            best_eorig = torch.where(win, eorig, best_eorig)

    min_angle = torch.arccos(best_cos.clamp(-1.0, 1.0))          # (B,)

    # ── Rotational-clearance direction (rot_n) ───────────────────────────────
    # The min angle is between the shape-surface direction u_s and the env
    # direction u_e (both from the placement centre). Rotating u_s towards u_e
    # reduces that angle the fastest about the axis (u_s x u_e); that unit axis,
    # placed in the rotation half of an SE(3) vector, is rot_n.
    b_ar = torch.arange(B, device=device)
    shape_pt = translated_rad_points[b_ar, best_bin, best_j]      # (B,3) world frame
    env_pt = env_points[best_eorig]                              # (B,3) world frame
    u_s = shape_pt - t                                          # (B,3)
    u_s = u_s / (u_s.norm(dim=1, keepdim=True) + EPS)
    u_e = env_pt - t                                            # (B,3)
    u_e = u_e / (u_e.norm(dim=1, keepdim=True) + EPS)
    rot_axis = torch.linalg.cross(u_s, u_e, dim=1)               # (B,3)
    if two_d:
        # Planar mode: only rotation about z exists, so keep that component and
        # drop the x/y tilt (pure residue of the shape's finite slab thickness).
        rot_axis[:, 0] = 0.0
        rot_axis[:, 1] = 0.0
    rot_axis = rot_axis / (rot_axis.norm(dim=1, keepdim=True) + EPS)
    zeros3 = torch.zeros(B, 3, device=device)
    rot_n = torch.cat([zeros3, rot_axis], dim=1)                 # (B,6)
    _prof_toc('angle_cosine', device)

    _prof_tic('collision', device)
    is_free = points_inside_tets(tets, env_kept)
    _prof_toc('collision', device)

    _prof_tic('clearance', device)
    dist, normal, n_ws = calculate_dist(faces, env_kept, configs[:, 0:3])
    if two_d:
        # Same projection for the clearance normal / contact direction: the z
        # translation and the x/y rotation components are not part of the planar
        # configuration space, so zero them and renormalize.
        normal[:, 2] = 0.0
        normal[:, 3] = 0.0
        normal[:, 4] = 0.0
        normal = normal / (torch.linalg.norm(normal, dim=1, keepdim=True) + EPS)
        n_ws[:, 2] = 0.0
        n_ws = n_ws / (torch.linalg.norm(n_ws, dim=1, keepdim=True) + EPS)
    _prof_toc('clearance', device)

    # ── Translational-clearance direction (trans_n) ──────────────────────────
    # n_ws is the unit direction the shape moves to *gain* clearance, so moving
    # the opposite way (-n_ws) reduces the distance to the obstacle the fastest.
    # Placed in the translation half of an SE(3) vector, that is trans_n.
    trans_n = torch.cat([-n_ws, zeros3], dim=1)                  # (B,6)

    if return_angle_pts:
        return (is_free.cpu(), dist.cpu(), min_angle.cpu(), normal.cpu(),
                trans_n.cpu(), rot_n.cpu(), shape_pt.cpu(), env_pt.cpu())
    return (is_free.cpu(), dist.cpu(), min_angle.cpu(), normal.cpu(),
            trans_n.cpu(), rot_n.cpu())


# ---------------------------------------------------------------------------
# Correlated-pair rejection sampling -- mirrors preprocess_dxf.generate_valid_pairs
#
#   x0 : drawn in the narrow band  offset < dist(x0) < margin
#   x1 : x0 + random SE(3) displacement, kept iff collision-free and dist > offset
# ---------------------------------------------------------------------------
def _sample_configs(n, hx, hy, hz, two_d=False):
    """Sample ``n`` uniformly-random SE(3) placements (position in bbox, random
    rotation as axis-angle with angle in [0, pi]).

    With ``two_d=True`` the samples are confined to the planar sub-space the
    ``--2d`` mode plans in: ``z`` is forced to 0 and the rotvec is ``(0, 0, rz)``
    with ``rz`` uniform in ``[-pi, pi)`` -- i.e. rotation about the z axis only.
    """
    c = torch.empty(n, 6)
    c[:, 0].uniform_(-hx, hx)
    c[:, 1].uniform_(-hy, hy)
    if two_d:
        c[:, 2] = 0.0
        c[:, 3] = 0.0
        c[:, 4] = 0.0
        c[:, 5].uniform_(-np.pi, np.pi)
        return c
    c[:, 2].uniform_(-hz, hz)
    axis = torch.randn(n, 3)
    axis = axis / (torch.linalg.norm(axis, dim=1, keepdim=True) + EPS)
    ang = torch.rand(n, 1) * np.pi
    c[:, 3:6] = axis * ang
    return c


def _min_center_dist(positions, env, device):
    """Smallest distance from each placement centre to any env point: (B,) CPU.

    An ``O(B*E)`` broad-phase with no per-triangle work.  Because the shape is
    contained in a ball of radius ``R`` about its centre, the true clearance is
    bounded by ``m - R <= clearance <= m + R`` (m = this value), which lets us
    cheaply reject placements that provably can't satisfy the clearance band
    *before* paying for the full evaluate_placements query.
    """
    p = positions.to(device)
    return torch.cdist(p, env).min(dim=1).values.cpu()


def generate_valid_pairs(number_pairs, tet_verts_local, face_verts_local,
                         env_points, half_extent,
                         margin, offset, rad_points, rad_bins, batch_size=256, device='cuda',
                         testing=False, yrot=False, two_d=False, track_angle_pts=False):
    """Sample ``number_pairs`` SE(3) placement pairs.

    Training mode (default): correlated pairs -- ``x0`` is drawn in the narrow
    band ``offset < clearance < margin`` and ``x1`` is a random SE(3) displacement
    away from it, kept iff collision-free with clearance > offset.

    Testing mode (``testing=True``): ``x0`` and ``x1`` are *independent*,
    uniformly-random placements, kept iff *both* are simply collision-free (no
    clearance band, no correlation).  Use this to generate start/goal test pairs.

    Y-rotation-only mode (``yrot=True``): the x- and z-rotation components of the
    rotvec are zeroed so the shape can only rotate about the y axis.

    Planar mode (``two_d=True``): every placement has ``z = 0`` and a rotvec of
    ``(0, 0, rz)``, so the sampled configuration space is the SE(2) slice
    ``(x, y, rz)`` embedded in the SE(3) layout (the unused coordinates stay 0).

    Returns
    -------
    pairs   : (number_pairs, 12)  raw [x0(6), x1(6)] (rotvec in radians)
    dists   : (number_pairs, 2)   raw clearance distances [d0, d1]
    normals : (number_pairs, 12)  SE(3) clearance normals [n0(6) | n1(6)]
    trans_normals : (number_pairs, 12)  unit translation directions that reduce the
        distance to the obstacle the most [trans_n0(6) | trans_n1(6)] (only the
        translation half is non-zero).
    rot_normals : (number_pairs, 12)  unit rotation axes that reduce the rotational
        clearance the most [rot_n0(6) | rot_n1(6)] (only the rotation half is non-zero).
    ang_shape, ang_env : (number_pairs, 3) each, or None unless ``track_angle_pts``.
        The shape-surface point and env point realizing x0's min-angle clearance.
    """
    # Hoist the static geometry to the device ONCE -- these never change across
    # batches, so re-transferring them inside every evaluate_placements call (twice
    # per batch, env three times each) is pure overhead. After this the inner
    # ``.to(device)`` calls are no-ops.
    env_t = torch.tensor(env_points, dtype=torch.float32).to(device)
    tet_verts_local = tet_verts_local.to(device)
    face_verts_local = face_verts_local.to(device)
    rad_points = rad_points.to(device)
    rad_bins = rad_bins.to(device)
    hx, hy, hz = float(half_extent[0]), float(half_extent[1]), float(half_extent[2])

    pairs = torch.zeros(number_pairs, 12)
    dists = torch.zeros(number_pairs, 2)
    angles = torch.zeros(number_pairs, 2)
    normals = torch.zeros(number_pairs, 12)
    trans_normals = torch.zeros(number_pairs, 12)
    rot_normals = torch.zeros(number_pairs, 12)
    # Winning min-angle pair for x0 (debug/visualization only).
    ang_shape = torch.zeros(number_pairs, 3) if track_angle_pts else None
    ang_env = torch.zeros(number_pairs, 3) if track_angle_pts else None
    count = 0
    sqrt6 = float(np.sqrt(6.0))
    two_pi = 2.0 * np.pi

    # Shape bounding radius -- drives the cheap broad-phase band gate (#1).
    R_shape = float(face_verts_local.reshape(-1, 3).norm(dim=1).max())

    def _eval0(cfg):
        """Full x0 evaluation; returns
        (free, dist, angle, normal, trans_n, rot_n, shp, envp), with shp/envp =
        None unless ``track_angle_pts``."""
        if track_angle_pts:
            return evaluate_placements(
                cfg, tet_verts_local, face_verts_local, env_t, rad_points, rad_bins,
                device, return_angle_pts=True, two_d=two_d)
        f, d, a, n, tn, rn = evaluate_placements(
            cfg, tet_verts_local, face_verts_local, env_t, rad_points, rad_bins, device,
            two_d=two_d)
        return f, d, a, n, tn, rn, None, None

    def _idx(mask, *tensors):
        """Index a bundle of tensors by ``mask`` (None passes through)."""
        return tuple(None if t is None else t[mask] for t in tensors)

    def _sample_disp(n):
        """Correlated SE(3) displacement for ``n`` placements (2pi-normalized)."""
        d = torch.rand(n, 6) - 0.5
        if yrot:
            d[:, 3] = 0
            d[:, 5] = 0
        if two_d:
            # keep x0 + delta inside the planar slice: no z travel, no x/y tilt.
            d[:, 2] = 0
            d[:, 3] = 0
            d[:, 4] = 0
        d = d / (torch.linalg.norm(d, dim=1, keepdim=True) + EPS)
        return d * (torch.rand(n, 1) * sqrt6)                   # normalized units

    while count < number_pairs:
        # ── Sample x0: position uniform in the env bbox; rotation uniform ──
        x0 = _sample_configs(batch_size, hx, hy, hz, two_d=two_d)
        if yrot:
            x0[:, 3] = 0
            x0[:, 5] = 0

        if testing:
            # ── x1: a second, *independent* uniformly-random placement ──
            x1 = _sample_configs(batch_size, hx, hy, hz, two_d=two_d)
            if yrot:
                x1[:, 3] = 0
                x1[:, 5] = 0
            in_bbox = (x1[:, 0].abs() <= hx) & (x1[:, 1].abs() <= hy) & (x1[:, 2].abs() <= hz)
            if int(in_bbox.sum()) == 0:
                continue
            x0, x1 = x0[in_bbox], x1[in_bbox]

            # #1 cheap gate: both endpoints require clearance > offset, so drop any
            # pair where an endpoint is provably too close (m + R < offset).
            m0 = _min_center_dist(x0[:, 0:3], env_t, device)
            m1 = _min_center_dist(x1[:, 0:3], env_t, device)
            feas = (m0 + R_shape >= offset) & (m1 + R_shape >= offset)
            if int(feas.sum()) == 0:
                continue
            x0, x1 = x0[feas], x1[feas]

            free0, dist0, angle0, normal0, trans0, rot0, shp0, envp0 = _eval0(x0)
            free1, dist1, angle1, normal1, trans1, rot1 = evaluate_placements(
                x1, tet_verts_local, face_verts_local, env_t, rad_points, rad_bins, device,
                two_d=two_d)
            keep = free0 & free1 & (dist0 > offset) & (dist1 > offset)
        else:
            # ── Training: x0 in narrow band, x1 a reachable displacement ──
            # #1 cheap gate on x0's band feasibility (clearance in (offset, margin)):
            # clearance >= m - R and <= m + R, so reject before the full query.
            m0 = _min_center_dist(x0[:, 0:3], env_t, device)
            feas0 = (m0 - R_shape <= margin) & (m0 + R_shape >= offset)
            if int(feas0.sum()) == 0:
                continue
            x0 = x0[feas0]

            free0, dist0, angle0, normal0, trans0, rot0, shp0, envp0 = _eval0(x0)
            keep_x0 = free0 & (dist0 > offset) & (dist0 < margin)   # narrow band
            if int(keep_x0.sum()) == 0:
                continue
            # #2 keep only x0 survivors, then build / evaluate x1 for *those* alone.
            x0, dist0, angle0, normal0, trans0, rot0, shp0, envp0 = _idx(
                keep_x0, x0, dist0, angle0, normal0, trans0, rot0, shp0, envp0)

            delta = _sample_disp(x0.shape[0])
            x1 = x0.clone()
            x1[:, 0:3] = x0[:, 0:3] + delta[:, 0:3]
            x1[:, 3:6] = wrap_rotvec(x0[:, 3:6] + delta[:, 3:6] * two_pi)

            in_bbox = (x1[:, 0].abs() <= hx) & (x1[:, 1].abs() <= hy) & (x1[:, 2].abs() <= hz)
            if int(in_bbox.sum()) == 0:
                continue
            x0, dist0, angle0, normal0, trans0, rot0, shp0, envp0 = _idx(
                in_bbox, x0, dist0, angle0, normal0, trans0, rot0, shp0, envp0)
            x1 = x1[in_bbox]

            # #1 cheap gate on x1 (only needs clearance > offset -> drop too-close).
            feas1 = _min_center_dist(x1[:, 0:3], env_t, device) + R_shape >= offset
            if int(feas1.sum()) == 0:
                continue
            x0, dist0, angle0, normal0, trans0, rot0, shp0, envp0 = _idx(
                feas1, x0, dist0, angle0, normal0, trans0, rot0, shp0, envp0)
            x1 = x1[feas1]

            free1, dist1, angle1, normal1, trans1, rot1 = evaluate_placements(
                x1, tet_verts_local, face_verts_local, env_t, rad_points, rad_bins, device,
                two_d=two_d)
            keep = free1 & (dist1 > offset)                        # x0 already in band

        nk = int(keep.sum())
        if nk == 0:
            continue

        take = min(nk, number_pairs - count)
        idx = torch.nonzero(keep, as_tuple=False).squeeze(1)[:take]

        pairs[count:count + take, 0:6] = x0[idx]
        pairs[count:count + take, 6:12] = x1[idx]
        dists[count:count + take, 0] = dist0[idx]
        dists[count:count + take, 1] = dist1[idx]
        angles[count:count + take, 0] = angle0[idx]
        angles[count:count + take, 1] = angle1[idx]
        normals[count:count + take, 0:6] = normal0[idx]
        normals[count:count + take, 6:12] = normal1[idx]
        trans_normals[count:count + take, 0:6] = trans0[idx]
        trans_normals[count:count + take, 6:12] = trans1[idx]
        rot_normals[count:count + take, 0:6] = rot0[idx]
        rot_normals[count:count + take, 6:12] = rot1[idx]
        if track_angle_pts:
            ang_shape[count:count + take] = shp0[idx]
            ang_env[count:count + take] = envp0[idx]
        count += take
        print('generated: {} / {}'.format(count, number_pairs))



    return (pairs, dists, angles, normals, trans_normals, rot_normals,
            ang_shape, ang_env)


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
                    wall_V=None, wall_F=None, cnt=50,
                    angle_shape_pts=None, angle_env_pts=None):
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
    angle_shape_pts, angle_env_pts : (N, 3) each, optional. When both are given,
        a red segment is drawn from the shape-surface point to the env point that
        realized each placement's min-angle clearance.
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

    # ── Min-angle clearance: red segment from shape point to env point ──
    if angle_shape_pts is not None and angle_env_pts is not None:
        seg_x, seg_y, seg_z = [], [], []
        for i in range(m):
            sp, ep = angle_shape_pts[i], angle_env_pts[i]
            seg_x += [sp[0], ep[0], None]   # None breaks the polyline between pairs
            seg_y += [sp[1], ep[1], None]
            seg_z += [sp[2], ep[2], None]
        traces.append(go.Scatter3d(
            x=seg_x, y=seg_y, z=seg_z, mode='lines+markers',
            line=dict(color='red', width=4),
            marker=dict(size=3, color='red'),
            name='min-angle pair',
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title='Sampled placements (shape mesh colored by clearance speed)',
        scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z',
                   aspectmode='data',
                   camera=dict(up=dict(x=0, y=1, z=0))))
    fig.write_html(save_path, include_plotlyjs='cdn')
    print('Saved sample visualization: ' + save_path)


def visual_speed_distribution(speed_pairs, save_path, label='speed', nbins=60):
    """Histogram of a per-placement (N, 2) value, written to an HTML file.

    Parameters
    ----------
    speed_pairs : (N, 2)  value for each endpoint [v0, v1]; both columns are
        pooled into one distribution over all 2N placements.
    save_path   : str     output HTML path.
    label       : str     name of the quantity (used in title / x-axis).
    nbins       : int     number of histogram bins.
    """
    speeds = np.asarray(speed_pairs).reshape(-1)        # pool x0 and x1
    speeds = speeds[np.isfinite(speeds)]

    fig = go.Figure(go.Histogram(x=speeds, nbinsx=nbins, marker_color='steelblue'))
    fig.update_layout(
        title='{} distribution (N={}, mean={:.3f}, min={:.3f}, max={:.3f})'.format(
            label,
            speeds.size,
            float(speeds.mean()) if speeds.size else float('nan'),
            float(speeds.min()) if speeds.size else float('nan'),
            float(speeds.max()) if speeds.size else float('nan')),
        xaxis_title=label, yaxis_title='count', bargap=0.02)
    fig.write_html(save_path, include_plotlyjs='cdn')
    print('Saved {} distribution: {}'.format(label, save_path))


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
    parser.add_argument('--margin', type=float, default=0.05,
                        help='Upper band: x0 must have clearance < margin; maps to speed=1.')
    parser.add_argument('--offset', type=float, default=0.001,
                        help='Lower band: x0 must have clearance > offset, and x1 '
                             '(paired goal) must also have clearance > offset. '
                             'Maps to the minimum speed value offset/margin.')
    parser.add_argument('--num_env_points', type=int, default=DEFAULT_ENV_POINTS,
                        help='Target number of points sampled on the env surface.')
    parser.add_argument('--num_radius_points', type=int, default=1000,
                        help='Number of shape-surface points binned by radius from origin.')
    parser.add_argument('--radius_bins', type=int, default=10,
                        help='Number of radial bins for the shape-surface points.')
    parser.add_argument('--tet_switches', default=DEFAULT_TET_SWITCHES,
                        help='tetgen switches used to tetrahedralize the shape.')
    parser.add_argument('--batch_size', type=int, default=3000,
                        help='Sampling batch size (each batch evaluates 2x configs).')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--visualize', action='store_true',
                        help='Also save a debug plot of the sampled placements.')
    parser.add_argument('--testing_data', action='store_true',
                        help='Generate TEST pairs: x0 and x1 are independent, '
                             'uniformly-random placements kept only if both are '
                             'collision-free (no narrow-band / clearance filtering).')
    parser.add_argument('--2d', dest='two_d', action='store_true',
                        help='Planar mode: flatten the environment onto z=0, '
                             'squash the shape to {} thick in z (so it still '
                             'tetrahedralizes), and sample only in the '
                             '(x, y, rz) slice -- z is pinned to 0 and the '
                             'rotation is about the z axis only.'.format(
                                 TWOD_SHAPE_THICKNESS))
    parser.add_argument('--yrot', action='store_true',
                        help='Restrict rotation to the y axis only: zero the x- '
                             'and z-rotation components of every sampled rotvec.')
    parser.add_argument('--profile', action='store_true',
                        help='Print per-block GPU timing of evaluate_placements '
                             '(adds cuda syncs, so leave off for production runs).')
    args = parser.parse_args()

    global _PROFILE
    _PROFILE = args.profile

    os.makedirs(args.out, exist_ok=True)

    # ── Environment: load, normalize to a unit box, sample its surface ──
    V_env, F_env, names_env = load_obj(args.env)
    if args.two_d:
        # Planar mode: collapse the whole environment onto the z=0 plane.  Done
        # before the bbox below so the normalization is driven by the x-y
        # footprint alone; the surface samples then all land on z=0 too, which is
        # exactly the plane the (flattened) shape sweeps through.
        V_env[:, 2] = 0.0
        print('[--2d] environment flattened onto z=0')
    # Bounding box uses *all* geometry (walls included) so the normalized frame
    # matches the true extent of the environment.
    bb_min = V_env.min(axis=0)
    bb_max = V_env.max(axis=0)
    center_env = 0.5 * (bb_min + bb_max)
    scale = float((bb_max - bb_min).max())            # largest extent -> 1.0
    V_env_n = (V_env - center_env) / scale


    sample_mask = np.array(['null' not in str(n).lower() for n in names_env])
    n_wall = int((~sample_mask).sum())
    F_sample = F_env[sample_mask]

    env_points = sample_surface_points(V_env_n, F_sample, args.num_env_points)
    env_points = env_points.astype(np.float32)

    ranges = (bb_max - bb_min) / scale
    half_extent = ranges * 0.5 - 0.01                 # small inset, like the 2-D code
    if args.two_d:
        # The z range collapsed to nothing; pin it to the single valid value so
        # the in-bbox test on sampled placements (|z| <= hz) accepts z == 0.
        half_extent[2] = 0.0

    # ── Shape: load, normalize into the same units, tetrahedralize ──
    V_sh, F_sh, _ = load_obj(args.shape)
    shape_center = 0.5 * (V_sh.min(axis=0) + V_sh.max(axis=0))
    V_sh_local = (V_sh - shape_center) / scale * args.shape_scale
    if args.two_d:
        # Squash the shape's z extent down to TWOD_SHAPE_THICKNESS: thin enough to
        # act as a flat 2-D footprint, thick enough to stay a solid that tetgen can
        # tetrahedralize.  It stays centred on z=0, i.e. on the flattened env plane.
        z = V_sh_local[:, 2]
        z_ext = float(z.max() - z.min())
        if z_ext > 1e-12:
            V_sh_local[:, 2] = (z - 0.5 * (z.max() + z.min())) * (TWOD_SHAPE_THICKNESS / z_ext)
        else:
            V_sh_local[:, 2] = 0.0
        print('[--2d] shape z extent {:.4f} -> {:.4f} (env-normalized units)'.format(
            z_ext, TWOD_SHAPE_THICKNESS))

    TV, TT, TF = tetrahedralize_shape(V_sh_local, F_sh, switches=args.tet_switches)
    tet_verts_local = torch.tensor(TV[TT], dtype=torch.float32)     # (K,4,3)
    face_verts_local = torch.tensor(TV[TF], dtype=torch.float32)    # (F,3,3)
    print('Tetrahedralized shape: {} tets, {} boundary triangles, {} verts'.format(
        TT.shape[0], TF.shape[0], TV.shape[0]))

    num_pairs = int(args.num_samples)
    mode = 'TESTING (independent collision-free pairs)' if args.testing_data \
        else 'TRAINING (narrow-band correlated pairs)'
    if args.two_d:
        mode += ' | PLANAR 2-D (z=0, rotation about z only)'
    print('Sampling {} (x0, x1) pairs [{}]   margin={}  offset={}  ...'.format(
        num_pairs, mode, args.margin, args.offset))
    t0 = time.time()

    radius_points, radius_bins = generate_radius_surface_points(
        V_sh_local, F_sh, args.num_radius_points, args.radius_bins)




    pairs, dists, angles, normals, trans_normals, rot_normals, ang_shape, ang_env = generate_valid_pairs(
        num_pairs, tet_verts_local, face_verts_local, env_points, half_extent,
        margin=args.margin, offset=args.offset,
        batch_size=args.batch_size, device=args.device,
        testing=args.testing_data, yrot=args.yrot, two_d=args.two_d,
        rad_points=radius_points, rad_bins=radius_bins,
        track_angle_pts=args.visualize)
    print('Sampling done in {:.1f}s'.format(time.time() - t0))
    _prof_report()

    pairs = pairs.cpu().numpy()

    #dists = np.max(0,dists-0.03)
    dists = dists.cpu().numpy()
    angles = angles.cpu().numpy()           # (N, 2)
    normals = normals.cpu().numpy()       # (N, 12)
    trans_normals = trans_normals.cpu().numpy()   # (N, 12)  [trans_n0 | trans_n1]
    rot_normals = rot_normals.cpu().numpy()       # (N, 12)  [rot_n0 | rot_n1]

    # Eikonal speed term: clearance clipped into [offset/margin, 1].

    speed_pairs = (np.clip(dists / args.margin,
                          a_min=args.offset / args.margin, a_max=1.0) + angles/np.pi)/2
    speed_angles = (angles/np.pi)
    speed_dists = np.clip(dists / args.margin, a_min=args.offset / args.margin, a_max=1.0)
    #speed_dists = np.clip(dists / args.margin, a_min=0.0001, a_max=1.0)

    if args.visualize:
        visual_training(pairs[:, 0:6].copy(), speed_pairs[:, 0], env_points,
                        V_sh_local, F_sh,
                        save_path=os.path.join(args.out, 'sampled_placements.html'),
                        wall_V=V_env_n, wall_F=F_env[~sample_mask],
                        angle_shape_pts=ang_shape.cpu().numpy(),
                        angle_env_pts=ang_env.cpu().numpy())
        visual_speed_distribution(
            speed_pairs, save_path=os.path.join(args.out, 'speed_distribution.html'),
            label='speed')
        visual_speed_distribution(
            speed_angles, save_path=os.path.join(args.out, 'speed_angles_distribution.html'),
            label='speed_angles')
        visual_speed_distribution(
            speed_dists, save_path=os.path.join(args.out, 'speed_dists_distribution.html'),
            label='speed_dists')

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
    np.save(os.path.join(args.out, 'speed_angles'), speed_angles)
    np.save(os.path.join(args.out, 'speed_dists'), speed_dists)
    np.save(os.path.join(args.out, 'normal'), normals)
    np.save(os.path.join(args.out, 'trans_n'), trans_normals)
    np.save(os.path.join(args.out, 'rot_n'), rot_normals)
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
        'two_d':       bool(args.two_d),
        'shape_z_thickness': float(TWOD_SHAPE_THICKNESS) if args.two_d else None,
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
    print('  trans_n.npy       ', trans_normals.shape)
    print('  rot_n.npy         ', rot_normals.shape)
    print('  env.npy           ', env_points.shape)


if __name__ == '__main__':
    main()
