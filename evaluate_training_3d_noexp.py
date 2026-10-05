"""Single-episode (non-batched) evaluation for the 3-D shape planner.


3-D (SE(3)) counterpart of ``evaluate_training.py``.  Same MPPI rollout, but the
configuration space is ``(x, y, z, rx, ry, rz)`` (dim = 6, rotvec stored
normalized by 2*pi -- exactly what ``dataprocessing/preprocess_obj.py`` writes).


Each episode is rendered interactively with `viser`.  Unlike the old plotly
version, the environment is drawn as its actual triangle *mesh* (obstacles solid
grey, walls translucent) rather than as sampled points.  The moving shape is
drawn as its actual transformed mesh at every trajectory waypoint (colored by
progress along the path), with the start pose in red and the goal pose in green.
The view is oriented y-up.


A viser server is launched on port 8080; episodes are browsed with a dropdown in
the viewer GUI.  Open it from a remote host with:


    http://<server-ip>:8080


(Summary scatter/histogram plots are still written as plotly HTML files under
./output_3d/ for offline inspection.)


Requires ``viser`` (``pip install viser``).


Run from the nested ntrl-demo root:


    python evaluate_training_3d.py
"""


import sys
sys.path.append('.')


import os
import json
import time
import argparse
from glob import glob
from timeit import default_timer as timer


import numpy as np
import torch
import igl
import viser
import plotly.graph_objects as go


from models.metric import model_train_metric as md
from dataprocessing.preprocess_obj import (
    load_obj, _rotvec_to_matrix_np, sample_surface_points)


# Sign convention for igl.signed_distance: the DEFAULT/WINDING_NUMBER types are
# unreliable in this binding, but FAST_WINDING_NUMBER returns negative-inside and
# is robust to per-face orientation (only requires a closed/watertight mesh).
_SDF_SIGN = igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
# Number of surface points sampled over the full environment mesh for collision.
ENV_COLLISION_POINTS = 50000
HTTP_PORT = 8080
DIM = 6


parser = argparse.ArgumentParser(
    description='Single-episode evaluation for the 3-D shape planner.')
parser.add_argument('--dataPath', default='./testing_data/3dshape/rectangle_env1',
                    help='Directory holding the test data (sampled_points.npy, '
                         'speed.npy, env.npy, meta.json) used as start/goal pairs.')
parser.add_argument('--out', default='./output_3d',
                    help='Output directory for the rendered HTML plots (also the '
                         'web root served over HTTP).')
args = parser.parse_args()


OUTPUT_DIR = args.out


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




def MPPI(womodel, XP, dim):
    """Run MPPI for a single start/goal pair.


    Returns the recorded waypoints, the iteration count, and a success flag.
    """
    found_path = False
    steps = 200
    sample_num = 50
    horizon = 5


    dP_prior = torch.zeros((1, dim)).cuda()


    point0 = []
    point0.append(XP[:, 0:dim].clone())


    iter = 0
    for iter in range(steps):
        XP_tmp = XP.clone()
        XP_tmp = XP_tmp.unsqueeze(0).repeat(sample_num, horizon, 1)


        dP = 0.015 * torch.normal(0, 1, size=(sample_num, 1, dim), dtype=torch.float32, device='cuda') \
            + 0.015 * torch.normal(0, 1, size=(sample_num, horizon, dim), dtype=torch.float32, device='cuda')
        dP = dP + 2 * dP_prior
        dP_norm = torch.norm(dP, dim=2, keepdim=True)
        dP = dP / (torch.clamp(dP_norm, min=0.015) / 0.015)
        dP_cumsum = torch.cumsum(dP, dim=1)
        XP_tmp[..., 0:dim] = XP_tmp[..., 0:dim] + dP_cumsum


        indices = [0, -1]
        cost = womodel.function.TravelTimes(XP_tmp[:, indices, :].reshape(-1, dim * 2))
        cost = cost.reshape(-1, 2)
        cost = 10 * cost[:, 0] + cost[:, 1]


        weight = torch.softmax(-50 * cost, dim=0)
        dP_prior = (weight @ dP[:, 0, :])


        XP[:, 0:dim] = dP_prior + XP[:, 0:dim]


        dis = torch.norm(XP[:, dim:dim * 2] - XP[:, 0:dim])
        point0.append(XP[:, 0:dim].clone())


        if dis < 0.01:
            found_path = True
            break


    point0.append(XP[:, dim:dim * 2].clone())
    return point0, iter, found_path




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




def _speed_color(s):
    """Map a model-predicted speed s in [0, 1] to an RGB tuple (viridis)."""
    return _progress_color(float(np.clip(s, 0.0, 1.0)))




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




def render_episode(server, ep, shape_V, shape_F, show_speed_labels=True):
    """Add the moving-shape sweep + start/goal poses for one episode.


    The shape mesh is drawn at every waypoint, colored by the model's PREDICTED
    SPEED at that config (viridis, 0=slow .. 1=fast) when ``ep['speeds']`` is
    present, otherwise by progress along the path.  When ``show_speed_labels`` is
    set, a small floating text label at each waypoint reports the numeric speed.
    The start pose is red and the goal pose is green.  Returns the list of scene
    handles so the caller can remove them before rendering the next episode.


    ``ep['waypoints']`` is a (T, 6) array of poses [x, y, z, rx, ry, rz] (rotvec
    in radians); ``ep['speeds']`` is a (T,) array of model-predicted speeds.
    """
    handles = []
    waypoints = ep['waypoints']
    speeds = ep.get('speeds')
    T = len(waypoints)
    for t in range(T):
        Vp = _placed_mesh(shape_V, waypoints[t])
        color = (_speed_color(speeds[t]) if speeds is not None
                 else _progress_color(t / max(T - 1, 1)))
        handles.append(server.scene.add_mesh_simple(
            f'/episode/traj/{t:04d}', vertices=Vp, faces=shape_F,
            color=color, opacity=0.5,
            flat_shading=True, side='double'))
        if speeds is not None and show_speed_labels:
            handles.append(server.scene.add_label(
                f'/episode/spd/{t:04d}', text=f'{speeds[t]:.3f}',
                position=tuple(Vp.mean(axis=0))))


    for cfg, col, nm in [(ep['begin_cfg'], (220, 30, 30), 'start'),
                         (ep['end_cfg'], (30, 180, 30), 'goal')]:
        if cfg is None:
            continue
        Vp = _placed_mesh(shape_V, cfg)
        handles.append(server.scene.add_mesh_simple(
            f'/episode/{nm}', vertices=Vp, faces=shape_F,
            color=col, opacity=0.9, flat_shading=True, side='double'))
    return handles




def launch_viser(episodes, shape_V, shape_F, env_V, obst_F, wall_F):
    """Serve an interactive viser scene with a dropdown to browse episodes."""
    server = viser.ViserServer(host='0.0.0.0', port=HTTP_PORT)
    server.scene.set_up_direction('+y')


    add_environment(server, env_V, obst_F, wall_F)


    options = [f"{ep['idx']:03d}_{ep['status']}" for ep in episodes]
    dropdown = server.gui.add_dropdown('Episode', options=options)
    speed_labels = server.gui.add_checkbox('Show speed labels', initial_value=True)
    speed_summary = server.gui.add_text('Predicted speed', initial_value='', disabled=True)


    current = []


    def show(i):
        for h in current:
            h.remove()
        current.clear()
        ep = episodes[i]
        current.extend(render_episode(
            server, ep, shape_V, shape_F, show_speed_labels=speed_labels.value))
        sp = ep.get('speeds')
        speed_summary.value = (
            'min {:.3f}  mean {:.3f}  max {:.3f}'.format(
                float(np.min(sp)), float(np.mean(sp)), float(np.max(sp)))
            if sp is not None and len(sp) else 'n/a')


    @dropdown.on_update
    def _(_):
        show(options.index(dropdown.value))


    @speed_labels.on_update
    def _(_):
        show(options.index(dropdown.value))


    if episodes:
        show(0)


    print(f"\nServing viser at http://0.0.0.0:{HTTP_PORT}  —  open this on your host PC")
    print("Use the 'Episode' dropdown to browse. Press Ctrl-C to stop.\n")
    while True:
        time.sleep(10)




# ──────────────────────────────────────────────────────────────────────────────
# Model & data setup
# ──────────────────────────────────────────────────────────────────────────────
modelPath = './Experiments/3dshape'
dataPath = args.dataPath


womodel = md.Model(modelPath, dataPath, DIM, [0.0] * DIM, device='cuda')


# Prefer the always-current latest.pt written every epoch by training; fall back
# to the most recent timestamped Model_Epoch_*.pt checkpoint.
latest = os.path.join(modelPath, 'latest.pt')
if os.path.exists(latest):
    pt = latest
else:
    ckpts = sorted(glob(os.path.join(modelPath, '*', 'Model_Epoch_*.pt')))
    if not ckpts:
        raise FileNotFoundError(
            f'No latest.pt and no checkpoints under {modelPath}/*/Model_Epoch_*.pt')
    pt = ckpts[-1]


#pt = './Experiments/3dshape/3dshape_06_11_22_49/Model_Epoch_03000_ValLoss_7.413567e-03.pt'
#pt = './pretrained/baseline_rectangle_env1.pt'


# 800k points diff4 model
#pt = './Experiments/3dshape/3dshape_06_12_23_00/Model_Epoch_05000_ValLoss_2.331354e-02.pt'


#pt = './Experiments/3dshape/3dshape_06_13_18_59/Model_Epoch_05000_ValLoss_1.170522e-02.pt'

#pt disabled for no-exp eval: use auto-detected top-level latest.pt
print(f'Loading checkpoint: {pt}')




womodel.load(pt)
womodel.network.eval()


arr = np.load(os.path.join(dataPath, 'sampled_points.npy'))       # (N, 12)
arr_speeds = np.load(os.path.join(dataPath, 'speed.npy'))         # (N, 2)


# Reconstruct the shape + wall meshes in the normalized frame from meta.json
# (written by preprocess_obj.py) so the visualization matches the sampling frame.
with open(os.path.join(dataPath, 'meta.json')) as f:
    meta = json.load(f)
env_scale = float(meta['env_scale'])
env_center = np.asarray(meta['env_center'], dtype=np.float64)
shape_scale = float(meta['shape_scale'])


V_sh, F_sh, _ = load_obj(meta['shape_obj'])
shape_center = 0.5 * (V_sh.min(axis=0) + V_sh.max(axis=0))
shape_V = np.ascontiguousarray((V_sh - shape_center) / env_scale * shape_scale,
                               dtype=np.float64)
shape_F = np.ascontiguousarray(F_sh, dtype=np.int64)
# Bounding radius of the shape about its local origin (used for collision broad phase).
shape_radius = float(np.linalg.norm(shape_V, axis=1).max())


V_env, F_env, names_env = load_obj(meta['env_obj'])
V_env_n = (V_env - env_center) / env_scale
wall_mask = np.array(['wall' in str(n).lower() for n in names_env])
wall_F = F_env[wall_mask]
obst_F = F_env[~wall_mask]


# Collision point cloud: env.npy is obstacles-only, so resample densely over the
# FULL environment mesh (obstacles + walls) -- everything counts as an obstacle.
env_collision_pts = np.ascontiguousarray(
    sample_surface_points(V_env_n, F_env, ENV_COLLISION_POINTS), dtype=np.float64)


test_list = []
test_list_speed = []
for i in range(100):
    curr = torch.tensor(arr[i]).cuda()
    test_list.append(curr)
    test_list_speed.append(min(arr_speeds[i]))


BASE = torch.zeros((1, 2 * DIM)).cuda()


# ──────────────────────────────────────────────────────────────────────────────
# Per-episode evaluation loop
# ──────────────────────────────────────────────────────────────────────────────
total = 0
n_collision = 0
n_no_conv = 0
n_lin_valid = 0       # cases where the straight-line start->goal path is collision-free
n_succ_lin_valid = 0  # planner successes among cases where linear interp is valid
n_succ_lin_invalid = 0  # planner successes among cases where linear interp is invalid
success_list = []
fail_list = []
test_list_speed_failed = []
episodes = []          # per-episode data for the interactive viser viewer


two_pi = 2 * np.pi
cnt2 = 0
for XP in test_list:
    # displacement goal - start over all 6 DOF (position part used for scatter)
    diff = (XP[DIM:2 * DIM] - XP[0:DIM]).detach().cpu().numpy().tolist()
    XP = (XP + BASE).reshape(1, 2 * DIM)


    start = timer()
    with torch.no_grad():
        point, iter, success = MPPI(womodel, XP.clone(), dim=DIM)
    end = timer()


    # Model-predicted speed at every waypoint.  Speed() runs network.out + the
    # gradient of tau w.r.t. the *current* config (the first DIM coords) and
    # returns 1/||grad tau|| -- i.e. what the model thinks the speed is at that
    # config (measured toward this episode's goal).  point holds the normalized
    # configs the network was trained on, so feed those directly (NOT the
    # rotvec-de-normalized waypoints).  Outside the no_grad block: Speed needs
    # autograd to differentiate tau.
    cur = torch.cat(point, dim=0)                      # (T, DIM) normalized
    goal = XP[0, DIM:2 * DIM].unsqueeze(0).expand(cur.shape[0], -1)
    speeds = womodel.function.Speed(
        torch.cat([cur, goal], dim=1)).detach().cpu().numpy()   # (T,)


    # Build (T, 6) waypoints; rotvec de-normalized to radians (stored / 2*pi).
    waypoints = np.zeros((len(point), DIM))
    for c, p in enumerate(point):
        waypoints[c, 0:3] = [p[0][0].item(), p[0][1].item(), p[0][2].item()]
        waypoints[c, 3:6] = [p[0][3].item() * two_pi,
                             p[0][4].item() * two_pi,
                             p[0][5].item() * two_pi]


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


    # Linear-interpolation baseline: is the straight line from start to goal a
    # collision-free path on its own?  (Independent of what the planner found.)
    lin_traj = linear_interp_traj(XP[0, 0:DIM], XP[0, DIM:2 * DIM])
    lin_collision = check_trajectory_collision(
        lin_traj, env_collision_pts, shape_V, shape_F, shape_radius)
    lin_valid = not lin_collision
    if lin_valid:
        n_lin_valid += 1


    ok = success and not collision
    status = 'success' if ok else 'fail'
    episodes.append({
        'idx': cnt2,
        'status': status,
        'waypoints': waypoints,
        'begin_cfg': begin_cfg,
        'end_cfg': end_cfg,
        'speeds': speeds,
    })


    print(
        f"[{cnt2:03d}] {'PASS' if ok else 'FAIL'}  "
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
        test_list_speed_failed.append(test_list_speed[cnt2])


    cnt2 += 1


n_total = len(test_list)
success_rate = total / n_total if n_total else 0.0
# Conditional planner success rates: how often the planner succeeds among cases
# where the trivial straight-line (linear interp) path is valid vs. invalid.
n_lin_invalid = n_total - n_lin_valid
lin_valid_success_rate = n_succ_lin_valid / n_lin_valid if n_lin_valid else 0.0
lin_invalid_success_rate = n_succ_lin_invalid / n_lin_invalid if n_lin_invalid else 0.0
print(f"total: {total} / {n_total}  ({success_rate:.1%})")
print(f"success | linear interp valid  : {n_succ_lin_valid} / {n_lin_valid}  "
      f"({lin_valid_success_rate:.1%})")
print(f"success | linear interp invalid: {n_succ_lin_invalid} / {n_lin_invalid}  "
      f"({lin_invalid_success_rate:.1%})")


# Success-rate summary written into the output directory.
summary_lines = [
    f"data_path                    : {dataPath}",
    f"model_path                   : {modelPath}",
    f"checkpoint                   : {pt}",
    f"episodes                     : {n_total}",
    f"successes                    : {total}",
    f"failures                     : {n_total - total}",
    f"  collision                  : {n_collision}",
    f"  no_converge                : {n_no_conv}",
    f"success_rate                 : {success_rate:.4f}  ({success_rate:.1%})",
    f"linear_interp_valid          : {n_lin_valid} / {n_total}",
    f"success_rate|lin_interp_valid: {lin_valid_success_rate:.4f}  "
    f"({lin_valid_success_rate:.1%})  [{n_succ_lin_valid}/{n_lin_valid}]",
    f"success_rate|lin_interp_fail : {lin_invalid_success_rate:.4f}  "
    f"({lin_invalid_success_rate:.1%})  [{n_succ_lin_invalid}/{n_lin_invalid}]",
]
with open(os.path.join(OUTPUT_DIR, 'success_rate.txt'), 'w') as f:
    f.write('\n'.join(summary_lines) + '\n')
print('Wrote ' + os.path.join(OUTPUT_DIR, 'success_rate.txt'))


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
# Interactive viser viewer — browse to http://<server-ip>:8080 from your host PC
# ──────────────────────────────────────────────────────────────────────────────
print(f"\nSummary plots saved to {OUTPUT_DIR}/")
launch_viser(episodes, shape_V, shape_F, V_env_n, obst_F, wall_F)

