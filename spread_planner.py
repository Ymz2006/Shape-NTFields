"""hlB planner + 'sampled-speed spread' mechanisms.  Evaluates success rate.

At every MPPI iteration the Bellman local term already evaluates
tau(cur -> cur + dP_i) for the S first-step samples, so each sample carries a
finite-difference speed estimate  s_i = |dP_i| / tau_i.  From those S values:

    s_mean, s_std, cv = s_std / s_mean            (cv is scale-free: 'relative')
    g = argmin_g sum_i (s_i - s_mean - g . dP_i)^2  (local speed gradient; verified
                                                    to point AWAY from obstacles)
    gate = clamp((cv - cv0) / (cv1 - cv0), 0, 1)   (near-obstacle belief from spread)

Mechanisms (all off = plain hlB):
  --steer-bias  b   executed step += b * step * gate * g/|g|
  --steer-cost  c   cost_i -= c * gate * (dP_i . g/|g|) / step     (reward samples aligned with g)
  --zlocal      w   cost_i += w * gate * (s_mean - s_i) / s_std     (relative, z-scored slowness)
  --rel-gate    k   cost_i += 1e3 * gate * [s_i < s_mean - k * s_std]
  --adapt-lw    a   local_w_eff = local_w * (1 + a * gate)
  --smean-gate      use a drop in s_mean (relative to the episode's running max)
                    instead of cv for the gate:  gate = clamp((1 - s_mean/s_ref - m0)/(m1 - m0), 0, 1)

    python _spread_planner.py --dataPath testing_data/3dshape/Ashape3d_env2 \
        --checkpoint ./Experiments/3dshape/3dshape_09_04_09_09/latest.pt \
        --device cuda:1 --cases 500 --steer-bias 0.5 --out results/spread/steer05
"""
import sys
sys.path.append('.')
import os, json, argparse, contextlib, io, time
import numpy as np
import torch
import igl
from models.metric import model_train_metric as md
from dataprocessing.preprocess_obj import load_obj, _rotvec_to_matrix_np, sample_surface_points

_SDF_SIGN = igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
DIM = 6
p = argparse.ArgumentParser()
p.add_argument('--dataPath', required=True)
p.add_argument('--checkpoint', required=True)
p.add_argument('--modelPath', default='./Experiments/3dshape')
p.add_argument('--device', default='cuda')
p.add_argument('--cases', type=int, default=500)
p.add_argument('--batch', type=int, default=250)
p.add_argument('--out', required=True)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--step', type=float, default=0.015)
p.add_argument('--steps', type=int, default=200)
p.add_argument('--local-weight', dest='local_w', type=float, default=0.03)
p.add_argument('--samples', type=int, default=50)
p.add_argument('--steer-bias', type=float, default=0.0)
p.add_argument('--steer-cost', type=float, default=0.0)
p.add_argument('--zlocal', type=float, default=0.0)
p.add_argument('--rel-gate', type=float, default=0.0)
p.add_argument('--adapt-lw', type=float, default=0.0)
p.add_argument('--cv0', type=float, default=0.10)
p.add_argument('--cv1', type=float, default=0.16)
p.add_argument('--smean-gate', action='store_true')
p.add_argument('--m0', type=float, default=0.15)
p.add_argument('--m1', type=float, default=0.40)
p.add_argument('--no-gate', action='store_true', help='gate = 1 always (mechanisms ungated)')
p.add_argument('--steer-trans', action='store_true', help='steer with the translational part of g only')
p.add_argument('--interp', type=int, default=0, help='also collision-check N interpolants between waypoints')
p.add_argument('--2d', dest='two_d', action='store_true',
               help="planar test set: sample only x, y, rz (auto-detected from meta.json's two_d)")
p.add_argument('--viser', action='store_true',
               help='after writing results, serve the interactive viser viewer from '
                    'evaluate_training_3d_batched.py (blocks; Ctrl-C to stop)')
p.add_argument('--viser-port', type=int, default=8081)
p.add_argument('--viser-top', type=int, default=20,
               help="size of the viewer's 'Longest N' tab (successful cases ranked by path length)")
a = p.parse_args()
# preprocess_obj.py --2d draws placements with z == 0 and a rotvec of (0, 0, rz),
# so only x, y and rz are free; sampling the other three plans through configs the
# field never saw (see PLANAR_FREE_DIMS in evaluate_training_3d_batched.py).
PLANAR_FREE_DIMS = (0, 1, 5)
FREE_MASK = None
torch.manual_seed(a.seed); np.random.seed(a.seed)
os.makedirs(a.out, exist_ok=True)
dev = a.device


def _candidate_cost(womodel, XP_tmp, cur, dP0, local_w, s_ref):
    """Returns cost (B,S), plus (s_mean, cv, g_unit, gate) from the first-step samples."""
    B, S, H, _ = XP_tmp.shape
    sel = XP_tmp                                         # all_horizon
    K = H
    f = womodel.function.TravelTimes(sel.reshape(-1, DIM * 2)).reshape(B, S, K)
    cand = sel[..., 0:DIM]
    src = cur[:, None, None, :].expand_as(cand)
    local = womodel.function.TravelTimes(
        torch.cat([src, cand], dim=3).reshape(-1, DIM * 2)).reshape(B, S, K)
    seg = torch.norm(cand - src, dim=3).clamp(min=1e-6)

    
    slow = local / seg                                   # (B,S,K)
    # ── spread statistics from the first-step samples ──
    s = 1.0 / slow[:, :, 0]                              # (B,S) sampled speeds
    s_mean = s.mean(1); s_std = s.std(1).clamp(min=1e-6)
    cv = s_std / s_mean
    y = (s - s_mean[:, None])[..., None]
    A = dP0                                              # (B,S,DIM)
    AtA = A.transpose(1, 2) @ A + 1e-8 * torch.eye(DIM, device=dev)
    g = torch.linalg.solve(AtA, A.transpose(1, 2) @ y)[..., 0]
    if a.steer_trans:
        g = g.clone(); g[:, 3:] = 0
    if FREE_MASK is not None:
        g = g * FREE_MASK
    g_unit = g / torch.norm(g, dim=1, keepdim=True).clamp(min=1e-9)
    if a.no_gate:
        gate = torch.ones(B, device=dev)
    elif a.smean_gate:
        drop = 1.0 - s_mean / s_ref.clamp(min=1e-6)
        gate = ((drop - a.m0) / (a.m1 - a.m0)).clamp(0, 1)
    else:
        gate = ((cv - a.cv0) / (a.cv1 - a.cv0)).clamp(0, 1)
    lw = local_w * (1.0 + a.adapt_lw * gate)             # (B,)
    f = f + lw[:, None, None] * slow
    cost = 10 * f[:, :, 0] + f[:, :, 1:].mean(dim=2)
    if a.steer_cost > 0:
        align = (dP0 * g_unit[:, None, :]).sum(2) / a.step        # (B,S) in [-1,1]
        cost = cost - a.steer_cost * gate[:, None] * align
    if a.zlocal > 0:
        z = (s_mean[:, None] - s) / s_std[:, None]
        cost = cost + a.zlocal * gate[:, None] * z
    if a.rel_gate > 0:
        bad = (s < (s_mean - a.rel_gate * s_std)[:, None]).float()
        cost = cost + 1e3 * gate[:, None] * bad
    return cost, s_mean, cv, g_unit, gate


def rollout(womodel, XP):
    B = XP.shape[0]; S = a.samples; H = 5; step = a.step
    XP = XP.clone()
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    chain_a = [XP[:, 0:DIM].clone()]; chain_b = [XP[:, DIM:].clone()]
    n_a = torch.ones(B, dtype=torch.long); n_b = torch.ones(B, dtype=torch.long)
    moving_a = True
    s_ref = torch.zeros(B, device=dev)                   # running max of s_mean (per episode)
    gate_log = []
    for it in range(a.steps):
        XP_tmp = XP.clone()[:, None, None, :].repeat(1, S, H, 1)
        dP = step * torch.randn(B, S, 1, DIM, device=dev) + step * torch.randn(B, S, H, DIM, device=dev)
        if FREE_MASK is not None:                        # clamp AFTER the projection, so the
            dP = dP * FREE_MASK                          # stride budget is spent in x-y-rz only
        n = torch.norm(dP, dim=3, keepdim=True)
        dP = dP / (torch.clamp(n, min=step) / step)
        XP_tmp[..., 0:DIM] = XP_tmp[..., 0:DIM] + torch.cumsum(dP, dim=2)
        cost, s_mean, cv, g_unit, gate = _candidate_cost(
            womodel, XP_tmp, XP[:, 0:DIM], dP[:, :, 0, :], a.local_w, s_ref)
        s_ref = torch.maximum(s_ref, s_mean)
        gate_log.append(gate.mean().item())
        weight = torch.softmax(-50 * cost, dim=1)
        step_prior = torch.bmm(weight.unsqueeze(1), dP[:, :, 0, :]).squeeze(1)
        if a.steer_bias > 0:
            step_prior = step_prior + a.steer_bias * step * gate[:, None] * g_unit
        live = ~done
        XP[:, 0:DIM] = XP[:, 0:DIM] + step_prior * live.unsqueeze(1)
        dis = torch.norm(XP[:, DIM:] - XP[:, 0:DIM], dim=1)
        lc = live.cpu().long()
        if moving_a:
            chain_a.append(XP[:, 0:DIM].clone()); n_a += lc
        else:
            chain_b.append(XP[:, 0:DIM].clone()); n_b += lc
        done = done | (dis < 0.01)
        if bool(done.all()):
            break
        XP = torch.cat([XP[:, DIM:], XP[:, 0:DIM]], dim=1)
        moving_a = not moving_a
    paths = []
    for b in range(B):
        cf = [chain_a[k][b] for k in range(int(n_a[b]))] + [chain_b[k][b] for k in reversed(range(int(n_b[b])))]
        paths.append(torch.stack(cf).cpu().numpy())
    return paths, done.cpu().numpy().tolist(), float(np.mean(gate_log))


# ── geometry ──
with open(os.path.join(a.dataPath, 'meta.json')) as fh:
    meta = json.load(fh)
env_scale = float(meta['env_scale']); env_center = np.asarray(meta['env_center'])
if a.two_d or meta.get('two_d'):
    FREE_MASK = torch.zeros(DIM, device=dev)
    FREE_MASK[list(PLANAR_FREE_DIMS)] = 1.0
    print('[--2d] planar rollout: free dims', PLANAR_FREE_DIMS)
_res = lambda q: q if os.path.exists(q) else os.path.join('./datasets/3dshape', os.path.basename(q))
with contextlib.redirect_stdout(io.StringIO()):
    V_sh, F_sh, _ = load_obj(_res(meta['shape_obj'])); V_env, F_env, names_env = load_obj(_res(meta['env_obj']))
shape_V = np.ascontiguousarray((V_sh - 0.5 * (V_sh.min(0) + V_sh.max(0))) / env_scale)
shape_F = np.ascontiguousarray(F_sh, dtype=np.int64)
shape_radius = float(np.linalg.norm(shape_V, axis=1).max())
np.random.seed(0)
env_pts = np.ascontiguousarray(sample_surface_points((V_env - env_center) / env_scale, F_env, 50000))
np.random.seed(a.seed)


def clearance(c):
    t = c[:3]; m = np.linalg.norm(env_pts - t, axis=1) <= shape_radius
    if not m.any():
        return 1.0
    R = _rotvec_to_matrix_np(c[3:6] * 2 * np.pi)
    S = igl.signed_distance(np.ascontiguousarray((env_pts[m] - t) @ R), shape_V, shape_F, _SDF_SIGN)[0]
    return float(S.min())


womodel = md.Model(a.modelPath, a.dataPath, DIM, [0.0] * DIM, device=dev)
womodel.load(a.checkpoint); womodel.network.eval()
arr = np.load(os.path.join(a.dataPath, 'sampled_points.npy'))[:a.cases]
valid = np.array([clearance(r[:DIM]) > 0 and clearance(r[DIM:]) > 0 for r in arr])
ids = np.nonzero(valid)[0]
X = torch.tensor(arr, dtype=torch.float32, device=dev)

t0 = time.time(); paths, conv, gates = [], [], []
with torch.no_grad():
    for s in range(0, len(ids), a.batch):
        b = ids[s:s + a.batch]
        pl, cv_, gm = rollout(womodel, X[b].clone())
        paths += pl; conv += cv_; gates.append(gm)
print(f'rollouts done in {time.time() - t0:.0f}s   mean gate {np.mean(gates):.3f}')

n_ok = n_coll = n_nc = 0; depths = []; recs = []; status = []
for k, P in enumerate(paths):
    cl = np.array([clearance(c) for c in P])
    coll = bool((cl < 0).any())
    if a.interp > 0 and not coll:
        for j in range(len(P) - 1):
            if any(clearance((1 - t) * P[j] + t * P[j + 1]) < 0 for t in np.linspace(0, 1, a.interp + 2)[1:-1]):
                coll = True; break
    ok = conv[k] and not coll
    n_ok += ok; n_coll += coll; n_nc += (not conv[k])
    if coll:
        depths.append(-cl.min())
    recs.append(dict(case=int(ids[k]), ok=ok, collided=coll, converged=bool(conv[k]), L=len(P),
                     max_depth=float(-cl.min()) if coll else 0.0))
    status.append('success' if ok else ('collision' if coll else 'no_converge'))
n = len(paths)
print(f'=== {vars(a)}')
print(f'cases {n}  success {n_ok} ({100 * n_ok / n:.1f}%)  collision {n_coll}  no_converge {n_nc}  '
      f'depth p50 {np.median(depths) if depths else 0:.4f}  path len p50 {np.median([r["L"] for r in recs]):.0f}')
json.dump(dict(args=vars(a), n=n, n_ok=n_ok, n_coll=n_coll, n_nc=n_nc, records=recs),
          open(os.path.join(a.out, 'records.json'), 'w'))
np.save(os.path.join(a.out, 'paths.npy'), np.array(paths, dtype=object), allow_pickle=True)


# ──────────────────────────────────────────────────────────────────────────────
# Interactive viser viewer (--viser) -- same scene as evaluate_training_3d_batched.py:
# environment mesh (grey obstacles, translucent walls), the shape swept along the
# path colored by progress (viridis, dark=start .. bright=goal), start pose red,
# goal pose green.  The spread planner produces ONE path per case, so instead of
# the batched script's 'Mode' dropdown the tabs split cases by outcome.
# ──────────────────────────────────────────────────────────────────────────────
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


def _to_waypoints(P):
    """(T, 6) normalized-frame path -> (T, 6) pose array with the rotvec in radians."""
    out = np.array(P, dtype=np.float64).reshape(-1, DIM).copy()
    out[:, 3:6] *= 2 * np.pi
    return out


def add_environment(server, env_V, obst_F, wall_F):
    """Draw the environment as its actual triangle mesh (static scene).

    Obstacles are solid grey; walls are translucent light-blue.
    """
    if len(obst_F) > 0:
        server.scene.add_mesh_simple(
            '/env/obstacles', vertices=env_V, faces=obst_F,
            color=(150, 150, 150), opacity=1.0, flat_shading=True, side='double')
    if len(wall_F) > 0:
        server.scene.add_mesh_simple(
            '/env/walls', vertices=env_V, faces=wall_F,
            color=(173, 216, 230), opacity=0.15, flat_shading=True, side='double')


def render_episode(server, ep, shape_V, shape_F):
    """Add the moving-shape sweep + start/goal poses for one episode.

    The shape mesh is drawn at every waypoint, colored by PROGRESS along the path
    (viridis, dark=start .. bright=goal).  The start pose is red and the goal pose
    is green.  Returns the list of scene handles so the caller can remove them
    before rendering the next episode.
    """
    handles = []
    waypoints = ep['waypoints']
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


def launch_viser(episodes, shape_V, shape_F, env_V, obst_F, wall_F, port, top=20):
    """Serve an interactive viser scene with tabs + a dropdown to browse episodes."""
    import viser
    server = viser.ViserServer(host='0.0.0.0', port=port)
    server.scene.set_up_direction('+y')

    add_environment(server, env_V, obst_F, wall_F)

    detail = server.gui.add_text('Outcome', initial_value='', disabled=True)

    # -- Case sets, one per tab --
    longest = sorted((k for k, ep in enumerate(episodes) if ep['status'] == 'success'),
                     key=lambda k: -episodes[k]['L'])[:top]
    TAB_SPECS = [
        ('All', lambda ep: True),
        ('Success', lambda ep: ep['status'] == 'success'),
        (f'Longest {top}', longest),
        ('Collision', lambda ep: ep['status'] == 'collision'),
        ('No converge', lambda ep: ep['status'] == 'no_converge'),
    ]
    NONE = '(none)'          # placeholder for an empty tab: viser needs an option

    current = []

    def show(i):
        for h in current:
            h.remove()
        current.clear()
        ep = episodes[i]
        current.extend(render_episode(server, ep, shape_V, shape_F))
        detail.value = '{} | converged {} | collided {} | L {} | max_depth {:.4f}'.format(
            ep['status'], ep['converged'], ep['collided'], ep['L'], ep['max_depth'])

    # Episode labels carry the outcome, so every dropdown doubles as the list of
    # that tab's cases with their status.
    def labels_for(subset):
        return [f"{episodes[i]['idx']:03d}_{episodes[i]['status']}_L{episodes[i]['L']}"
                for i in subset] or [NONE]

    # viser grew add_tab_group early on, but fall back to plain stacked dropdowns
    # (one per case set) if this install predates it.
    has_tabs = hasattr(server.gui, 'add_tab_group')
    tab_group = server.gui.add_tab_group() if has_tabs else None

    for title, pred in TAB_SPECS:
        # a tab is either a predicate over episodes or an explicit ordered index list
        subset = pred if isinstance(pred, list) else [k for k, ep in enumerate(episodes) if pred(ep)]
        label = f'{title} ({len(subset)})'
        if has_tabs:
            with tab_group.add_tab(label):
                dd = server.gui.add_dropdown('Episode', options=labels_for(subset))
        else:
            dd = server.gui.add_dropdown(f'Episode [{label}]', options=labels_for(subset))

        @dd.on_update
        def _(_, subset=subset, dd=dd):
            if not subset or dd.value == NONE:
                return
            show(subset[list(dd.options).index(dd.value)])

    if episodes:
        show(0)

    print(f"\nServing viser at http://0.0.0.0:{port}  —  open this on your host PC")
    print(f"Use the tabs to pick a case set (All / Success / Longest {top} / Collision / "
          "No converge) and 'Episode' to browse; labels are <case>_<status>_L<waypoints>.")
    print("Picking a tab does not move the scene on its own (viser exposes no "
          "tab-change callback) -- choose an episode from that tab's 'Episode' "
          "dropdown to display it.")
    print("Path is colored by progress (dark=start .. bright=goal); start pose red, "
          "goal green.")
    print("Press Ctrl-C to stop.\n")
    while True:
        time.sleep(10)


if a.viser:
    env_V_n = (V_env - env_center) / env_scale
    wall_mask = np.array(['wall' in str(n).lower() for n in names_env])
    episodes = []
    for k, P in enumerate(paths):
        wp = _to_waypoints(P)
        goal = arr[ids[k], DIM:].astype(np.float64).copy(); goal[3:6] *= 2 * np.pi
        episodes.append(dict(idx=int(ids[k]), status=status[k], converged=recs[k]['converged'],
                             collided=recs[k]['collided'], L=recs[k]['L'],
                             max_depth=recs[k]['max_depth'],
                             waypoints=wp, begin_cfg=wp[0].copy(), end_cfg=goal))
    launch_viser(episodes, shape_V, shape_F, env_V_n, F_env[~wall_mask], F_env[wall_mask],
                 a.viser_port, top=a.viser_top)
