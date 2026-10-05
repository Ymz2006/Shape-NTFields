# testing_data_1k_complete

1000 start/goal pairs per cell that are collision-free **and** known to be
connectable: every pair ships with an RRT-Connect path that was actually found.
`testing_data/` guarantees only the first half of that, so a success rate
measured on it has an unknown ceiling; these sets have a demonstrated one.

Layout matches `testing_data/3dshape/<cell>/` exactly -- same eight `.npy` files,
same dtypes and column counts, same `meta.json` keys -- so any `--dataPath` that
accepts one accepts the other.  Verified end to end by running
`evaluate_training_3d_batched.py --2d` against a generated cell with an existing
checkpoint.

56 cells: 28 in SE(3) (7 shapes x `env1`..`env4`; the four `Tshape3d` cells were
added 2026-09-12) and 28 planar (7 shapes x `2denv1`..`2denv4`), plus the
Gibson cell `Lcouch_Corozal` (added 2026-09-13), which has **500** pairs
instead of 1000 -- the Corozal scan is 85k triangles and its collision checks
are correspondingly slower.  Its dataset cloud (`env.npy`) is the 39 844 mesh
vertices rather than an area-weighted sample: `sample_surface_points` seeds
with every referenced vertex and the mesh has more of those than
`--num_env_points`, so the hurdle-1 clearance test is weaker there and 19% of
its candidates were caught by the 50k-point audit (hurdle 2) instead of 0-0.3%
in the synthetic cells.  The written pairs are audited the same way as every
other cell.

## How they were built

```
python dataprocessing/generate_testing_data_rrt.py [--2d] \
    --shape  datasets/3dshape/<shape>.obj \
    --env    datasets/3dshape/<env>.obj \
    --offset 0.02 \
    --out    testing_data_1k_complete/3dshape/<cell>
```

Sweep driver `_run_gen_testing_1k.sh` (resumable via `.done` markers), summary
`_summarize_testing_1k.py`, audit/repair `_repair_testing_1k.py`.  Needs a
container with **both** torch and the OMPL bindings -- `ntrl_mpnet` has both, the
pytorchserver image does not.

Every pair cleared three hurdles:

1. **Sampled clearance** -- both placements collision-free with clearance above
   `--offset` (0.02 here), via `preprocess_obj.generate_valid_pairs(testing=True)`.
2. **Re-checked collision** -- both re-tested against three independently drawn
   50 000-point clouds, the density `evaluate_training_3d_batched.py` resamples at
   evaluation time.
3. **Demonstrated path** -- RRT-Connect returned an exact path within 180 s, and
   that path was re-walked waypoint by waypoint before the pair was accepted.

The collision model and planner are imported from
`baselines/baseline_ompl/rrt_connect_eval.py`, so "a path exists" means exactly
what it means in the RRT-Connect baseline table.

## Known limitation

Collision-free is defined against a *sampled* point cloud -- a Monte Carlo
stand-in for a true mesh-mesh test.  A placement that grazes an obstacle can pass
one draw and fail another.

Measured: one independent 50k redraw over all 104 000 placements flagged 3
(0.003%); three further 3-cloud audit passes replaced 9 more pairs, finding 4,
then 3, then 2 -- a different marginal subset each time, so the sequence reduces
the count without converging to zero.  Expect roughly **0.008% of placements to be
draw-dependent**.

Root cause: `generate_valid_pairs` measures clearance against the 10 000-point
dataset cloud, whose point spacing is the same order as `offset = 0.02`, so the
clearance estimate carries error comparable to the threshold it is tested against.
Eliminating rather than reducing it means regenerating with
`--num_env_points 50000` or higher (and a smaller `--batch_size` to fit).

The pre-existing `testing_data/` sets have the same property -- the RRT-Connect
baseline found 0-2 invalid endpoints per config on them -- so these sets are no
worse on this axis and measurably better, having been audited and repaired.

Cells whose pairs were replaced carry `repaired_rows` in their `meta.json`; every
cell carries the full audit trail in `rrt_verification.json`.

## Acceptance rates

The interesting column is **Accept**: the fraction of uniformly-sampled
collision-free pairs that RRT-Connect could connect within 180 s.  That is a
property of the environment, not of the generator.  `2denv2` is where it falls
apart -- in `Tshape3d_2denv2`, 838 of 1838 candidate pairs had no path.

Note this filtering makes these sets a **different population** from
`testing_data/`: the hardest pairs have been removed, so success rates measured
here will read higher than anything in `experiments_ours.md`.  The two are not
the same test measured twice.

| Cell | Pairs | Planned | Accept | Collision rej | Unsolved | Solve mean/max (s) | Wall (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rectangle_env1 | 1000 | 1000 | 100.0% | 0 | 0 | 0.41 / 2.5 | 60 |
| Lshape3d_env1 | 1000 | 1000 | 100.0% | 0 | 0 | 0.79 / 8.8 | 106 |
| Fshape3d_env1 | 1000 | 1000 | 100.0% | 0 | 0 | 1.42 / 18.0 | 157 |
| Ashape3d_env1 | 1000 | 1001 | 99.9% | 0 | 1 | 3.56 / 25.4 | 374 |
| Vshape3d_env1 | 1000 | 1000 | 100.0% | 0 | 0 | 2.27 / 88.5 | 230 |
| 4shape3d_env1 | 1000 | 1000 | 100.0% | 0 | 0 | 1.70 / 11.7 | 188 |
| rectangle_env2 | 1000 | 1001 | 99.9% | 1 | 0 | 0.24 / 1.7 | 47 |
| Lshape3d_env2 | 1000 | 1002 | 99.8% | 2 | 0 | 0.42 / 2.6 | 74 |
| Fshape3d_env2 | 1000 | 1002 | 99.8% | 2 | 0 | 0.89 / 5.5 | 135 |
| Ashape3d_env2 | 1000 | 1000 | 100.0% | 0 | 0 | 2.48 / 12.7 | 285 |
| Vshape3d_env2 | 1000 | 1001 | 99.9% | 1 | 0 | 1.52 / 10.1 | 180 |
| 4shape3d_env2 | 1000 | 1002 | 99.8% | 2 | 0 | 1.08 / 5.3 | 151 |
| rectangle_env3 | 1000 | 1002 | 99.8% | 2 | 0 | 0.13 / 2.3 | 45 |
| Lshape3d_env3 | 1000 | 1001 | 99.9% | 1 | 0 | 0.22 / 2.4 | 74 |
| Fshape3d_env3 | 1000 | 1001 | 99.9% | 1 | 0 | 0.38 / 5.2 | 115 |
| Ashape3d_env3 | 1000 | 1000 | 100.0% | 0 | 0 | 0.89 / 9.5 | 219 |
| Vshape3d_env3 | 1000 | 1003 | 99.7% | 3 | 0 | 0.58 / 6.5 | 138 |
| 4shape3d_env3 | 1000 | 1001 | 99.9% | 1 | 0 | 0.40 / 3.8 | 118 |
| rectangle_env4 | 1000 | 1000 | 100.0% | 0 | 0 | 0.12 / 1.0 | 45 |
| Lshape3d_env4 | 1000 | 1000 | 100.0% | 0 | 0 | 0.20 / 2.0 | 78 |
| Fshape3d_env4 | 1000 | 1000 | 100.0% | 0 | 0 | 0.35 / 4.9 | 126 |
| Ashape3d_env4 | 1000 | 1001 | 99.9% | 1 | 0 | 0.84 / 6.6 | 237 |
| Vshape3d_env4 | 1000 | 1000 | 100.0% | 0 | 0 | 0.55 / 4.3 | 150 |
| 4shape3d_env4 | 1000 | 1001 | 99.9% | 1 | 0 | 0.39 / 4.0 | 132 |
| rectangle_2denv1 | 1000 | 1000 | 100.0% | 0 | 0 | 0.82 / 12.0 | 77 |
| Lshape3d_2denv1 | 1000 | 1000 | 100.0% | 0 | 0 | 1.51 / 29.9 | 127 |
| Fshape3d_2denv1 | 1000 | 1000 | 100.0% | 0 | 0 | 2.76 / 49.8 | 208 |
| Ashape3d_2denv1 | 1000 | 1019 | 98.1% | 0 | 19 | 7.08 / 176.3 | 667 |
| Vshape3d_2denv1 | 1000 | 1018 | 98.2% | 0 | 18 | 5.99 / 157.6 | 558 |
| 4shape3d_2denv1 | 1000 | 1000 | 100.0% | 0 | 0 | 3.44 / 53.3 | 231 |
| Tshape3d_2denv1 | 1000 | 1064 | 94.0% | 0 | 64 | 7.05 / 177.8 | 991 |
| rectangle_2denv2 | 1000 | 1000 | 100.0% | 0 | 0 | 1.76 / 17.2 | 124 |
| Lshape3d_2denv2 | 1000 | 1000 | 100.0% | 0 | 0 | 4.27 / 37.9 | 252 |
| Fshape3d_2denv2 | 1000 | 1000 | 100.0% | 0 | 0 | 9.26 / 83.4 | 517 |
| Ashape3d_2denv2 | 1000 | 1560 | 64.1% | 0 | 560 | 27.13 / 179.5 | 5603 |
| Vshape3d_2denv2 | 1000 | 1422 | 70.3% | 0 | 422 | 28.49 / 177.3 | 4552 |
| 4shape3d_2denv2 | 1000 | 1000 | 100.0% | 0 | 0 | 12.82 / 116.6 | 664 |
| Tshape3d_2denv2 | 1000 | 1838 | 54.4% | 0 | 838 | 16.00 / 179.8 | 7231 |
| rectangle_2denv3 | 1000 | 1000 | 100.0% | 0 | 0 | 0.16 / 1.2 | 38 |
| Lshape3d_2denv3 | 1000 | 1000 | 100.0% | 0 | 0 | 0.19 / 2.4 | 40 |
| Fshape3d_2denv3 | 1000 | 1000 | 100.0% | 0 | 0 | 0.28 / 3.0 | 47 |
| Ashape3d_2denv3 | 1000 | 1000 | 100.0% | 0 | 0 | 0.69 / 10.1 | 76 |
| Vshape3d_2denv3 | 1000 | 1000 | 100.0% | 0 | 0 | 0.55 / 5.4 | 65 |
| 4shape3d_2denv3 | 1000 | 1000 | 100.0% | 0 | 0 | 0.32 / 5.5 | 50 |
| Tshape3d_2denv3 | 1000 | 1000 | 100.0% | 0 | 0 | 0.95 / 10.2 | 93 |
| rectangle_2denv4 | 1000 | 1000 | 100.0% | 0 | 0 | 0.16 / 1.7 | 37 |
| Lshape3d_2denv4 | 1000 | 1000 | 100.0% | 0 | 0 | 0.20 / 4.1 | 46 |
| Fshape3d_2denv4 | 1000 | 1000 | 100.0% | 0 | 0 | 0.30 / 4.2 | 57 |
| Ashape3d_2denv4 | 1000 | 1000 | 100.0% | 0 | 0 | 0.82 / 11.8 | 96 |
| Vshape3d_2denv4 | 1000 | 1000 | 100.0% | 0 | 0 | 0.65 / 26.4 | 74 |
| 4shape3d_2denv4 | 1000 | 1000 | 100.0% | 0 | 0 | 0.34 / 4.5 | 58 |
| Tshape3d_2denv4 | 1000 | 1000 | 100.0% | 0 | 0 | 1.20 / 43.1 | 112 |
| Lcouch_Corozal | 500 | 755 | 66.2% | 145 | 110 | 0.66 / 7.6 | 1138 |
