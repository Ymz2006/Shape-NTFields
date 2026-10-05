# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A neural Eikonal-equation path planner. A network learns a travel-time field `tau(x0, x1)`
between pairs of configurations; its spatial gradient gives the local speed (clearance), and
following `-grad(tau)` yields collision-free paths. The same architecture is reused across
several configuration spaces, distinguished only by `dim`:

- **2-D shape** (`dim=3`): rigid body in SE(2) — `(x, y, theta)`.
- **3-D shape** (`dim=6`): rigid body in SE(3) — `(x, y, z, rx, ry, rz)`, where `(rx,ry,rz)` is a rotation vector (axis-angle).
- **arm** (`dim=6`): UR5 joint space, handled by the parallel `models/metric_arm/` package.
- **maze / gibson**: 2-D point-robot variants.

All rotation coordinates are stored **normalized by `2*pi`** so every coordinate lives on a
comparable scale (the network applies identical Fourier features to all of them).

## Pipeline

Every task follows **preprocess → train → evaluate**. The full set of concrete commands per
environment lives in `README.md`; the shapes of those commands are:

```
# 2-D (DXF inputs)
python dataprocessing/preprocess_dxf.py --env <env.dxf> --shape <shape.dxf> --out <dir> --num_samples 400000 [--visualize]
python train/train_2dshape.py --dataPath datasets/2dshape/<name>
python evaluate_training.py --dataPath testing_data/2dshape/<name> --out ./results/output_2d/<name>

# 3-D (OBJ inputs)
python dataprocessing/preprocess_obj.py --env <env.obj> --shape <shape.obj> --out <dir> --num_samples 400000 [--visualize] [--yrot] [--device cuda:2]
python train/train_3dshape.py --dataPath datasets/3dshape/<name>
python evaluate_training_3d.py --dataPath testing_data/3dshape/<name> --out ./results/output_3d/<name>
```

- Add `--testing_data --num_samples 1000` to a preprocess call to generate the held-out start/goal pairs used for evaluation.
- There is **no test suite, linter, or build step.** Validation is done by training and inspecting the rendered travel-time / trajectory plots.
- `evaluate_training_3d.py` launches a `viser` web viewer on port **8080** (browse episodes via a GUI dropdown) and also writes plotly HTML summaries to `--out`.

## Environment

Runs inside Docker (the host has a Conda env too, but Docker is the supported path). Build once,
then run with X11 + GPU forwarding — see `README.md` for the exact `docker run` invocation
(it bind-mounts the repo to `/workspace` and forwards port 8080).

```
docker build -f Dockerfile.server -t pytorchserver .
```

Key deps: PyTorch 2.2 / CUDA 12.1, `libigl` (mesh SDF / tetrahedralization), `ezdxf`/`shapely`
(2-D geometry), `viser` + `plotly` (3-D eval rendering), `wandb`.

## Architecture

### The `Model` / `Function` / `NN` triad

Each task package is a near-copy of the same three files:

- `models/metric_2dshape/` — the 2-D shape pipeline (`train/train_2dshape.py`, `evaluate_training.py`,
  `evaluate_training_batched.py`).
- `models/metric/` — 3-D shape, maze, gibson, and all the ad-hoc probe/diagnostic scripts.
- `models/metric_arm/` — the UR5.

**Changes to the core algorithm usually need to be mirrored across all three packages.**

- **`model_network_metric.py` — `NN`**: the field network. Inputs are Fourier-mapped
  (`input_mapping`), passed through Lipschitz-normalized layers (`lip_norm`), and the travel time
  between the two endpoints is a learned **metric**: each endpoint is embedded, and `tau` is a
  smooth (`logsumexp`) aggregation of per-group distances between the two embeddings. `out(coords)`
  returns `(tau, w, coords)` and is the single entry point used everywhere. Both endpoints are
  stacked and run through the net in one batch.
- **`model_function_metric.py` — `Function`**: the loss and all field queries. `Loss` is the
  Eikonal residual: `|grad tau| * speed == 1`, with separate distance/angle speed terms, plus a
  surface-normal alignment term and an optional time-difference (`tau_loss`) term. `TravelTimes`,
  `Speed`, and `Gradient` are the inference-time queries used by evaluation. `plot` renders the
  travel-time field.
- **`model_train_metric.py` — `Model`**: the training driver. Owns hyperparameters in the
  `self.Params` dict (epochs, batch size, LR, etc.), the training loop, and checkpointing.

### Training loop specifics (non-obvious)

- The loop in `Model.train()` does **only 5 batches per epoch** (`if ii>4: break`) and divides
  losses by 4.
- It uses an **accept/reject scheme**: each epoch is retried (network + optimizer rolled back to a
  random one of the last 5 saved states) until the loss ratio `current_diff/prev_diff < 1.2`. Watch
  for this when reasoning about "why didn't my change take effect" — bad epochs are silently reverted.
- `beta` (loss scale) is reset to `1/total_diff` each epoch; `speed`/`speed_dist`/`speed_angle` are
  passed through fixed polynomial warps before entering the loss (see lines around the dataloader
  unpacking) — these reshape the clearance signal, not raw clearance.
- LR is **hard-overwritten to `5e-4` inside the loop**, so `--lr` / `self.Params` LR is largely
  cosmetic; change the in-loop assignment to actually alter LR.
- On the first `train()` call the contents of the task's model package (`source_folder` in
  `model_train_metric.py`) are **copied into the run folder**
  (`Experiments/<task>/<dataset>_<timestamp>/models/`) to snapshot the exact code used.

### Checkpoints

- Per-cadence checkpoints `Model_Epoch_XXXXX_ValLoss_*.pt` plus a rolling `latest.pt`, written to
  both the run folder and the top-level `ModelPath` (`./Experiments/<task>/`). Eval scripts load
  `latest.pt` by default.
- A checkpoint stores `model_state_dict`, `optimizer_state_dict`, and crucially `B_state_dict` (the
  random Fourier frequency matrix). The network **cannot** be reconstructed without `B`, so always
  load it via `Model.load` / `load_pretrained_state_dict`.

### Data format

`models/data_mlp.py::Database` loads five `.npy` files from a dataset dir and concatenates them into
one tensor per row, unpacked by offset in the training loop:

```
sampled_points.npy (N, 2*dim)  speed.npy (N, 2)  normal.npy (N, 2*dim)
speed_dists.npy (N, 2)  speed_angles.npy (N, 2)
```

The column layout in the concatenated tensor (see `model_train_metric.py`) is:
`points[:2*dim] | speed[2] | normal[2*dim] | speed_dist[2] | speed_angle[2]`. Each row is a
**correlated pair** `(x0, x1)`: `x0` sampled in the narrow clearance band, `x1` a random
collision-free SE(k) displacement.

### Experiment tracking

W&B is **on by default**. Shared helpers in `train/wandb_utils.py` (`add_wandb_args`,
`apply_overrides`, `start_run`, `finish_run`) are wired into the trainers; per-epoch `wandb.log`
calls in `model_train_metric.py` are guarded so they no-op without a run. Disable with `--no-wandb`.
Common overrides: `--epochs`, `--batch-size`, `--lr`, `--data`, `--set KEY=VALUE`.

## Conventions

- Trainers and eval scripts do `sys.path.append('.')` and must be **run from this nested `ntrl-demo`
  root** (the one containing `models/`, `train/`, `datasets/`).
- Source/goal configurations are hardcoded as the `source` argument to `md.Model(...)` in each
  trainer (e.g. SE(3) goal in `train_3dshape.py`); change them there.
- `dim` is passed explicitly to `Model` and threaded everywhere; it is the single switch between the
  2-D / 3-D / arm variants.
