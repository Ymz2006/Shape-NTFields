## About
ntrl shape

## Setup
1. git clone this repo
2. run `docker build -f Dockerfile.server -t pytorchserver .` under the root directory of this repo, once you built the docker image, you don't need to build it again unless you change the dockerfile.
3. run `docker run --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume="/media/corallab-s1/4tbhdd/Jeffrey/ntrlshape/n/Eikonal_Planning/ntrl-demo/ntrl-demo:/workspace" --volume="/usr/lib/x86_64-linux-gnu/:/glu" --volume="/media/corallab-s1/4tbhdd/Jeffrey/ntrlshape/n/.local:/.local" --env="QT_X11_NO_MITSHM=1"  --gpus all -p 8081:8081 -p 8080:8080 -ti --rm pytorchserver` to start the docker container.

All commands below are run from the repo root inside the container.


## General recipe

Every cell is named `<shape>_<env>` and runs the same three steps:
preprocess (training set + held-out test set) -> train -> evaluate.

```
# 1a. training set
python dataprocessing/preprocess_obj.py \
    --env   datasets/3dshape/<env>.obj \
    --shape datasets/3dshape/<shape>.obj \
    --out   datasets/3dshape/<shape>_<env> \
    --num_samples 800000 --batch_size <BS> --device <cuda:N> [--visualize] [--2d] \
    [--margin <m> --offset <o>]

# 1b. test set (start/goal pairs)
python dataprocessing/preprocess_obj.py \
    --env   datasets/3dshape/<env>.obj \
    --shape datasets/3dshape/<shape>.obj \
    --out   testing_data/3dshape/<shape>_<env> \
    --num_samples 1000 --testing_data --offset 0.02 --batch_size <BS> [--visualize] [--2d]

# 2. train
python train/train_3dshape.py --dataPath datasets/3dshape/<shape>_<env> \
    [--modelPath ./Experiments/<task> --name <shape>_<env>] [--epochs N]

# 3. evaluate
python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/<shape>_<env> \
    --out ./results/output_3d/<shape>_<env> \
    [--checkpoint ./Experiments/<task>/<shape>_<env>/latest.pt] [--2d]
```

Notes:

- **`--batch_size`** only chunks the rejection sampler; it never changes the sampled
  distribution, so lower it freely if you OOM. Values known to fit a 24 GB card:

  | setting | `<BS>` |
  | --- | --- |
  | 3-D, box envs (`env1..env4`, `Corozal`) | 2000 (500 for `Tshape3d`) |
  | 2-D (`--2d`) | 500 (1000 works for the sparse `2denv3`) |
  | Gibson `Aloha` | 1000 |
  | Gibson `Badger` | 250 |

- **Epochs.** The trainer default is 5000. The 3-D rows in `../../experiments_ours.md`
  were trained for 10000, so pass `--epochs 10000` for a comparable 3-D cell.
- **`--modelPath` / `--name`** pin the run folder to `Experiments/<task>/<name>/`;
  without them it is `Experiments/<task>/3dshape_<timestamp>/`. Either way, point
  `--checkpoint` at its `latest.pt` (eval otherwise loads the top-level `latest.pt`).


## Pipelines

### 3-D shape (SE(3))

Shapes: `rectangle`, `Lshape3d`, `Fshape3d`, `Ashape3d`, `Vshape3d`, `4shape3d`,
`Tshape3d` (plus `Lcouch` on `Corozal`). Envs: `env1..env4`, `Corozal`.

Example -- `Tshape3d_env1`:

```
python dataprocessing/preprocess_obj.py \
    --env datasets/3dshape/env1.obj --shape datasets/3dshape/Tshape3d.obj \
    --out datasets/3dshape/Tshape3d_env1 \
    --num_samples 800000 --visualize --batch_size 500 --device cuda:2

python dataprocessing/preprocess_obj.py \
    --env datasets/3dshape/env1.obj --shape datasets/3dshape/Tshape3d.obj \
    --out testing_data/3dshape/Tshape3d_env1 \
    --num_samples 1000 --testing_data --offset 0.02 --batch_size 500 --visualize

python train/train_3dshape.py --dataPath datasets/3dshape/Tshape3d_env1 \
    --modelPath ./Experiments/3dshape --name Tshape3d_env1 --epochs 10000

python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/Tshape3d_env1 \
    --out ./results/output_3d/Tshape3d_env1 \
    --checkpoint ./Experiments/3dshape/Tshape3d_env1/latest.pt

# optional: spread (SD) planner, the "SR Alternate Bellman Horizon SD" column
python spread_planner.py --dataPath testing_data/3dshape/Tshape3d_env1 \
    --checkpoint ./Experiments/3dshape/Tshape3d_env1/latest.pt \
    --cases 1000 --steer-bias 0.5 --no-gate --out ./results/spread_sd/Tshape3d_env1
```

To skip training, point `--checkpoint` at `./pretrained/Tshape3d_env1.pt` (the 10000-epoch
checkpoint). The evaluator serves a viser viewer on port 8080 after it finishes and blocks
until Ctrl-C; pass `--no-viser` to exit right away. Add `--cases N` to either script for a
quick run.

### 2-D shape (SE(2) slice, `--2d`)

Same network and trainer; `--2d` restricts the space to `(x, y, rz)` (data is still stored
in the 6-D layout with `z, rx, ry = 0`). Envs: `2denv1..2denv4`, tagged `2denvN` to keep
them apart from the 3-D `envN` cells. Same seven shapes.

**Meshes must be z-up.** The source OBJs are extruded along +Y; convert once (writes
`<name>_zup.obj` beside each input), otherwise the sampler never finds a valid placement:

```
python dataprocessing/obj_yup_to_zup.py datasets/3dshape/<env-or-shape>.obj [...]
```

Example -- `rectangle_2denv4`:

```
python dataprocessing/preprocess_obj.py \
    --env datasets/3dshape/2denv4_zup.obj --shape datasets/3dshape/rectangle_zup.obj \
    --out datasets/3dshape/rectangle_2denv4 \
    --num_samples 800000 --2d --visualize --batch_size 500 --device cuda:2

python dataprocessing/preprocess_obj.py \
    --env datasets/3dshape/2denv4_zup.obj --shape datasets/3dshape/rectangle_zup.obj \
    --out testing_data/3dshape/rectangle_2denv4 \
    --num_samples 1000 --testing_data --offset 0.02 --2d --batch_size 500 --visualize

python train/train_3dshape.py --dataPath datasets/3dshape/rectangle_2denv4 \
    --modelPath ./Experiments/3dshape_2d --name rectangle_2denv4

python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/rectangle_2denv4 \
    --out ./results/output_3d/rectangle_2denv4 \
    --checkpoint ./Experiments/3dshape_2d/rectangle_2denv4/latest.pt --2d
```

### Gibson scans (`Aloha`, `Aloha-y`, `Badger`)

Uses `datasets/gibson/preprocess_obj_gibson.py`, run **from `datasets/gibson`** (so the
output paths are relative to it). It keeps an inside-mesh gate on so every placement lies
inside the building. Shapes are boxes next to the script: `drone`, `drone_small`, `lamp`,
`toycar`, `tabletop`, `pole`. Small shapes need a tighter training band, e.g.
`--margin 0.02 --offset 0.0005` (`drone_small`), `--margin 0.015 --offset 0.001`
(`tabletop`), `--margin 0.01 --offset 0.001` (`pole`); defaults are 0.05 / 0.001.
`Aloha-y.obj` (the `y <= 0` half of Aloha) is made with
`python _make_cut_mesh.py Aloha.obj Aloha-y.obj --axis y --max 0 --cap_res 0.02`.

Example -- `drone_small_Badger`:

```
cd datasets/gibson
python preprocess_obj_gibson.py --env Badger.obj --shape drone_small.obj \
    --out ../3dshape/drone_small_Badger \
    --num_samples 800000 --margin 0.02 --offset 0.0005 --visualize --batch_size 250 --device cuda:0

python preprocess_obj_gibson.py --env Badger.obj --shape drone_small.obj \
    --out ../../testing_data/3dshape/drone_small_Badger \
    --num_samples 1000 --testing_data --offset 0.02 --batch_size 250 --visualize
cd ../..

python train/train_3dshape.py --dataPath datasets/3dshape/drone_small_Badger

python evaluate_training_3d_batched.py --dataPath testing_data/3dshape/drone_small_Badger \
    --out ./results/output_3d/drone_small_Badger
```


## Sweep and scoring scripts

Each script skips cells whose output already exists, so they resume after interruption.

| script | what it does |
| --- | --- |
| `_run_3d_tshape.sh` | all `Tshape3d_env*` cells end to end (+ SD planner). Env: `STAGES`, `ENVS`, `SLOTS`, `BS` |
| `_run_3d_eval_tight.sh` | re-score 3-D cells on `testing_data/3dshape/<cell>_tight` (`--offset 0.005`). Env: `STAGES`, `SLOT_LIST` |
| `_run_2denv{1,2,4}_preprocess.sh` | preprocess all shapes for one 2-D env. Env: `SHAPES`, `GPU_LIST`, `NWORKERS` |
| `_run_2d_train.sh` / `_run_2d_eval.sh` | train / evaluate all 2-D cells |
| `_run_gibson_eval_1k.sh` | score `Lcouch_Corozal` on its RRT-verified set (`testing_data_1k_complete/`) |

Table writers (into `../../experiments_ours.md`): `_make_experiments_3d.py`,
`_make_3d_tight_table.py`, `_make_3d_1k_table.py`, `_make_gibson_table.py`.
Do **not** run `_make_spread_sd.py` after them -- it rewrites the SD column of every table
from `results/spread_sd/` and clobbers the 1k/tight tables.

RRT-verified test sets are built (in the `ntrl_mpnet` env, needs OMPL) with
`dataprocessing/generate_testing_data_rrt.py --shape ... --env ... --out testing_data_1k_complete/3dshape/<cell> --offset 0.02 --num_samples <N> --rrt_time 180 --workers 20`.

### Legacy names

- `datasets/3dshape/Tshape3d_2denv4` was originally called `Tshape3d_env4` (renamed
  2026-09-10); its old test set / results carry a `_legacy` suffix.
- `*/Tshape3d_env1_aug18` is a pre-README 3-D T-shape run (test set at `--offset 0.01`).
