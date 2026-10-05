"""Training entry point for the 3-D shape path-planning task.

Run ``dataprocessing/preprocess_obj.py`` first to produce the .npy training
data, then start training with:

    python train/train_3dshape.py

The configuration space is SE(3): (x, y, z, rx, ry, rz), so dim = 6.  The three
rotation coordinates (rx, ry, rz) are a rotation vector stored normalized by
2*pi, exactly as produced by ``preprocess_obj.py``.  ``source`` is the goal
configuration used for the periodic travel-time plots written into the
experiment folder.

Experiment tracking (Weights & Biases) is on by default; configure it with the
flags from ``train/wandb_utils.py``, e.g.::

    python train/train_3dshape.py --project ntrl-demo --tags rectangle env1 \
        --notes "narrow margin" --epochs 5000 --batch-size 2000
    python train/train_3dshape.py --no-wandb            # disable tracking
"""

import sys
sys.path.append('.')

import os
import time
import argparse

from models.metric import model_train_metric as md
from train.wandb_utils import add_wandb_args, apply_overrides, start_run, finish_run

parser = argparse.ArgumentParser(description='Train the 3-D shape planner.')
parser.add_argument('--dataPath', default='./datasets/3dshape/rectangle_env1_yrot',
                    help='Directory holding the .npy training data.')
parser.add_argument('--modelPath', default='./Experiments/3dshape',
                    help='Experiment root the run folder is created under.')
parser.add_argument('--name', default=None,
                    help='Explicit run-folder name under --modelPath. Without it '
                         'the folder is <dataset parent>_<timestamp>, which two '
                         'runs launched in the same minute would collide on -- '
                         'pass it when sweeping several datasets in parallel.')
parser.add_argument('--device', default='cuda:0',
                    help='Torch device to train on, e.g. cuda:0, cuda:2, cpu.')
add_wandb_args(parser)
args = parser.parse_args()

modelPath = args.modelPath

# --dataPath / --data set the dataset path; resolve BEFORE building the Model
# since the experiment folder name is derived from it.
dataPath = args.data or args.dataPath

# source / goal configuration (x, y, z, rx, ry, rz) -- rotvec stored normalized by 2*pi
model = md.Model(modelPath, dataPath, 6, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=args.device)
if args.name:
    model.folder = os.path.join(modelPath, args.name)

apply_overrides(model, args)
start_run(args, model, task='3dshape')

start = time.time()
model.train()
print('Training time: {:.1f}s'.format(time.time() - start))

finish_run(args)
