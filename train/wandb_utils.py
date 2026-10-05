"""Shared Weights & Biases helpers for the training entry points.

Both ``train_2dshape.py`` and ``train_3dshape.py`` (and any other trainer) use
these so experiment tracking is configured the same way everywhere.  The actual
per-epoch ``wandb.log`` calls live in ``models/metric/model_train_metric.py`` and
are guarded so they are a no-op unless a run has been started here.

Typical use in a trainer::

    parser = argparse.ArgumentParser()
    add_wandb_args(parser)
    args = parser.parse_args()
    model = md.Model(modelPath, dataPath, dim, source, device)
    apply_overrides(model, args)
    start_run(args, model, task='3dshape')
    model.train()
    finish_run(args)
"""

import argparse
import os


def add_wandb_args(parser):
    """Register the common W&B / experiment flags on an ArgumentParser."""
    g = parser.add_argument_group('wandb / experiment tracking')
    g.add_argument('--wandb', dest='use_wandb', action='store_true', default=True,
                   help='Enable Weights & Biases logging (default: on).')
    g.add_argument('--no-wandb', dest='use_wandb', action='store_false',
                   help='Disable W&B logging (offline run, nothing uploaded).')
    g.add_argument('--project', default='ntrl-demo',
                   help='W&B project name.')
    g.add_argument('--entity', default=None,
                   help='W&B entity (team/user); defaults to your wandb login.')
    g.add_argument('--run-name', default=None,
                   help='Name for this run (defaults to the experiment folder '
                        'name, e.g. 3dshape_06_07_19_10).')
    g.add_argument('--tags', nargs='*', default=None,
                   help='Tags for this run, e.g. --tags baseline rectangle env1.')
    g.add_argument('--notes', default=None,
                   help='Free-text notes describing this run.')
    g.add_argument('--mode', default='online', choices=['online', 'offline', 'disabled'],
                   help='wandb mode.')
    g.add_argument('--set', nargs='*', default=None, metavar='KEY=VALUE',
                   help='Extra config key=value pairs recorded in the run config.')

    # Common training overrides (also recorded in the run config).
    o = parser.add_argument_group('training overrides')
    o.add_argument('--data', default=None,
                   help='Override the dataset path (resolve BEFORE building Model).')
    o.add_argument('--epochs', type=int, default=None,
                   help='Override Number of Epochs.')
    o.add_argument('--batch-size', type=int, default=None,
                   help='Override Batch Size.')
    o.add_argument('--lr', type=float, default=None,
                   help='Override the (initial) Learning Rate config value.')
    return parser


def _parse_set(pairs):
    """Turn ['a=1', 'b=foo'] into {'a': 1, 'b': 'foo'} (numbers auto-cast)."""
    out = {}
    for item in pairs or []:
        if '=' not in item:
            raise ValueError("--set expects KEY=VALUE, got '{}'".format(item))
        k, v = item.split('=', 1)
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        out[k.strip()] = v
    return out


def apply_overrides(model, args):
    """Apply --epochs / --batch-size / --lr overrides onto a built Model.

    (``--data`` must be handled by the trainer before constructing the Model,
    since the experiment folder name is derived from the data path.)
    """
    if getattr(args, 'epochs', None) is not None:
        model.Params['Training']['Number of Epochs'] = args.epochs
    if getattr(args, 'batch_size', None) is not None:
        model.Params['Training']['Batch Size'] = args.batch_size
    if getattr(args, 'lr', None) is not None:
        model.Params['Training']['Learning Rate'] = args.lr


def run_name(model):
    """Name of the experiment folder this run writes into (may be None).

    ``Model.__init__`` builds ``self.folder`` as
    ``<ModelPath>/<dataset>_<MM_DD_HH_MM>``; we use just the last component so
    the wandb run and the on-disk folder share the same name.
    """
    folder = getattr(model, 'folder', None)
    if not folder:
        return None
    return os.path.basename(folder.rstrip('/'))


def build_config(model, args, task):
    """Assemble the wandb run config from the model params + CLI extras."""
    config = {
        'task': task,
        'dim': model.dim,
        'data_path': model.Params['DataPath'],
        'model_path': model.Params['ModelPath'],
        'device': model.Params['Device'],
        'experiment_folder': getattr(model, 'folder', None),
    }
    config.update({'train/' + k: v for k, v in model.Params['Training'].items()})
    config.update(_parse_set(getattr(args, 'set', None)))
    return config


def start_run(args, model, task):
    """Initialize a wandb run (or return None if disabled)."""
    if not getattr(args, 'use_wandb', False):
        return None
    import wandb
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name or run_name(model),
        tags=args.tags,
        notes=args.notes,
        mode=args.mode,
        config=build_config(model, args, task),
    )
    print('wandb run: {} ({})'.format(run.name, run.url))
    return run


def finish_run(args):
    """Cleanly close the wandb run if one was started."""
    if not getattr(args, 'use_wandb', False):
        return
    import wandb
    if wandb.run is not None:
        wandb.finish()
