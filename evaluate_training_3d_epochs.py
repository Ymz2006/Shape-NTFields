"""Run evaluate_training_3d.py for EVERY epoch snapshot in a run directory.

Same signature as evaluate_training_3d.py:

    python evaluate_training_3d_epochs.py --dataPath testing_data/3dshape/rectangle_env1 \
        --out ./results/output_3d/rectangle_env1 --dir 3dshape_07_17_15_47

The added ``--dir`` flag names a run directory under ./Experiments/3dshape (a bare
name like ``3dshape_07_17_15_47`` or a full path).  For each ``Model_Epoch_*.pt``
checkpoint in that directory this script simply RUNS evaluate_training_3d.py --
the evaluation logic is not reimplemented here.  Each epoch's output goes to its
own ``<out>/epoch_XXXXX/`` subfolder, and a combined ``success_rate_by_epoch.txt``
/ ``.csv`` summary is written to ``<out>``.

evaluate_training_3d.py hardcodes its checkpoint and, at the end, launches a
blocking viser viewer.  So that it can be driven per-checkpoint in a loop, each
run is executed in a fresh subprocess through a tiny shim that (a) forces
``Model.load`` to load the chosen epoch snapshot and (b) makes constructing the
viser server exit cleanly (every summary file is written before the viewer
starts).  Nothing in evaluate_training_3d.py itself is modified.

Run from the nested ntrl-demo root (inside the pytorch docker), same as the
single-episode script.
"""

import os
import re
import sys
import csv
import argparse
import subprocess
import tempfile
from glob import glob


MODEL_ROOT = './Experiments/3dshape'
EVAL_SCRIPT = 'evaluate_training_3d.py'


# Shim executed (as a fresh python process) once per epoch snapshot.  It runs the
# unmodified evaluate_training_3d.py but overrides which checkpoint gets loaded and
# skips the blocking interactive viewer.
_SHIM = r'''
import os, sys, runpy
ckpt   = os.environ['EVAL_EPOCHS_CKPT']
script = os.environ['EVAL_EPOCHS_SCRIPT']
dp     = os.environ['EVAL_EPOCHS_DATAPATH']
out    = os.environ['EVAL_EPOCHS_OUT']

# Present evaluate_training_3d.py with exactly its normal CLI.
sys.argv = [script, '--dataPath', dp, '--out', out]
sys.path.insert(0, '.')

# Force the checkpoint: evaluate_training_3d.py hardcodes `pt` then calls
# womodel.load(pt); make load() ignore that and load the epoch snapshot instead.
from models.metric import model_train_metric as md
_orig_load = md.Model.load
def _forced_load(self, filepath=None):
    # The target script prints its own (hardcoded) checkpoint path; make the log
    # state the snapshot actually being loaded so per-epoch logs are truthful.
    print('[epochs-wrapper] loading checkpoint: %s' % ckpt, flush=True)
    return _orig_load(self, ckpt)
md.Model.load = _forced_load

# Neutralize the blocking viser viewer (all summary files are already written by
# the time launch_viser() runs).  Constructing the server exits the process.
try:
    import viser
    def _no_viser(*a, **k):
        raise SystemExit(0)
    viser.ViserServer = _no_viser
except Exception:
    pass

try:
    runpy.run_path(script, run_name='__main__')
except SystemExit:
    pass
'''


def resolve_dir(d):
    """Accept a full path or a bare run-dir name under Experiments/3dshape."""
    if os.path.isdir(d):
        return d
    cand = os.path.join(MODEL_ROOT, d)
    if os.path.isdir(cand):
        return cand
    raise FileNotFoundError('run directory not found: %s (also tried %s)' % (d, cand))


def epoch_of(path):
    """Sort key: integer epoch parsed from Model_Epoch_XXXXX_ValLoss_*.pt."""
    m = re.search(r'Model_Epoch_(\d+)_', os.path.basename(path))
    return int(m.group(1)) if m else -1


def parse_success(txt_path):
    """Pull a few numbers out of an evaluate_training_3d.py success_rate.txt."""
    fields = {}
    if not os.path.isfile(txt_path):
        return fields
    with open(txt_path) as f:
        for line in f:
            if ':' not in line:
                continue
            key, val = line.split(':', 1)
            fields[key.strip()] = val.strip()
    return fields


def main():
    p = argparse.ArgumentParser(
        description='Run evaluate_training_3d.py for every epoch snapshot in a run dir.')
    p.add_argument('--dataPath', default='./testing_data/3dshape/rectangle_env1',
                   help='Test data dir (passed straight through to evaluate_training_3d.py).')
    p.add_argument('--out', default='./output_3d',
                   help='Parent output dir; each epoch writes to <out>/epoch_XXXXX/.')
    p.add_argument('--dir', required=True,
                   help='Run directory under ./Experiments/3dshape (bare name or full path).')
    p.add_argument('--script', default=EVAL_SCRIPT,
                   help='Evaluation script to run per snapshot (default: %s).' % EVAL_SCRIPT)
    p.add_argument('--skip-existing', action='store_true',
                   help='Skip epochs whose output already has success_rate.txt.')
    args = p.parse_args()

    run_dir = resolve_dir(args.dir)
    ckpts = sorted(glob(os.path.join(run_dir, 'Model_Epoch_*.pt')), key=epoch_of)
    if not ckpts:
        raise FileNotFoundError('no Model_Epoch_*.pt checkpoints in %s' % run_dir)

    os.makedirs(args.out, exist_ok=True)
    print('run dir      : %s' % run_dir)
    print('data path    : %s' % args.dataPath)
    print('output root  : %s' % args.out)
    print('snapshots    : %d  (epochs %s)\n'
          % (len(ckpts), ', '.join(str(epoch_of(c)) for c in ckpts)))

    # Write the shim once to a temp file re-used for every epoch.
    shim = tempfile.NamedTemporaryFile('w', suffix='.py', delete=False)
    shim.write(_SHIM)
    shim.close()

    rows = []
    try:
        for i, ckpt in enumerate(ckpts):
            ep = epoch_of(ckpt)
            out_sub = os.path.join(args.out, 'epoch_%05d' % ep)
            os.makedirs(out_sub, exist_ok=True)
            sr_txt = os.path.join(out_sub, 'success_rate.txt')

            if args.skip_existing and os.path.isfile(sr_txt):
                print('[%2d/%2d] epoch %5d  -> skip (exists)' % (i + 1, len(ckpts), ep))
            else:
                env = dict(os.environ)
                # Reduce fragmentation OOMs when the GPU is shared/busy.
                env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
                env['EVAL_EPOCHS_CKPT'] = os.path.abspath(ckpt)
                env['EVAL_EPOCHS_SCRIPT'] = args.script
                env['EVAL_EPOCHS_DATAPATH'] = args.dataPath
                env['EVAL_EPOCHS_OUT'] = out_sub
                log_path = os.path.join(out_sub, 'eval.log')
                print('[%2d/%2d] epoch %5d  -> %s  (log: %s)'
                      % (i + 1, len(ckpts), ep, out_sub, log_path))
                with open(log_path, 'w') as log:
                    ret = subprocess.run([sys.executable, shim.name],
                                         env=env, stdout=log, stderr=subprocess.STDOUT)
                if ret.returncode != 0:
                    print('        WARNING: eval exited with code %d (see %s)'
                          % (ret.returncode, log_path))

            f = parse_success(sr_txt)
            rows.append({
                'epoch': ep,
                'checkpoint': os.path.basename(ckpt),
                'episodes': f.get('episodes', ''),
                'successes': f.get('successes', ''),
                'collision': f.get('collision', ''),
                'no_converge': f.get('no_converge', ''),
                'success_rate': f.get('success_rate', ''),
            })
            sr = rows[-1]['success_rate']
            if sr:
                print('        success_rate: %s' % sr)
    finally:
        os.unlink(shim.name)

    # Combined summary across epochs.
    csv_path = os.path.join(args.out, 'success_rate_by_epoch.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    txt_path = os.path.join(args.out, 'success_rate_by_epoch.txt')
    with open(txt_path, 'w') as f:
        header = '%-8s %-40s %-10s %-10s %-11s %-13s %s' % (
            'epoch', 'checkpoint', 'episodes', 'successes', 'collision',
            'no_converge', 'success_rate')
        f.write(header + '\n')
        for r in rows:
            f.write('%-8s %-40s %-10s %-10s %-11s %-13s %s\n' % (
                r['epoch'], r['checkpoint'], r['episodes'], r['successes'],
                r['collision'], r['no_converge'], r['success_rate']))

    print('\n' + '=' * 96)
    print('SUCCESS RATE BY EPOCH  (run dir: %s)' % run_dir)
    print('=' * 96)
    with open(txt_path) as f:
        sys.stdout.write(f.read())
    print('\nwrote %s' % txt_path)
    print('wrote %s' % csv_path)


if __name__ == '__main__':
    main()
