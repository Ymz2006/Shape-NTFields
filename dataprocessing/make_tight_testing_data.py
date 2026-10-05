"""Regenerate every testing set at a tighter clearance.

Reads the ``--testing_data`` preprocess commands straight out of README.md -- so
the env / shape / flag combinations stay in sync with the documented pipeline --
and reruns each one with a smaller ``--offset``, writing to ``<name>_tight``.
A tighter offset lets start and goal poses sit closer to obstacles, which makes
the queries harder without changing the environments.

One test set that exists on disk but is not in README.md (``rectangle_env1_yrot``)
is appended from EXTRA below.  (``Tshape3d_env1`` used to be listed here; it has
been a README cell since 2026-09-10.)

    python dataprocessing/make_tight_testing_data.py                # all, offset 0.005
    python dataprocessing/make_tight_testing_data.py --dry-run
    python dataprocessing/make_tight_testing_data.py --only rectangle_env1 --jobs 1
"""

import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Test sets present on disk but absent from README.md: (name, env, shape, flags)
EXTRA = [
    ('rectangle_env1_yrot', 'datasets/3dshape/rectangle.obj', 'datasets/3dshape/env1.obj', ['--yrot']),
]


def readme_jobs(readme):
    """(name, shape, env, extra_flags) for every --testing_data block in the README."""
    jobs = []
    for block in re.findall(r'```(.*?)```', open(readme).read(), re.S):
        if '--testing_data' not in block:
            continue
        cmd = ' '.join(block.split())
        name = os.path.basename(re.search(r'--out\s+(\S+)', cmd).group(1))
        env = re.search(r'--env\s+(\S+)', cmd).group(1)
        shape = re.search(r'--shape\s+(\S+)', cmd).group(1)
        flags = [f for f in ('--2d', '--yrot') if f in cmd]
        scale = re.search(r'--shape_scale\s+(\S+)', cmd)
        if scale:
            flags += ['--shape_scale', scale.group(1)]
        jobs.append((name, shape, env, flags))
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--offset', type=float, default=0.005)
    p.add_argument('--suffix', default='_tight')
    p.add_argument('--num-samples', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=1000)
    p.add_argument('--out-root', default='testing_data/3dshape')
    p.add_argument('--jobs', type=int, default=3, help='Concurrent preprocess runs.')
    p.add_argument('--devices', default='cuda:0,cuda:1,cuda:2',
                   help='Comma-separated devices, assigned round-robin. A single '
                        'preprocess run can peak past 13 GB, so keep at most one job '
                        'per GPU or it will OOM.')
    p.add_argument('--only', default=None, help='Substring filter on the test set name.')
    p.add_argument('--force', action='store_true', help='Regenerate sets that already exist.')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    jobs = readme_jobs(os.path.join(ROOT, 'README.md')) + EXTRA
    if args.only:
        jobs = [j for j in jobs if args.only in j[0]]

    log_dir = os.path.join(args.out_root, '_logs')
    os.makedirs(log_dir, exist_ok=True)

    devices = [d.strip() for d in args.devices.split(',') if d.strip()]

    def run(job_with_index):
        index, job = job_with_index
        name, shape, env, flags = job
        device = devices[index % len(devices)] if devices else 'cuda'
        out = os.path.join(args.out_root, name + args.suffix)
        if os.path.exists(os.path.join(out, 'sampled_points.npy')) and not args.force:
            return name, 'skip', 'already exists'
        cmd = [sys.executable, '-u', 'dataprocessing/preprocess_obj.py',
               '--env', env, '--shape', shape, '--out', out,
               '--num_samples', str(args.num_samples), '--testing_data',
               '--offset', str(args.offset), '--batch_size', str(args.batch_size),
               '--device', device, '--visualize'] + flags
        if args.dry_run:
            return name, 'dry-run', ' '.join(cmd)
        log = os.path.join(log_dir, name + args.suffix + '.log')
        with open(log, 'w') as fh:
            rc = subprocess.call(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        ok = rc == 0 and os.path.exists(os.path.join(out, 'sampled_points.npy'))
        return name, 'ok' if ok else 'FAIL', out if ok else log

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        results = list(pool.map(run, enumerate(jobs)))

    for name, status, detail in results:
        print('[{:>7}] {:<24} {}'.format(status, name, detail))
    bad = [r for r in results if r[1] == 'FAIL']
    print('{} sets: {} ok, {} skipped, {} failed'.format(
        len(results), sum(r[1] == 'ok' for r in results),
        sum(r[1] == 'skip' for r in results), len(bad)))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
