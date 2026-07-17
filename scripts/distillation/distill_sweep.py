"""
Run the distillation sweep over the three symbolic variants against a single
frozen teacher. Each variant runs `distill_train.py` and the wrapper collates
their summary.json into a single sweep_summary.json.

Usage:
  python distill_sweep.py --teacher-checkpoints <dir> --teacher-model-type <MT> \
    --thresholds-json <path> --epochs 30 \
    --out runs/distill_sweep_<teacher>
"""
import argparse, os, sys, json, subprocess, time

THIS = os.path.dirname(os.path.abspath(__file__))
VARIANTS = ['barycenter', 'localreco', 'blend']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--teacher-checkpoints', required=True)
    p.add_argument('--teacher-model-type', required=True)
    p.add_argument('--thresholds-json', required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--mdmm-eta', type=float, default=1e-3)
    p.add_argument('--warmup-steps', type=int, default=200)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--variants', nargs='+', default=VARIANTS,
                   help='subset of variants to run')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    summary_path = os.path.join(args.out, 'sweep_summary.json')

    sweep = {
        'teacher_model_type': args.teacher_model_type,
        'teacher_checkpoints': args.teacher_checkpoints,
        'epochs': args.epochs,
        'variants': args.variants,
        'results': {},
        't_start': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    json.dump(sweep, open(summary_path, 'w'), indent=1)

    for variant in args.variants:
        variant_out = os.path.join(args.out, variant)
        os.makedirs(variant_out, exist_ok=True)
        cmd = [
            sys.executable, os.path.join(THIS, 'distill_train.py'),
            '--variant', variant,
            '--teacher-checkpoints', args.teacher_checkpoints,
            '--teacher-model-type', args.teacher_model_type,
            '--thresholds-json', args.thresholds_json,
            '--epochs', str(args.epochs),
            '--lr', str(args.lr),
            '--mdmm-eta', str(args.mdmm_eta),
            '--warmup-steps', str(args.warmup_steps),
            '--out', variant_out,
            '--seed', str(args.seed),
        ]
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] launching variant={variant}", flush=True)
        rc = subprocess.call(cmd)
        dt = time.time() - t0
        s = {}
        sp = os.path.join(variant_out, 'summary.json')
        if os.path.exists(sp):
            s = json.load(open(sp))
        s['_returncode'] = rc
        s['_wallclock_s'] = round(dt, 1)
        sweep['results'][variant] = s
        json.dump(sweep, open(summary_path, 'w'), indent=1)
        print(f"[{time.strftime('%H:%M:%S')}] {variant} done rc={rc} dt={dt:.0f}s",
              flush=True)

    # rank by best validation KL (lower is better, the constraint we care about)
    ranked = sorted(
        sweep['results'].items(),
        key=lambda kv: kv[1].get('best_val_kl', float('inf'))
    )
    sweep['ranking_by_best_val_kl'] = [k for k, _ in ranked]
    sweep['t_end'] = time.strftime('%Y-%m-%d %H:%M:%S')
    json.dump(sweep, open(summary_path, 'w'), indent=1)
    print(f"\n=== SWEEP DONE ===")
    print(f"Winner by best_val_kl: {sweep['ranking_by_best_val_kl'][0]}")
    print(f"Full summary: {summary_path}")


if __name__ == '__main__':
    main()
