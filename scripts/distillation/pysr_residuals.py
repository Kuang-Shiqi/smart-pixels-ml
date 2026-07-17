"""
Symbolic regression on residuals between a trained teacher and the
barycenter physics ansatz, per output. The goal: discover closed-form
correction terms that capture residual physics the barycenter formula
misses, then promote those terms into a `pysr_aug` variant.

For each of the 4 outputs (x, y, cot a, cot b) we fit:
    residual_i = f_i(features)
where features are per-cluster summary scalars: cluster widths w_x, w_y,
total charge, charge asymmetries, head/tail ratios, max-pixel charge.

Usage:
  python pysr_residuals.py \
    --teacher-checkpoints <dir> --teacher-model-type Conv2D_Max \
    --thresholds-json <path> \
    --n-events 10000 --niters 40 \
    --out runs/pysr_results
"""
import argparse, os, sys, json

HELPERS = "/work/users/das214/SmartPixels/smart-pixels-ml/two_bit_optimization_helpers"
sys.path.insert(0, HELPERS)

import numpy as np
import tensorflow as tf

from prepare_tfrecords import generate_tfrecords, load_tfrecords
from train import create_model
from symbolic import PhysicsAnsatz


def best_checkpoint(d):
    fs = [f for f in os.listdir(d) if f.endswith('.hdf5')]
    vl = [float(f.split('-v')[1].split('.hdf5')[0]) for f in fs]
    return os.path.join(d, fs[int(np.argmin(vl))])


def cluster_features(charge):
    """Per-cluster summary features. charge: (B, 16, 16, 2)."""
    q = np.sum(charge, axis=-1)        # (B, 16, 16) integrated over time
    prof_x = np.sum(q, axis=2)         # (B, 16)
    prof_y = np.sum(q, axis=1)         # (B, 16)
    active_x = (prof_x > 0).astype(np.float32)
    active_y = (prof_y > 0).astype(np.float32)
    w_x = active_x.sum(axis=1)
    w_y = active_y.sum(axis=1)
    q_tot = q.sum(axis=(1, 2))
    q_max = q.max(axis=(1, 2))
    # head/tail charges along x and y profiles
    def head_tail(p, act):
        # first/last active indices
        idx = np.arange(p.shape[1])
        first = np.argmax(act, axis=1)
        last = (p.shape[1] - 1) - np.argmax(act[:, ::-1], axis=1)
        qF = p[np.arange(p.shape[0]), first]
        qL = p[np.arange(p.shape[0]), last]
        return qF, qL
    qFx, qLx = head_tail(prof_x, active_x)
    qFy, qLy = head_tail(prof_y, active_y)
    ratio_x = (qLx - qFx) / (qLx + qFx + 1e-6)
    ratio_y = (qLy - qFy) / (qLy + qFy + 1e-6)
    # time-slice charge difference (sign-of-alpha proxy)
    q_t0 = charge[..., 0].sum(axis=(1, 2))
    q_t1 = charge[..., 1].sum(axis=(1, 2))
    t_asym = (q_t1 - q_t0) / (q_t1 + q_t0 + 1e-6)

    return np.stack([w_x, w_y, q_tot, q_max,
                     qFx, qLx, qFy, qLy,
                     ratio_x, ratio_y, t_asym], axis=1)


FEATURE_NAMES = ['w_x', 'w_y', 'q_tot', 'q_max',
                 'qFx', 'qLx', 'qFy', 'qLy',
                 'rx', 'ry', 'tasym']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--teacher-checkpoints', required=True)
    p.add_argument('--teacher-model-type', required=True)
    p.add_argument('--thresholds-json', required=True)
    p.add_argument('--dataset',
                   default="/depot/cms/users/das214/datasets/largerWindowPreliminary/dataset_3sr_16x16_50x12P5_centeredIncidence_parquets")
    p.add_argument('--n-events', type=int, default=10000)
    p.add_argument('--niters', type=int, default=40)
    p.add_argument('--maxsize', type=int, default=20)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    thr = json.load(open(args.thresholds_json))
    thresholds = np.array(thr['thresholds'], dtype=np.float32)
    levels = np.array(thr['levels'], dtype=np.float32)

    _, _, tfr_tr, tfr_val = generate_tfrecords(
        dataset_dir=args.dataset, model_type=args.teacher_model_type,
        train_batch_size=5000, val_batch_size=5000,
        select_contained=False, timeslices=2,
        tfrecords_exist=True, seed=args.seed,
    )
    _, vg = load_tfrecords(tfr_tr, tfr_val, noise=-1, digitize=True,
                           digitize_levels=levels, digitize_thresholds=thresholds,
                           seed=args.seed)

    teacher = create_model(args.teacher_model_type, timeslices=2,
                           soft_quantize_layer=False)
    teacher.load_weights(best_checkpoint(args.teacher_checkpoints))

    ansatz = PhysicsAnsatz(variant='barycenter')

    Xs, Ys, Fs = [], [], []
    total = 0
    for batch in vg:
        x, _ = batch
        x = x.numpy() if hasattr(x, 'numpy') else np.asarray(x)
        t14 = teacher(x, training=False).numpy()
        sym = ansatz(x).numpy()   # (B, 4)  — unsigned |cot|
        # apply teacher signs to symbolic angles for residual comparison
        sign_a = np.sign(t14[:, 4]); sign_a[sign_a == 0] = 1
        sign_b = np.sign(t14[:, 6]); sign_b[sign_b == 0] = 1
        sym_signed = sym.copy()
        sym_signed[:, 2] *= sign_a
        sym_signed[:, 3] *= sign_b
        teacher_means = t14[:, 0:8:2]    # (B, 4)
        residual = teacher_means - sym_signed   # (B, 4)
        feats = cluster_features(x)             # (B, F)
        Xs.append(x); Ys.append(residual); Fs.append(feats)
        total += x.shape[0]
        if total >= args.n_events:
            break

    F = np.concatenate(Fs, axis=0)[:args.n_events]
    R = np.concatenate(Ys, axis=0)[:args.n_events]
    print(f"collected {F.shape[0]} events, {F.shape[1]} features")

    # save features + residuals for inspection
    np.savez(os.path.join(args.out, 'features_residuals.npz'),
             features=F, residuals=R, feature_names=np.array(FEATURE_NAMES, dtype=object))

    # PySR import is lazy (Julia precompile)
    from pysr import PySRRegressor

    output_names = ['x', 'y', 'cot_alpha', 'cot_beta']
    discovered = {}
    for i, name in enumerate(output_names):
        out_dir = os.path.join(args.out, name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n=== PySR for residual[{name}] ===", flush=True)
        model = PySRRegressor(
            niterations=args.niters,
            maxsize=args.maxsize,
            binary_operators=['+', '-', '*', '/'],
            unary_operators=['exp', 'tanh', 'square', 'sqrt'],
            populations=15,
            random_state=args.seed,
            output_directory=out_dir,
            equation_file=os.path.join(out_dir, 'hall_of_fame.csv'),
            progress=True,
            verbosity=0,
        )
        model.fit(F, R[:, i], variable_names=FEATURE_NAMES)
        eq = model.get_best()
        discovered[name] = {
            'equation': str(eq['equation']),
            'loss': float(eq['loss']),
            'complexity': int(eq['complexity']),
        }
        print(f"  best: {discovered[name]['equation']}  loss={discovered[name]['loss']:.4g}")

    json.dump(discovered, open(os.path.join(args.out, 'discovered_equations.json'), 'w'),
              indent=1)
    print("\nDONE. Discovered:")
    for k, v in discovered.items():
        print(f"  {k}: {v['equation']}")


if __name__ == '__main__':
    main()
