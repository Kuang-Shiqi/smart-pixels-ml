"""
For each student variant in the distillation sweep, run the saved model
on the validation generator (with teacher signs applied) and write a
performance parquet in the EXACT schema `save_performance_parquet` uses,
so the existing `scripts/comparison_plots.py` can compare them
apples-to-apples against the teacher's parquet.
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import tensorflow as tf

HELPERS = "/work/users/das214/SmartPixels/smart-pixels-ml/two_bit_optimization_helpers"
REPO = "/work/users/das214/SmartPixels/smart-pixels-ml"
sys.path.insert(0, HELPERS)

from prepare_tfrecords import generate_tfrecords, load_tfrecords
from train import create_model
from models.student_max import build_student_max, pack_14, apply_sign, teacher_signs

MAX_PRED_COLS = ['x','M11','y','M22','cotA','M33','cotB','M44',
                 'M21','M31','M32','M41','M42','M43']
TRUTH_COLS = ['xtrue','ytrue','cotAtrue','cotBtrue']


def best_checkpoint(d):
    fs = [f for f in os.listdir(d) if f.endswith('.hdf5')]
    vl = [float(f.split('-v')[1].split('.hdf5')[0]) for f in fs]
    return os.path.join(d, fs[int(np.argmin(vl))])


def write_parquet(preds14, truth, outfile):
    df = pd.DataFrame(preds14, columns=MAX_PRED_COLS)
    for i, c in enumerate(TRUTH_COLS):
        df[c] = truth[:, i]
    for m in ['M11','M22','M33','M44']:
        df[m] = 1e-9 + np.maximum(df[m], 0.0)
    df['sigmax']    = np.abs(df['M11'])
    df['sigmay']    = np.sqrt(df['M21']**2 + df['M22']**2)
    df['sigmacotA'] = np.sqrt(df['M31']**2 + df['M32']**2 + df['M33']**2)
    df['sigmacotB'] = np.sqrt(df['M41']**2 + df['M42']**2 + df['M43']**2 + df['M44']**2)
    for t in ['x','y','cotA','cotB']:
        df[f'residuals_{t}'] = df[t + 'true'] - df[t]
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    df.to_parquet(outfile)
    print(f'wrote {outfile}  ({len(df)} rows)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sweep-dir', default=f"{REPO}/runs/distill_sweep_vit_teacher")
    p.add_argument('--teacher-summary', default=f"{REPO}/runs/vit_max_run/summary.json")
    p.add_argument('--thresholds-json', default=f"{REPO}/runs/vit_max_run/optimized_thresholds.json")
    p.add_argument('--variants', nargs='+', default=['barycenter','localreco','blend'])
    p.add_argument('--out-dir', default=f"{REPO}/runs/processed_parquets/test_3src/2bit_optimized_students")
    args = p.parse_args()

    thr = json.load(open(args.thresholds_json))
    thresholds = np.array(thr['thresholds'], dtype=np.float32)
    levels = np.array(thr['levels'], dtype=np.float32)

    dataset = "/depot/cms/users/das214/datasets/largerWindowPreliminary/dataset_3sr_16x16_50x12P5_centeredIncidence_parquets"
    _, _, tfr_tr, tfr_val = generate_tfrecords(
        dataset_dir=dataset, model_type='ViT_Max',
        train_batch_size=5000, val_batch_size=5000,
        select_contained=False, timeslices=2,
        tfrecords_exist=True, seed=42)
    _, vg = load_tfrecords(tfr_tr, tfr_val, noise=-1, digitize=True,
                           digitize_levels=levels, digitize_thresholds=thresholds, seed=42)

    teacher = create_model('ViT_Max', timeslices=2, soft_quantize_layer=False)
    tckpt = json.load(open(args.teacher_summary))['part2_checkpoints']
    teacher.load_weights(best_checkpoint(tckpt))
    teacher.trainable = False
    print(f'teacher loaded: {tckpt}')

    for variant in args.variants:
        weights = os.path.join(args.sweep_dir, variant, 'student_final.weights.h5')
        if not os.path.exists(weights):
            print(f'!! missing student weights for {variant}: {weights}')
            continue
        student = build_student_max(variant)
        student.load_weights(weights)

        preds_l, truth_l = [], []
        for batch in vg:
            x, y = batch
            t14 = teacher(x, training=False).numpy()
            sa, sb = teacher_signs(tf.constant(t14))
            means_uns, chol = student(x, training=False)
            signed = apply_sign(means_uns, sa, sb)
            s14 = pack_14(signed, chol).numpy()
            preds_l.append(s14)
            truth_l.append(y.numpy() if hasattr(y, 'numpy') else np.asarray(y))
        preds14 = np.concatenate(preds_l, axis=0)
        truth = np.concatenate(truth_l, axis=0)
        outfile = os.path.join(args.out_dir,
                               f'2t-Student_{variant}-distilled-vit_teacher-vars.parquet')
        write_parquet(preds14, truth, outfile)


if __name__ == '__main__':
    main()
