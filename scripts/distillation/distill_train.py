"""
Distill a frozen teacher (Conv2D_Max or ViT_Max checkpoint) into a tiny
symbolic + 354-param error MLP student under L_data + lambda * KL[T||S]
with MDMM on lambda.

Usage:
  python distill_train.py --variant barycenter \
    --teacher-checkpoints runs/weights/weights-2t-Conv2D_Max-2bit_optimized-<fp>-checkpoints \
    --teacher-model-type Conv2D_Max \
    --thresholds-json runs/conv2d_max_run/optimized_thresholds.json \
    --epochs 30 \
    --out runs/distill_runs/conv2d_teacher_barycenter
"""
import argparse, os, sys, json, time, traceback, csv

THIS = os.path.dirname(os.path.abspath(__file__))
HELPERS = "/work/users/das214/SmartPixels/smart-pixels-ml/two_bit_optimization_helpers"
sys.path.insert(0, HELPERS)

import numpy as np
import tensorflow as tf

for g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

from prepare_tfrecords import generate_tfrecords, load_tfrecords
from train import create_model
from models.student_max import build_student_max
from distill import Distiller


def best_checkpoint(ckpt_dir):
    files = [f for f in os.listdir(ckpt_dir) if f.endswith('.hdf5')]
    if not files:
        raise FileNotFoundError(f"no .hdf5 files in {ckpt_dir}")
    vloss = [float(f.split('-v')[1].split('.hdf5')[0]) for f in files]
    best = files[int(np.argmin(vloss))]
    return os.path.join(ckpt_dir, best)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', required=True,
                   choices=['barycenter', 'localreco', 'blend'])
    p.add_argument('--teacher-checkpoints', required=True,
                   help='dir containing Part-2 .hdf5 checkpoints of the teacher')
    p.add_argument('--teacher-model-type', required=True,
                   help='model_type used to construct the teacher architecture')
    p.add_argument('--thresholds-json', required=True,
                   help='optimized_thresholds.json with "thresholds" and "levels"')
    p.add_argument('--dataset', default="/depot/cms/users/das214/datasets/largerWindowPreliminary/dataset_3sr_16x16_50x12P5_centeredIncidence_parquets")
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--mdmm-eta', type=float, default=1e-3)
    p.add_argument('--kl-target', type=float, default=0.0)
    p.add_argument('--warmup-steps', type=int, default=200)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, 'train.log'), 'a', buffering=1)
    stamp = lambda m: log.write(f"[{time.strftime('%H:%M:%S')}] {m}\n") or log.flush()

    try:
        stamp(f"variant={args.variant} teacher={args.teacher_model_type} "
              f"ckpt={args.teacher_checkpoints}")

        # load Part-1 optimized thresholds (used to digitize the inputs the teacher saw)
        thr_json = json.load(open(args.thresholds_json))
        thresholds = np.array(thr_json['thresholds'], dtype=np.float32)
        levels = np.array(thr_json['levels'], dtype=np.float32)

        # load TFRecords (reuse cached)
        _, _, tfr_tr, tfr_val = generate_tfrecords(
            dataset_dir=args.dataset, model_type=args.teacher_model_type,
            train_batch_size=5000, val_batch_size=5000,
            select_contained=False, timeslices=2,
            tfrecords_exist=True, seed=args.seed,
        )
        tg, vg = load_tfrecords(tfr_tr, tfr_val, noise=-1, digitize=True,
                                digitize_levels=levels, digitize_thresholds=thresholds,
                                seed=args.seed)
        stamp(f"TFRecords loaded: train={tfr_tr}")

        # build teacher and load weights
        teacher = create_model(args.teacher_model_type, timeslices=2,
                               soft_quantize_layer=False)
        teacher.load_weights(best_checkpoint(args.teacher_checkpoints))
        teacher.trainable = False
        stamp(f"teacher loaded, params={teacher.count_params()}")

        # build student, passing labels_scale from TFRecord metadata so the
        # symbolic ansatz output is pre-normalized to label units (v2 fix
        # for the ansatz-collapse pathology in v1).
        meta = json.load(open(os.path.join(tfr_tr, 'metadata.json')))
        labels_scale = meta['labels_scale']
        stamp(f"labels_scale from metadata: {labels_scale}")
        student = build_student_max(args.variant,
                                    ansatz_kwargs={'labels_scale': labels_scale})
        stamp(f"student built, params={student.count_params()}")

        # wire distiller
        distiller = Distiller(student, teacher,
                              lambda_init=0.0, mdmm_eta=args.mdmm_eta,
                              kl_target=args.kl_target,
                              warmup_steps=args.warmup_steps)
        distiller.compile(optimizer=tf.keras.optimizers.Adam(args.lr, clipnorm=1.0))

        # callbacks
        csv_path = os.path.join(args.out, 'history.csv')
        callbacks = [
            tf.keras.callbacks.CSVLogger(csv_path, append=False),
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss_data', patience=50,
                restore_best_weights=True, verbose=1),
        ]

        stamp(f"starting fit: epochs={args.epochs}")
        h = distiller.fit(tg, validation_data=vg, epochs=args.epochs,
                          callbacks=callbacks, shuffle=False, verbose=1)
        stamp("fit complete")

        # save artifacts
        json.dump({k: [float(x) for x in v] for k, v in h.history.items()},
                  open(os.path.join(args.out, 'history.json'), 'w'), indent=1)
        # save student weights
        student.save_weights(os.path.join(args.out, 'student_final.weights.h5'))
        # save trainable-scalar values from the symbolic head
        ansatz = student.get_layer('physics_ansatz')
        ansatz_vars = {w.name: w.numpy().tolist() for w in ansatz.trainable_weights}
        json.dump(ansatz_vars, open(os.path.join(args.out, 'ansatz_scalars.json'), 'w'),
                  indent=1, default=float)

        summary = {
            'variant': args.variant,
            'teacher_model_type': args.teacher_model_type,
            'teacher_checkpoints': args.teacher_checkpoints,
            'student_params': int(student.count_params()),
            'teacher_params': int(teacher.count_params()),
            'epochs': args.epochs,
            'final_train_loss_data': float(h.history['loss_data'][-1]),
            'best_val_loss_data': float(min(h.history.get('val_loss_data', [float('inf')]))),
            'final_train_kl': float(h.history['kl'][-1]),
            'best_val_kl': float(min(h.history.get('val_kl', [float('inf')]))),
            'final_lambda': float(h.history['lam'][-1]),
            'ansatz_scalars': ansatz_vars,
        }
        json.dump(summary, open(os.path.join(args.out, 'summary.json'), 'w'),
                  indent=1, default=float)
        stamp(f"summary: {json.dumps({k: v for k, v in summary.items() if k != 'ansatz_scalars'})}")

    except Exception:
        traceback.print_exc()
        with open(os.path.join(args.out, 'FAILED.txt'), 'w') as f:
            f.write(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
