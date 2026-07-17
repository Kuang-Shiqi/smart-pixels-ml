import tensorflow as tf
import numpy as np
import pandas as pd
import random
import os
import shutil
import gc
from natsort import natsorted
from tqdm import tqdm
from AnnealingScheduler import *
from loss import (
    custom_loss,
    custom_diag_loss,
    custom_sse_loss
)
from models.conv2d_model_nonquantized import (
    Conv2D_Max,
    Conv2D_Full,
    Conv2D_Slim,
    Conv2D_Max_SoftQuantizer,
    Conv2D_Full_SoftQuantizer,
    Conv2D_Slim_SoftQuantizer,
)
from models.conv1d_model_nonquantized import (
    Conv1D_Full,
    Conv1D_Slim,
    Conv1D_Full_SoftQuantizer,
    Conv1D_Slim_SoftQuantizer,
)
from models.mlp_encoder_model_nonquantized import (
    Mlp_Full,
    Mlp_Slim,
    Mlp_Full_SoftQuantizer,
    Mlp_Slim_SoftQuantizer,
)
from models.transformer_model_nonquantized import (
    ViT_Max,
    ViT_Full,
    ViT_Slim,
    ViT_Max_SoftQuantizer,
    ViT_Full_SoftQuantizer,
    ViT_Slim_SoftQuantizer,
)
from models.conv2d_model_quantized import (
    QConv2D_Max,
    QConv2D_Full,
    QConv2D_Slim,
    QConv2D_Max_SoftQuantizer,
    QConv2D_Full_SoftQuantizer,
    QConv2D_Slim_SoftQuantizer,
)
from models.conv1d_model_quantized import (
    QConv1D_Full,
    QConv1D_Slim,
    QConv1D_Full_SoftQuantizer,
    QConv1D_Slim_SoftQuantizer,
)
from models.mlp_encoder_model_quantized import (
    QMlp_Full,
    QMlp_Slim,
    QMlp_Full_SoftQuantizer,
    QMlp_Slim_SoftQuantizer,
)
model_list = {
    'Conv2D_Max': [Conv2D_Max, Conv2D_Max_SoftQuantizer],
    'Conv2D_Full': [Conv2D_Full, Conv2D_Full_SoftQuantizer],
    'Conv2D_Slim': [Conv2D_Slim, Conv2D_Slim_SoftQuantizer],

    'Conv1D_Full': [Conv1D_Full, Conv1D_Full_SoftQuantizer],
    'Conv1D_Slim': [Conv1D_Slim, Conv1D_Slim_SoftQuantizer],

    'Mlp_Full': [Mlp_Full, Mlp_Full_SoftQuantizer],
    'Mlp_Slim': [Mlp_Slim, Mlp_Slim_SoftQuantizer],

    'ViT_Max': [ViT_Max, ViT_Max_SoftQuantizer],
    'ViT_Full': [ViT_Full, ViT_Full_SoftQuantizer],
    'ViT_Slim': [ViT_Slim, ViT_Slim_SoftQuantizer],

    'QConv2D_Max': [QConv2D_Max, QConv2D_Max_SoftQuantizer],
    'QConv2D_Full': [QConv2D_Full, QConv2D_Full_SoftQuantizer],
    'QConv2D_Slim': [QConv2D_Slim, QConv2D_Slim_SoftQuantizer],

    'QConv1D_Full': [QConv1D_Full, QConv1D_Full_SoftQuantizer],
    'QConv1D_Slim': [QConv1D_Slim, QConv1D_Slim_SoftQuantizer],

    'QMlp_Full': [QMlp_Full, QMlp_Full_SoftQuantizer],
    'QMlp_Slim': [QMlp_Slim, QMlp_Slim_SoftQuantizer],
}

def create_model(
    model_type,
    timeslices=2,
    soft_quantize_layer=False,
    initial_thresholds=None,
    threshold_offset=0.0,
    initial_levels=None,
):
    if initial_thresholds is None:
        initial_thresholds = [400, 1000, 2000]
    if initial_levels is None:
        initial_levels = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)

    # Construct model
    if soft_quantize_layer:
        if 'Conv2D' in model_type:
            return model_list[model_type][1](
                (16, 16, timeslices),
                n_filters=5,
                pool_size=3,
                initial_thresholds=initial_thresholds,
                threshold_offset=threshold_offset,
                initial_levels=initial_levels,
            )
        else:
            return model_list[model_type][1](
                (16,16,timeslices),
                initial_thresholds=initial_thresholds,
                threshold_offset=threshold_offset,
                initial_levels=initial_levels,
            )
    else:
        if 'Conv2D' in model_type:
            return model_list[model_type][0](
                (16, 16, timeslices),
                n_filters=5,
                pool_size=3,
            )
        else: 
            return model_list[model_type][0]((16, 16, timeslices))

def train(
    model,
    model_type, 
    weights_directory,
    training_generator,
    validation_generator, 
    timeslices=2,
    train_type=None, # full_precision, soft_quantize_layer, 2bit_optimized
    epochs=1,
    seed=10, 
    verbose=1):

    if 'Max' in model_type:
        loss=custom_loss
    elif 'Full' in model_type:
        loss=custom_diag_loss
    elif 'Slim' in model_type:
        loss=custom_sse_loss

    model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-3),
        loss=loss
    )
    model.summary()

    fingerprint = '%08x' % random.randrange(16**8)
    if train_type:
        weights_directory = weights_directory + '/weights-{}t-{}-{}-{}-checkpoints'.format(
            timeslices, model_type, train_type, fingerprint
        )
    else:
        print('Please specify train_type: soft_quantize_layer or 2bit_optimized.')
        return 0
    print(f'Checkpoints saved to {weights_directory}')
    
    # --- Clear old checkpoint directory if it exists ---
    if os.path.exists(weights_directory):
        shutil.rmtree(weights_directory)
    os.makedirs(weights_directory)  # Create fresh directory

    checkpoint_filepath = weights_directory + '/weights.{epoch:02d}-t{loss:.2f}-v{val_loss:.2f}.hdf5'
    
    mcp = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        save_best_only=False,
    )

    print('Model fingerprint: {}'.format(fingerprint))

    history = None
    if train_type == 'soft_quantize_layer':
        scheduler_callback = AnnealingScheduler(
            schedule='cosine',  
            target_layer_name='soft_quantizer_output', 
            initial_k=1.0,
            final_k=67.0,
            verbose=1      
        )
    
        history = model.fit(
            x=training_generator,
            validation_data=validation_generator,
            callbacks=[mcp, scheduler_callback],
            epochs=epochs,
            shuffle=False,
            verbose=verbose
        )
    else:
        history = model.fit(
            x=training_generator,
            validation_data=validation_generator,
            callbacks=[mcp],
            epochs=epochs,
            shuffle=False,
            verbose=verbose
        )
    
    return weights_directory, fingerprint, history
    
def get_best_thresholds(
    checkpoints,
    model_type,
    timeslices=2,
    initial_thresholds=[400, 1000, 2000],
    threshold_offset=80.0,
    initial_levels=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    ):

    # Create model based on model type (it needs to be model w/ soft quantize layer)
    if 'Conv2D' in model_type:
        model = model_list[model_type][1](
            (16, 16, timeslices),
            n_filters=5,
            pool_size=3,
            initial_thresholds=initial_thresholds,
            threshold_offset=threshold_offset,
            initial_levels=initial_levels,
        )
    else:
        model = model_list[model_type][1](
            (16,16,timeslices),
            initial_thresholds=initial_thresholds,
            threshold_offset=threshold_offset,
            initial_levels=initial_levels,
        )
    
    # Find the best checkpoint (lowest validation loss)
    checkpoints_contents = os.listdir(checkpoints)
    vlosses = [float(c.split('-v')[1].split('.hdf5')[0]) for c in checkpoints_contents]
    bestfile = checkpoints_contents[np.argmin(vlosses)]
    model.load_weights(checkpoints+'/'+bestfile)
    print('Best model: {}'.format(bestfile))

    # Extract the thresholds of the best checkpoint
    layer = model.get_layer('soft_quantizer_output')
    print('---Soft Quantize Layer Details---')
    thresholds = layer.thresholds.numpy()
    levels = layer.levels.numpy()
    print(f'  Thresholds (e): {thresholds}') 
    print(f'  Levels: {levels}')

    return thresholds, levels

def cleanup_models_and_generators(objects: list):
    '''
    Deletes models and generators, clears TF session, and forces garbage collection.
    
    Args:
        objects (list): List of models, generators, or other large objects to delete.
    '''
    for obj in objects:
        try:
            # If the object has large internal data, clear it first
            if hasattr(obj, 'current_dataframes'):
                obj.current_dataframes = None
            if hasattr(obj, 'files'):
                obj.files = None

            del obj
        except Exception as e:
            print(f'Warning: could not delete object {obj}: {e}')

    # Clear TF/Keras session to free GPU memory
    tf.keras.backend.clear_session()
    
    # Force Python garbage collection
    gc.collect()
    
    print('Cleanup complete: models, generators, and GPU memory freed.')

def get_all_losses(input_dir):
    checkpoints = natsorted(os.listdir(input_dir))
    train_losses = np.array([float(f.split('-t')[-1].split('-v')[0]) for f in checkpoints])
    validation_losses = np.array([float(f.split('-v')[-1].split('.hdf5')[0]) for f in checkpoints])

    return train_losses, validation_losses

def get_all_thresholds(
    input_dir, 
    model_type, 
    initial_thresholds=[400, 1000, 2000], 
    initial_levels=np.array([0,1,2,3]), 
    threshold_offset=80.0, 
    timeslices=2
): 
    checkpoints = natsorted(os.listdir(input_dir))
    thresholds_1 = []
    thresholds_2 = []
    thresholds_3 = []
    
    model = create_model(
        model_type=model_type,
        timeslices=timeslices,
        soft_quantize_layer=True,
        initial_thresholds=initial_thresholds,
        threshold_offset=threshold_offset,
        initial_levels=initial_levels,
    )
            
    for i in tqdm(range(len(checkpoints))):
        model.load_weights(f'{input_dir}/{checkpoints[i]}', by_name=True, skip_mismatch=True)
        sq_layer = model.get_layer(name='soft_quantizer_output')
        thresholds_1.append(sq_layer.thresholds.numpy()[0])
        thresholds_2.append(sq_layer.thresholds.numpy()[1])
        thresholds_3.append(sq_layer.thresholds.numpy()[2])
        
    return thresholds_1, thresholds_2, thresholds_3

def save_performance_parquet(
    checkpoints,
    output_directory,
    test_generator, 
    model_type,
    train_type, # full_precision, soft_quantize_layer, 2bit_optimized
    fingerprint,
    timeslices=2,
    soft_quantize_layer=False,
    initial_thresholds=[400, 1000, 2000],
    threshold_offset=80.0,
    initial_levels=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
):

    # -------- Create model --------
    model = create_model(
        model_type=model_type,
        timeslices=timeslices,
        soft_quantize_layer=soft_quantize_layer,
        initial_thresholds=initial_thresholds,
        threshold_offset=threshold_offset,
        initial_levels=initial_levels,
    )

    # -------- Find best checkpoint --------
    files = [f for f in os.listdir(checkpoints) if f.endswith(".hdf5")]
    vloss = [float(f.split("-v")[1].split(".hdf5")[0]) for f in files]
    bestfile = files[np.argmin(vloss)]

    model.load_weights(os.path.join(checkpoints, bestfile))
    print(f"Best model: {bestfile}")

    # -------- Predict and collect truth --------
    preds = model.predict(test_generator)
    truth = np.concatenate([y for _, y in test_generator], axis=0)

    # -------- Define model-specific output schema --------
    schemas = {
        "Max":  dict(
            pred_cols=['x','M11','y','M22','cotA','M33','cotB','M44','M21','M31','M32','M41','M42','M43'],
            truth_cols=['xtrue','ytrue','cotAtrue','cotBtrue'],
        ),
        "Full": dict(
            pred_cols=['x','y','cotA','cotB','M11','M22','M33','M44'],
            truth_cols=['xtrue','ytrue','cotAtrue','cotBtrue'],
        ),
        "Slim": dict(
            pred_cols=['x','y','cotB'],
            truth_cols=['xtrue','ytrue','cotBtrue'],
        ),
    }

    key = "Max" if "Max" in model_type else "Full" if "Full" in model_type else "Slim" if "Slim" in model_type else None
    if key is None:
        raise ValueError('INVALID model_type: must contain "Max", "Full", or "Slim"')

    pred_cols = schemas[key]["pred_cols"]
    truth_cols = schemas[key]["truth_cols"]

    df = pd.DataFrame(preds, columns=pred_cols)

    for i, col in enumerate(truth_cols):
        df[col] = truth[:, i]

    if key == "Max":
        for m in ["M11","M22","M33","M44"]:
            df[m] = 1e-9 + tf.math.maximum(df[m], 0.0)

        df["sigmax"]     = abs(df["M11"])
        df["sigmay"]     = np.sqrt(df["M21"]**2 + df["M22"]**2)
        df["sigmacotA"]  = np.sqrt(df["M31"]**2 + df["M32"]**2 + df["M33"]**2)
        df["sigmacotB"]  = np.sqrt(df["M41"]**2 + df["M42"]**2 + df["M43"]**2 + df["M44"]**2)

    elif key == "Full":
        mapping = {
            "M11": "sigmax",
            "M22": "sigmay",
            "M33": "sigmacotA",
            "M44": "sigmacotB",
        }
        for m, new_name in mapping.items():
            df[new_name] = tf.nn.softplus(df[m]) + 1e-9

    target_names = ["x", "y", "cotA", "cotB"]
    for i, t in enumerate(target_names):
        t_pred = t
        t_true = t + "true"
        if t_pred in df.columns and t_true in df.columns:
            df[f"residuals_{t}"] = df[t_true] - df[t_pred]

    os.makedirs(output_directory, exist_ok=True)
    outfile = f"{output_directory}/{timeslices}t-{model_type}-{train_type}-{fingerprint}-vars.parquet"
    df.to_parquet(outfile)
    print(f'Successfully saved to {outfile}')
    return 1
