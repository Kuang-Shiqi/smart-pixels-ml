import shutil
from pathlib import Path
import tensorflow as tf
import keras
import numpy as np
import os

def safe_remove_directory(directory_path):
    if Path(directory_path).exists():
        print(f"Directory {directory_path} is removed...")
        shutil.rmtree(directory_path)
    else:
        print(f"Directory {directory_path} does not exist and cannot be removed.")

def check_GPU():
    # set gpu growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
    else:
        print("No GPU(s)")

def data_prep_quantizer(data, bits=3, int_bits=0): # remember there's a secret sign bit
    frac_bits = bits - int_bits
    return np.round(data * 2**frac_bits) * 2**-frac_bits

def diffable_quantizer(data, bits=7, int_bits=0): # remember there's a secret sign bit
    frac_bits = bits - int_bits
    return tf.math.round(data * 2**frac_bits) * 2**-frac_bits

class LearnedScale(keras.layers.Layer):
    def __init__(self, input_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.scale = self.add_weight(
            shape=(self.input_dim, ), initializer="glorot_uniform", trainable=True
        )
        #self.shift = self.add_weight(shape=(input_dim, ), initializer="zeros", trainable=True)

    def call(self, inputs):
        return inputs * tf.math.softplus(self.scale) # + self.shift

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim
        })
        return config

def load_best_model(basedir, model):
    """Load model weights from the checkpoint with lowest validation loss."""
    if not os.path.exists(basedir):
        raise ValueError(f"Directory {basedir} does not exist")

    files = os.listdir(basedir)

    # Filter files that match the expected pattern (contain "-v" and end with ".hdf5")
    valid_files = []
    vlosses = []
    for f in files:
        if "-v" in f and f.endswith(".hdf5"):
            try:
                vloss = float(f.split("-v")[1].split(".hdf5")[0])
                valid_files.append(f)
                vlosses.append(vloss)
            except (IndexError, ValueError):
                continue

    if not valid_files:
        raise ValueError(f"No valid model checkpoint files found in {basedir}")

    bestfile = valid_files[np.argmin(vlosses)]
    best_weights_path = os.path.join(basedir, bestfile)
    model.load_weights(best_weights_path)
    return model, bestfile

def save_best_model(basedir, model):
    """Save model in multiple formats (hdf5, keras, weights, architecture json)."""
    # Create directory if it doesn't exist
    os.makedirs(basedir, exist_ok=True)

    best_model_hdf5 = os.path.join(basedir, 'best_model.hdf5')
    best_model_keras = os.path.join(basedir, 'best_model.keras')
    best_model_weights_hdf5 = os.path.join(basedir, 'best_model_weights.hdf5')
    best_model_weights_keras = os.path.join(basedir, 'best_model_weights.keras')
    model_architecture_json = os.path.join(basedir, 'model_architecture.json')

    # Save (best) model information to files
    model.save(best_model_hdf5)
    model.save(best_model_keras)
    model.save_weights(best_model_weights_hdf5)
    model.save_weights(best_model_weights_keras)
    model_json = model.to_json()
    with open(model_architecture_json, "w") as json_file:
        json_file.write(model_json)

def save_data_as_npy(generator, x_test_npy="npy/X_val.npy", y_test_npy="npy/y_val.npy"):
    """Extract all batches from generator and save as numpy arrays."""
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(x_test_npy), exist_ok=True)
    os.makedirs(os.path.dirname(y_test_npy), exist_ok=True)

    x_val_all = []
    y_val_all = []

    num_batches = len(generator)

    for i_batch in range(num_batches):
        x_val, y_val = generator[i_batch]

        # Convert to numpy if needed
        if hasattr(x_val, 'numpy'):
            x_val = x_val.numpy()
        if hasattr(y_val, 'numpy'):
            y_val = y_val.numpy()

        x_val_all.append(x_val)
        y_val_all.append(y_val)

    x_val_all = np.concatenate(x_val_all)
    y_val_all = np.concatenate(y_val_all)

    np.save(x_test_npy, x_val_all)
    np.save(y_test_npy, y_val_all)

