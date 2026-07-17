#!/usr/bin/env python
# coding: utf-8

# ##### Change `sys.path.insert(0, '/home/dajiang/smart-pixels-ml/two_bit_optimization_helpers')` to **your** full path to `two_bit_optimization_helpers`


import tensorflow as tf
import random
import numpy as np
import json
import sys
sys.path.insert(0, '/work/users/das214/SmartPixels/smart-pixels-ml/two_bit_optimization_helpers') # Change this line to wherever the two_bit_optimization_helpers is located 
from prepare_tfrecords import generate_tfrecords, load_tfrecords
from train import create_model, train, get_best_thresholds, cleanup_models_and_generators, save_performance_parquet
from viz_utils import viz_history
from utils import load_best_model, save_best_model, save_data_as_npy


# ##### SET THESE PARAMETERS before you `Run All Cells`


### REQUIRED TO CHANGE ##############################################################################################################################################################################
dataset_3src_dir='/depot/cms/users/das214/datasets/largerWindowPreliminary/dataset_3sr_16x16_50x12P5_centeredIncidence_parquets'
dataset_2sc_dir='/depot/cms/users/das214/datasets/largerWindowPreliminary/dataset_3sr_16x16_50x12P5_centeredIncidence_parquets'  # no separate 2sc set: Part-3 reuses the 3sr test split (matches legacy workflow)

weights_directory='/work/users/das214/SmartPixels/smart-pixels-ml/runs/weights/'
performance_directory_3src='/work/users/das214/SmartPixels/smart-pixels-ml/runs/processed_parquets/test_3src/2bit_optimized/'
performance_directory_2sc='/work/users/das214/SmartPixels/smart-pixels-ml/runs/processed_parquets/test_2sc/2bit_optimized/'


# For non-quantized models: Conv2D_Max, Conv2D_Full, Conv2D_Slim, Conv1D_Full, Conv1D_Slim, Mlp_Full, Mlp_Slim, ViT_Max, ViT_Full, ViT_Slim
# For quantized models: QConv2D_Max, QConv2D_Full, QConv2D_Slim, QConv1D_Full, QConv1D_Slim, QMlp_Full, QMlp_Slim
model_type='Conv2D_Max' 
#####################################################################################################################################################################################################

### OPTIONAL TO CHANGE ##############################################################################################################################################################################
tfrecords_exist_3src=False
tfrecords_exist_2sc=False

initial_thresholds=[247.8, 668.4, 1662.9]
seed=42

# If you want to SKIP sections of the pipeline, change these flags to True
skip_part_1=False
skip_part_2=False

# IF SKIPPING PART 1, THESE ARE REQUIRED FOR PART 2:
tfrecords_dir_train=''                                    # Path to TFRecords for dataset_3src train_contained
tfrecords_dir_val=''                                      # Path to TFRecords for dataset_3src test_contained
tfrecords_dir_test=''                                     # Path to TFRecords for dataset_2sc test_contained
thresholds=[100, 200, 300]                                # CUSTOM THRESHOLDS
labels_scale=[]                                           # CUSTOM labels scale for 2sc (Max/Full models require 4 elements, Slim models require 3 elements)
#####################################################################################################################################################################################################

### DON'T CHANGE UNLESS YOU KNOW WHAT YOU ARE DOING #################################################################################################################################################
timeslices=2
select_contained=False  # dataset has train/ + test/, not *_contained/ subdirs
train_batch_size=5000
val_batch_size=5000
threshold_offset=80.0
initial_levels=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)   # 2-bit in outputs (same for Part 1 and Part 2)

epochs1=1000
noise1=[0,80]                                                     # Gaussian noise for Part 1
train_type1='soft_quantize_layer'                                 # soft_quantize_layer training for Part 1
soft_quantize_layer1=True                                         # soft_quantize_layer added to model in Part 1

epochs2=1000
noise2=-1                                                         # No noise for Part 2
train_type2='2bit_optimized'                                      # 2bit_optimized training for Part 2
soft_quantize_layer2=False                                        # default model architecture in Part 2
#####################################################################################################################################################################################################


# ##### Set random seeds for tensorflow and random


if not skip_part_1:
    tf.random.set_seed(seed)
    random.seed(seed)


# ##### Generate and save TFRecords. Then load the TFRecords with noise added.


if not skip_part_1:
    dataset_train_dir, dataset_validation_dir, tfrecords_dir_train, tfrecords_dir_val = generate_tfrecords(
        dataset_dir=dataset_3src_dir,
        model_type=model_type,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        select_contained=select_contained,
        timeslices=timeslices,
        tfrecords_exist=tfrecords_exist_3src,
        seed=seed,
    )
    
    training_generator1, validation_generator1 = load_tfrecords(
        tfrecords_dir_train, 
        tfrecords_dir_val,
        noise=noise1,
        seed=seed,
    )


# ##### Obtain the labels scaling done on thes 3src test set and apply on the generating the 2sc TFRecords.


if not skip_part_1:
    with open(f'{tfrecords_dir_val}/metadata.json', 'r') as file:
        data = json.load(file)
    labels_scale = data['labels_scale']

    dataset_test_dir, tfrecords_dir_test = generate_tfrecords(
        dataset_dir=dataset_2sc_dir, 
        train_batch_size=train_batch_size, 
        val_batch_size=val_batch_size,
        select_contained=select_contained,
        timeslices=timeslices, 
        seed=seed, 
        max_workers=1, 
        tfrecords_exist=tfrecords_exist_2sc, 
        model_type=model_type,
        labels_scale=np.array(labels_scale),
        test_only=True, # don't load training and validation data-generators
    )


# ##### Training part 1


if not skip_part_1:
    model1 = create_model(
        model_type=model_type,
        timeslices=timeslices,
        soft_quantize_layer=soft_quantize_layer1,
        initial_thresholds=initial_thresholds,
        threshold_offset=threshold_offset,
        initial_levels=initial_levels,
    )
    
    checkpoints_directory1, fingerprint1, history1 = train(
        model=model1,
        model_type=model_type, 
        weights_directory=weights_directory,
        training_generator=training_generator1,
        validation_generator=validation_generator1, 
        timeslices=timeslices,
        train_type=train_type1,
        epochs=epochs1,
        seed=seed, 
    )


# ##### Input Digitization


if not skip_part_1:
    thresholds, levels = get_best_thresholds(
        checkpoints=checkpoints_directory1,
        model_type=model_type,
        timeslices=timeslices,
        initial_thresholds=initial_thresholds,
        threshold_offset=threshold_offset,
        initial_levels=initial_levels,
    )


# ##### Cleanup of previous models and generators


if not skip_part_1:
    cleanup_models_and_generators([model1, training_generator1, validation_generator1])


# ##### Load TFRecords without noise added, then digitized to 2bits according to thresholds and levels from part 1 (or hardcoded thresholds if skip part 1)


if not skip_part_2:
    training_generator2, validation_generator2 = load_tfrecords(
        tfrecords_dir_train, 
        tfrecords_dir_val,
        noise=noise2,
        digitize=True,
        digitize_levels=initial_levels,
        digitize_thresholds=thresholds,
        seed=seed,
    )


# ##### Reset random seeds for tensorflow and random


if not skip_part_2:
    tf.random.set_seed(seed)
    random.seed(seed)


# ##### Training part 2


if not skip_part_2:
    model2 = create_model(
        model_type=model_type,
        timeslices=timeslices,
        soft_quantize_layer=soft_quantize_layer2,
    )
    
    checkpoints_directory2, fingerprint2, history2 = train(
        model=model2,
        model_type=model_type, 
        weights_directory=weights_directory,
        training_generator=training_generator2,
        validation_generator=validation_generator2, 
        timeslices=timeslices,
        train_type=train_type2,
        epochs=epochs2,
        seed=seed, 
    )


# ##### Save best weights from part 2 and process to parquet files with performance variables


if not skip_part_2:
    save_performance_parquet(
        checkpoints=checkpoints_directory2,
        output_directory=performance_directory_3src,
        test_generator=validation_generator2, 
        model_type=model_type,
        train_type=train_type2,
        fingerprint=fingerprint2,
        timeslices=2,
        soft_quantize_layer=False,
    )


if not skip_part_2:
    cleanup_models_and_generators([model2, training_generator2, validation_generator2])


# ##### Generate TFRecords for 2sc, then load with thresholds


if not skip_part_2:
    dataset_test_dir, tfrecords_dir_test = generate_tfrecords(
        dataset_dir=dataset_2sc_dir, 
        train_batch_size=train_batch_size, 
        val_batch_size=val_batch_size,
        select_contained=select_contained,
        timeslices=timeslices, 
        seed=seed, 
        max_workers=1, 
        tfrecords_exist=tfrecords_exist_2sc, 
        model_type=model_type,
        labels_scale=np.array(labels_scale),
        test_only=True, # don't load training and validation data-generators
    )
    
    test_generator = load_tfrecords(
        tfrecords_dir_test=tfrecords_dir_test, 
        noise=noise2,
        quantize=False, 
        shuffle=True,
        digitize=True,
        digitize_levels=initial_levels,
        digitize_thresholds=thresholds,
        seed=seed,
        test_only=True, # don't load training and validation data-generators
    )


# ##### Test the model on dataset_2sc and save result to performance_directory


if not skip_part_2:
    save_performance_parquet(
        checkpoints=checkpoints_directory2,
        output_directory=performance_directory_2sc,
        test_generator=test_generator, 
        model_type=model_type,
        train_type=train_type2,
        fingerprint=fingerprint2,
        timeslices=timeslices,
        soft_quantize_layer=soft_quantize_layer2,
    )


# Other notebooks plots the history, but I find it helpful here too
#viz_history(history2, title=model_type, prefix='best-model/dataset_3src_16x16_50x12P5_centeredIncidence_weights/history-{}t-{}-2bit_optimized-{}-checkpoints'.format(timeslices, model_type, fingerprint2))


#viz_history(history2, title=model_type, start_epoch=500, prefix='best-model/dataset_3src_16x16_50x12P5_centeredIncidence_weights/history-{}t-{}-2bit_optimized-{}-checkpoints'.format(timeslices, model_type, fingerprint2))


# Load the best model
# Is there a better way? Is it already loaded somewhere?
#weights_dir = './smart-pixels/weights/dataset_3src_16x16_50x12P5_centeredIncidence_weights/weights-{}t-{}-2bit_optimized-{}-checkpoints'.format(timeslices, model_type, fingerprint2)
#model2_best, bestfile = load_best_model(weights_dir, model2)

#print('Best weights: {}'.format(bestfile))


# Save the model in several formats (h5, keras, json)
#weights_dir = 'best-model/dataset_3src_16x16_50x12P5_centeredIncidence_weights/weights-{}t-{}-2bit_optimized-{}-checkpoints'.format(timeslices, model_type, fingerprint2) # CHANGE ME
#save_best_model(weights_dir, model2_best)


# Get validation data. Is there a better way?
#from OptimizedDataGenerator_v3 import OptimizedDataGenerator

#test_generator = OptimizedDataGenerator(
#    load_from_tfrecords_dir = '{}/TFR_files/{}t/TFR_val_contained/'.format(dataset_dir, timeslices), # CHANGE ME
#    quantize = False # False for soft quantizer and manually quantized inputs
#)


#save_data_as_npy(test_generator, x_test_npy="npy/dataset_3sr_16x16_50x12P5_centeredIncidence_X_val_contained.npy", y_test_npy="npy/dataset_3sr_16x16_50x12P5_centeredIncidence_y_val_contained.npy")


# Run prediction with the best model
#p_test = model2_best.predict(test_generator)


