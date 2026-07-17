#!/usr/bin/env python
# coding: utf-8

# ##### Change `sys.path.insert(0, '/home/dajiang/smart-pixels-ml/two_bit_optimization_helpers')` to **your** full path to `two_bit_optimization_helpers`


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sys
sys.path.insert(0, "/home/dajiang/smart-pixels-ml/two_bit_optimization_helpers") # Change this line to wherever the two_bit_optimization_helpers is located 
from prepare_tfrecords import load_tfrecords
from train import create_model, get_best_thresholds, cleanup_models_and_generators, get_all_losses, get_all_thresholds


# ### Plot the Loss Curve for Part 1 or Part 2 Training
# ##### Arguments to change:
# * `model_checkpoints` is the path to the directory containing all the saved models for each epoch.
# ##### Optional arguments to change:
# * Any format/styling choices for matplotlib can be changed for your needs.


##### EXTRACT THE MODEL CHECKPOINTS ##################################################################################

model_checkpoints='/data/dajiang/smart-pixels/weights/dataset_3src_16x16_50x12P5_centeredIncidence_weights/weights-2t-Conv2D_Max-2bit_optimized-392456de-checkpoints'
train_losses, validation_losses = get_all_losses(f'{model_checkpoints}')
epochs = np.arange(1, len(train_losses)+1, 1)
#####################################################################################################################


##### PLOT THE MODEL CHECKPOINTS ###################################################################################
fig, ax = plt.subplots()
ax.scatter(x=epochs, y=train_losses, s=2, c='red', label='training loss')
ax.scatter(x=epochs, y=validation_losses, s=2, c='blue', label='validation loss')
ax.set_ylabel('Loss')
ax.set_xlabel('Epoch')
ax.legend()
fig.suptitle(f'Loss vs Epoch')
fig.tight_layout()
# fig.savefig([INSERT FILEPATH TO WHERE YOU WANT TO SAVE THE PLOT], dpi=300)
fig.show()
####################################################################################################################


# ### Extract the best thresholds from Part 1 Training to plot the threshold optimization curve for Part 1 
# ##### Arguments to change:
# * `input_dir` is the path to the directory containing all the saved models for each epoch.
# * `model_type` is the corresponding model type.
# ##### Optional arguments to change:
# * `threshold_offset` is set to 80.0 (default). If you didn't set it to 80.0 in the trainings, please change to what was the offset set in the training.
# * `initial_levels` is set to 0.0, 1.0, 2.0, and 3.0 (default). If this was not used in the trainings, please change to what was set in the training.
# * `timeslices` should always be set to 2, unless much of the training notebook was changed to allow for 20-timeslices trainings.
# * Any format/styling choices for matplotlib can be changed for your needs.


thresholds_1, thresholds_2, thresholds_3 = get_all_thresholds(
    input_dir='/data/dajiang/smart-pixels/weights/dataset_3src_16x16_50x12P5_centeredIncidence_weights/weights-2t-Conv2D_Max-soft_quantize_layer-392456de-checkpoints',
    model_type='Conv2D_Max',
    threshold_offset=80.0,
    initial_levels=np.array([0.0, 1.0, 2.0, 3.0]),
    timeslices=2,
)


fig, ax = plt.subplots(1,3, figsize=(12,4), sharey=True)
epochs=np.arange(1,len(thresholds_1)+1,1)
ax[0].scatter(x=thresholds_1, y=epochs, s=2, c='red', label='threshold 1')
ax[1].scatter(x=thresholds_2, y=epochs, s=2, c='orange', label='threshold 2')
ax[2].scatter(x=thresholds_3, y=epochs, s=2, c='green', label='threshold 3')

ax[0].set_ylabel('Epoch')

ax[0].set_xlabel('Charge (e)')
ax[1].set_xlabel('Charge (e)')
ax[2].set_xlabel('Charge (e)')

ax[0].xaxis.set_major_locator(mticker.MaxNLocator(integer=False, nbins=6))
ax[1].xaxis.set_major_locator(mticker.MaxNLocator(integer=False, nbins=6))
ax[2].xaxis.set_major_locator(mticker.MaxNLocator(integer=False, nbins=6))

fig.legend(framealpha=1)
fig.suptitle(f'Charge Threshold Progression')
fig.tight_layout()
# fig.savefig([INSERT FILEPATH TO WHERE YOU WANT TO SAVE THE PLOT], dpi=300)
fig.show()


# ### Print out the best thresholds used to digitize inputs in Part 2 
# ##### Arguments to change:
# * `input_dir` is the path to the directory containing all the saved models for each epoch.
# * `model_type` is the corresponding model type.
# ##### Optional arguments to change:
# * `threshold_offset` is set to 80.0 (default). If you didn't set it to 80.0 in the trainings, please change to what was the offset set in the training.
# * `initial_levels` is set to 0.0, 1.0, 2.0, and 3.0 (default). If this was not used in the trainings, please change to what was set in the training.
# * `timeslices` should always be set to 2, unless much of the training notebook was changed to allow for 20-timeslices trainings.
# * Any format/styling choices for matplotlib can be changed for your needs.


thresholds, levels = get_best_thresholds(
    checkpoints='/data/dajiang/smart-pixels/weights/dataset_3src_16x16_50x12P5_centeredIncidence_weights/weights-2t-Conv2D_Max-soft_quantize_layer-392456de-checkpoints',
    model_type='Conv2D_Max',
    initial_thresholds=[247.8, 668.4, 1662.9],
    threshold_offset=80.0,
    initial_levels=np.array([0.0, 1.0, 2.0, 3.0]),
    timeslices=2,
)


# ### Plot the Residuals vs True Values

# ##### 1) required functions


def residual_plot(ax, thisdf, var1, var2, name, color, label=None, scaling=1.0, alpha=0.2, slim=False, edgecolor=None, hatch=None):
    
    nbins = 50
    
    var1_scaled = thisdf[var1] * scaling
    var2_scaled = thisdf[var2] * scaling
    residual_scaled = var1_scaled - var2_scaled
    
    xmin = np.min(var1_scaled)
    xmax = np.max(var1_scaled)
    
    step = 1.0*(xmax-xmin)/nbins
    
    x = sns.regplot(x=var1_scaled, y=residual_scaled, x_bins=np.linspace(xmin,xmax,nbins), fit_reg=None, marker='.', ax=ax, color=color, label=label)
    ax.set_xlabel('True ' + name)
    ax.set_ylabel('True - predicted ' + name)
    
    thisdf['residual'+var2] = residual_scaled
    print(var1)
    
    means = []
    upbar = []
    downbar = []
    for i in range(0,nbins):
        means += [np.mean(thisdf['residual'+var2][(var1_scaled>xmin + i*step) & (var1_scaled<xmin + (i+1)*step)])]
        if slim == False:
            upbar += [means[i] + np.mean(thisdf['sigma'+var2][(var1_scaled>xmin + i*step) & (var1_scaled<xmin + (i+1)*step)] * scaling)]
            downbar += [means[i] - np.mean(thisdf['sigma'+var2][(var1_scaled>xmin + i*step) & (var1_scaled<xmin + (i+1)*step)] * scaling)]
    if slim == False:
        ax.fill_between(x=np.linspace(xmin,xmax,nbins),y1=upbar,y2=downbar, alpha=alpha, color=color, edgecolor=edgecolor, hatch=hatch)

pi = 3.14159265359

def inverse_cot(cota):
    a = np.arctan(1.0/cota)
    a[np.where(a<0)] = a[np.where(a<0)] + pi
    return a    

def residual_plot_deg(ax, thisdf, var1, var2, name, color, label=None, scaling=1.0, alpha=0.2, slim=False, edgecolor=None, hatch=None):
    # positions
    if 'cot' not in var1:
        residual_plot(ax, thisdf, var1, var2, name, scaling=scaling)
        return

    thisdf['angle'] = inverse_cot(thisdf[var2].values * scaling)*180/pi

    if slim == False:
        thisdf['angleup'] = abs(inverse_cot((thisdf[var2].values + thisdf['sigma'+var2].values) * scaling)*180/pi - thisdf['angle'])
        thisdf['angledown'] = abs(inverse_cot((thisdf[var2].values - thisdf['sigma'+var2].values) * scaling)*180/pi - thisdf['angle'])
    thisdf['angletrue'] = inverse_cot(thisdf[var1].values * scaling)*180/pi
        
    var1 = 'angletrue'
    var2 = 'angle'
    
    nbins = 50
    xmin = np.min(thisdf[var1])
    xmax = np.max(thisdf[var1])
    
    step = 1.0*(xmax-xmin)/nbins
        
    x = sns.regplot(x=thisdf[var1], y=(thisdf[var1]-thisdf[var2]), x_bins=np.linspace(xmin,xmax,nbins), fit_reg=None, marker='.', ax=ax, color=color, label=label)
    ax.set_xlabel('True ' + name)
    ax.set_ylabel('True - predicted ' + name)
    
    thisdf['residual'+var2] = (thisdf[var1]-thisdf[var2])
    print(var1)
    
    means = []    
    upbar = []
    downbar = []
    for i in range(0,nbins):
        means += [np.mean(thisdf['residual'+var2][(thisdf[var1]>xmin + i*step) & (thisdf[var1]<xmin + (i+1)*step)])]
        if slim == False:
            upbar += [means[i] + np.mean(thisdf['angleup'][(thisdf[var1]>xmin + i*step) & (thisdf[var1]<xmin + (i+1)*step)])]
            downbar += [means[i] - np.mean(thisdf['angledown'][(thisdf[var1]>xmin + i*step) & (thisdf[var1]<xmin + (i+1)*step)])]
    if slim == False:
        ax.fill_between(x=np.linspace(xmin,xmax,nbins),y1=upbar,y2=downbar, alpha=alpha, color=color, edgecolor=edgecolor, hatch=hatch)


# ##### 2) plots for Max or Full Models (4 plots)
# * `performance_parquet` is the filepath to the parquet contains the results of the model tested on a test set (default is dataset_2sc)


##### EXTRACT THE MODEL CHECKPOINTS ##################################################################################
performance_parquet='/home/dajiang/smart-pixels-ml/processed_parquets/dataset_3src_16x16_50x12P5_centeredIncidence/test_dataset_2sc_16x16_50x12P5_centeredIncidence/2bit_optimized/2t-Conv2D_Max-2bit_optimized-392456de-vars.parquet'
df = pd.read_parquet(performance_parquet)

# This is the labels scale list that you can access in the metadata.json created with the TFRecords 
# In this example, I am using the dataset_2sc test-set TFRecords metadata.json because the parquet is associated with testing on that dataset
labels_scale = [
    122.89689703635774,
    30.903849401109394,
    6.560914919886437,
    1.91222249583349,
]
#####################################################################################################################
fig, ax = plt.subplots(2,2, figsize=(15,10))

residual_plot(ax[0,0], df, 'xtrue', 'x', name=r'$x$ $[\mu m]$', edgecolor='red', color='red', scaling=labels_scale[0])
ax[0,0].axhline(alpha=0.4, ls='dashed')
ax[0,0].set_title(r'$x$ residuals + uncertainties')
ax[0,0].legend(loc='upper left')

residual_plot(ax[0,1], df, 'ytrue', 'y', name=r'$y$ $[\mu m]$', edgecolor='red', color='red', scaling=labels_scale[1])
ax[0,1].axhline(alpha=0.4, ls='dashed')
ax[0,1].set_title(r'$y$ residuals + uncertainties')
ax[0,1].legend(loc='upper left')

residual_plot_deg(ax[1,0], df, 'cotAtrue', 'cotA', name=r'$\alpha$ $[deg]$', edgecolor='red', color='red', scaling=labels_scale[2])
ax[1,0].axhline(alpha=0.4, ls='dashed')
ax[1,0].set_title(r'$\alpha$ residuals + uncertainties')
ax[1,0].legend(loc='upper left')

residual_plot_deg(ax[1,1], df, 'cotBtrue', 'cotB', name=r'$\beta$ $[deg]$', edgecolor='red', color='red', scaling=labels_scale[3])
ax[1,1].axhline(alpha=0.4, ls='dashed')
ax[1,1].set_title(r'$\beta$ residuals + uncertainties')
ax[1,1].legend(loc='lower right')

fig.tight_layout(pad=1.0)
# fig.savefig([INSERT FILEPATH TO WHERE YOU WANT TO SAVE THE PLOT], bbox_inches='tight', dpi=300)
fig.show()


# ##### 3) plots for Slim Models (3 plots)
# * `performance_parquet` is the filepath to the parquet contains the results of the model tested on a test set (default is dataset_2sc)


##### EXTRACT THE MODEL CHECKPOINTS ##################################################################################
performance_parquet='/home/dajiang/smart-pixels-ml/processed_parquets/dataset_3src_16x16_50x12P5_centeredIncidence/test_dataset_2sc_16x16_50x12P5_centeredIncidence/2bit_optimized/2t-Mlp_Slim-2bit_optimized-34c2da80-vars.parquet'
df = pd.read_parquet(performance_parquet)

# This is the labels scale list that you can access in the metadata.json created with the TFRecords 
# In this example, I am using the dataset_2sc test-set TFRecords metadata.json because the parquet is associated with testing on that dataset
labels_scale = [
    122.89689703635774,
    30.903849401109394,
    1.91222249583349,
]
#####################################################################################################################
fig, ax = plt.subplots(1,3, figsize=(20,6))

residual_plot(ax[0], df, 'xtrue', 'x', name=r'$x$ $[\mu m]$', color='red', scaling=labels_scale[0], slim=True)
ax[0].axhline(alpha=0.4, ls='dashed')
ax[0].set_title(r'$x$ residuals + uncertainties')

residual_plot(ax[1], df, 'ytrue', 'y', name=r'$y$ $[\mu m]$', color='red', scaling=labels_scale[1], slim=True)
ax[1].axhline(alpha=0.4, ls='dashed')
ax[1].set_title(r'$y$ residuals + uncertainties')

residual_plot_deg(ax[2], df, 'cotBtrue', 'cotB', name=r'$\beta$ $[deg]$', color='red', scaling=labels_scale[2], slim=True)
ax[2].axhline(alpha=0.4, ls='dashed')
ax[2].set_title(r'$\beta$ residuals + uncertainties')

fig.tight_layout(pad=1.0)
# fig.savefig([INSERT FILEPATH TO WHERE YOU WANT TO SAVE THE PLOT], bbox_inches='tight', dpi=300)
fig.show()


