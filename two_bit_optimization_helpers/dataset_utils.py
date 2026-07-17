import numpy as np
from copy import deepcopy
import pandas as pd
import math
import time

# import warnings
# warnings.filterwarnings("ignore")

# charge_levels = [[-9e19, 400], [400, 800], [800,1200], [1200, 9e19]]

def add_noise(x, mu = 0, sig = 80, shuffled = True, seed = None):
    '''
    Add random gaussian noise to input data 
        x (np.array or pd.DataFrame): input data with dims (event, time, 2d image)
        mu (float): mean of noise 
        sig (float): standard deviation of noise
        shuffled (bool): was the dataset megashuffled? ie. are the labels and data both present in the df?
        seed (None, int): seed for reproducable random noise sampling. Set to None to disable
    '''
    df = deepcopy(x)
    if shuffled:
        cols = [c for c in x.columns if c.isnumeric()]
        data = df[cols]
    else:
        data=df
    
    dshape = data.shape
    
    rng = np.random.default_rng(seed = seed)
    noise = rng.normal(mu, sig, dshape)

    # if integrate: #DEPRECIATED
    #     noise = np.cumsum(noise, axis = 1)
    data = data + noise
    if cols:
        df[cols] = data
    else:
        df = data
    return df

def apply_threshold(x, thresh = 400, minimum=True, shuffled=True):
    '''
    Apply a threshold to input data
        data (np.array or pd.DataFrame): input data
        minimum (bool): zero charge bellow threshold?
        thresh (float): Charge threshold to zero out charge bellow if minimum==True. 
                        zeros charge above thresh if minimum==False.
    '''
    df = deepcopy(x)
    
    if shuffled: 
        cols = [c for c in df.columns if c.isnumeric()]
        data = df[cols]
    else:
        data=df

    if minimum:
        bellowthresh = data < thresh
        data[bellowthresh] = 0
    else:
        abovethresh = data > thresh
        data[abovethresh] = thresh
    
    if shuffled:
        df[cols] = data
    else:
        df=data
        
    return df

def quantize_manual(x, 
                    charge_levels=[400,800,1200], 
                    quant_values=[0,1,2,3], 
                    shuffled=True
                   ):
    '''
    Quantize a df with manually defined charge level boundaries
        x (np.array or pd.DataFrame): input data (with or without labels).
        charge_levels (list, shape=(N-1)): finite charge levels for boundaries of N bins. 
            eg. for N=4 bins with boundaries [-9e19, 400], [400, 800], [800,1200], [1200, 9e19]
            use: charge_levels = [400, 800, 1200]
        quant_values (list): list of values for each of N charge bins
        shuffled (bool, default: True): is this dataframe from a dataset shuffled? ie. are
            clusters and labels both present in the dataframe x
    '''
    # start_time = time.time()
    df = deepcopy(x)
    if shuffled: 
        cols = [c for c in df.columns if c.isnumeric()]
        data = df[cols].values
    else:
        data = df.values
    # print(f'get data: {time.time()-start_time:.4f}', f'total: {time.time()-start_time:.4f}')
    # newtime = time.time()
    charge_levels= np.array(charge_levels)
    minval, maxval = [-9e19], [9e19]

    #pad the charge levels with +/- inf
    charge_levels = np.append(minval, charge_levels)
    charge_levels= np.append(charge_levels, maxval)

    #turn charge_levels into bin boundaries
    bins = None
    for c in range(len(charge_levels)-1):
        if bins is None:
            bins = [[charge_levels[c], charge_levels[c+1]]]
        else:
            bins = np.append(bins, [[charge_levels[c], charge_levels[c+1]]], axis =0)
    # print(f'make bins: {time.time()-newtime:.4f}', f'total: {time.time()-start_time:.4f}')
    # newtime = time.time()

    #quantize the data
    dfq = pd.DataFrame(np.zeros_like(data),index=df.index, columns=cols)
    for j, binbounds in enumerate(bins):
        #mask pixels by charge bin
        mask = (data>binbounds[0]) & (data<binbounds[1])
        dfq = dfq.mask(mask, quant_values[j])
    if shuffled:
        df[cols] = dfq
    else:
        df = dfq
    # print(f'make quantized data: {time.time()-newtime:.4f}', f'total: {time.time()-start_time:.4f}')

    try:
        return df
    finally:
        del dfq, data, mask, df, cols

def apply_offset(block, offset, pixel_array_sizeX, pixel_array_sizeY):
    '''
    Apply an offset to the per-time-stamp data from the entire pixel array
        block (list): input data at a specific time stamp (2D)
        offset (tuple): (x_offset, y_offset) to apply to the block
        pixel_array_sizeX (int): size of the pixel array in X dimension
        pixel_array_sizeY (int): size of the pixel array in Y dimension
    '''
    rows, cols = pixel_array_sizeY, pixel_array_sizeX
    new_block = [[0 for _ in range(cols)] for _ in range(rows)]
    
    for i in range(rows):
        for j in range(cols):
            if block[i][j] != 0:
                new_i = i + offset[0]
                new_j = j + offset[1]
                if 0 <= new_i < rows and 0 <= new_j < cols:
                    new_block[new_i][new_j] = block[i][j]
    return new_block

def contained(df, axis = 'y'):
    '''
    Yeilds the subset of a datframe df with no charge at the edge bins of the image along a specified axis. NOTE: This assumes the combined data and label format in shuffled datasets.
        axis (str): axis of cluster to project along. 
    '''
    dshape = (20,13,21)
    cols = np.arange(math.prod(dshape)).astype('str')
    data = df[cols].values.reshape(-1, *dshape)
    
    if axis == 'x':
        projection = data[:,-1,:,:].sum(axis=1)
    elif axis == 'y':
        projection = data[:,-1,:,:].sum(axis=2)
    else:
        raise ValueError("axis must be either 'x' or 'y'.")
    
    contained = np.zeros(len(data)).astype('bool')
    for i, proj in enumerate(projection):
        if (proj[0] == 0 and proj[-1] == 0):
            contained[i] = True
        
    dfc = df.loc[contained]
    # print('OG shape:', df.shape, 'Contained shape:', dfc.shape, 'Frac=',f"{dfc.shape[0]/df.shape[0]:.3f}")
    
    return dfc