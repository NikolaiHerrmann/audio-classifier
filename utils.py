import os
import random

import numpy as np

import tensorflow as tf
from multiprocessing import Pool
from functools import partial

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

RANDOM_STATE = 44


def seed():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)
    os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


def stretch(rec, new_len):
    rec = rec.T
    current_len = rec.shape[1]

    new_rec = np.zeros((12, new_len))
    new_rec.fill(np.nan)

    # indices of scaled array where old values get copied to
    idxs = np.linspace(0, new_len - 1, current_len).round().astype(int)

    for i in range(12):
        new_rec[i, idxs] = rec[i, :] # copy values
        new_rec[i, :] = pd.Series(new_rec[i, :]).interpolate(method="linear") # fill-in nan values

    return new_rec.T


def uniform_scaling(X, new_len):
    """
    Stretch all recordings to length new_len using linear interpolation
    returns new copy of X with modified elements
    """
    with Pool() as pool:
        return pool.map(partial(stretch, new_len=new_len), X)


def slope_mean_std(rec, n_windows):
    feats = []
    rec_sections = np.split(rec.T, n_windows, axis=1)
        
    for section in rec_sections:
        for channel_y in section:

            x = np.arange(len(channel_y)).reshape(-1, 1)
            y_flat = channel_y.reshape(-1, 1)
            reg = LinearRegression().fit(x, y_flat)
            slope = reg.coef_[0][0]

            mean = np.mean(channel_y)
            std = np.std(channel_y)

            feats += [slope, mean, std]
            
    return feats


def extract_features(X, n_windows=3):
    """
    Extract mean, standard deviation and slope from each window
    """
    assert X[0].shape[0] % n_windows == 0, "Cannot split windows equally! " + str(X[0].shape[0])
    
    with Pool() as pool:
        return np.array(pool.map(partial(slope_mean_std, n_windows=n_windows), X))
