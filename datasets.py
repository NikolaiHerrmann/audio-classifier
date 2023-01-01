import os

import numpy as np
from utils import uniform_scaling, RANDOM_STATE

from sklearn.model_selection import train_test_split


def get_labels(data, block_lens):
    """
    return format of X:
        - array of recordings
        - per recording:
            - each row is a recording frame (time step), beware varies! 
            - each column is a channel (always 12)
    """
    X = []
    y = []
    prev_row_idx = 0
    speaker_num = 0
    count = 0

    for i in range(data.shape[0]): # go through all rows
        
        # check each row 
        # rows of 1 indicate new recording
        if np.all(data[i,:] == 1):
            
            X.append(data[prev_row_idx:i,:])
            prev_row_idx = i + 1 # plus 1 to not include row of 1's
            
            y.append(speaker_num)
            count += 1 # count examples per speaker
            
            if count == block_lens[speaker_num]:
                count = 0
                speaker_num += 1 # next class
                
    return X, y


def get_japanese_vowels():
    data_train_vowels = np.loadtxt(os.path.join("japanese_vowels", "ae.train"))
    data_test_vowels = np.loadtxt(os.path.join("japanese_vowels", "ae.test"))
    train_block_lens = [30] * 9
    test_block_lens = [31, 35, 88, 44, 29, 24, 40, 50, 29]

    X_train_vowels, y_train_vowels = get_labels(data_train_vowels, train_block_lens)
    X_test_vowels, y_test_vowels = get_labels(data_test_vowels, test_block_lens)
    return X_train_vowels, y_train_vowels, X_test_vowels, y_test_vowels


def get_spoken_digits(is_lpcc=True):
    DIGIT_SPEAKERS = ["george", "jackson", "lucas", "nicolas", "theo", "yweweler"]
    path = os.path.join("spoken_digits", "txt_lpccs")
    X = []
    y_digit = []
    y_speaker = []

    for speaker_idx, speaker in enumerate(DIGIT_SPEAKERS):
        for digit in range(10):
            for i in range(50):
                type_ = "lpcc" if is_lpcc else "mfcc"
                filename = str(digit) + "_" + speaker + "_" + str(i) + "_" + type_ + ".txt"
                m = np.genfromtxt(os.path.join(path, filename), delimiter=',')
                X.append(m)
                y_digit.append(digit)
                y_speaker.append(speaker_idx)

    X_train_digits, X_test_digits, y_train_digits, y_test_digits = train_test_split(X, y_speaker, test_size=0.5, stratify=y_speaker, random_state=RANDOM_STATE)

    return X_train_digits, y_train_digits, X_test_digits, y_test_digits


def pre_process(X_train, X_test, rec_len):
    X_train_uni = uniform_scaling(X_train, rec_len)
    X_test_uni = uniform_scaling(X_test, rec_len)

    return X_train_uni, X_test_uni
