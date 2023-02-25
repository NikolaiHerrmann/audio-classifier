## Goal

This research is aimed to classify multivariate time-series to perform speaker identification. We specifically investigate speech recordings from which linear prediction cepstral coefficients (LPCCs) coefficients have been extracted. Three classifiers are being examined for this task: a simple Convolutional Neural Network (CNN) using 1D convolution, a Random Forest classifier and a Support Vector Machine. The latter two use hand-crafted features (i.e. mean, standard deviation and slope of each time-series). These classifiers were trained and tested on two different datasets, namely the Japanese Vowels dataset and the (English) Free Spoken Digit dataset. In this manner, the classifiers' performances are evaluated in two different scenarios, as the datasets vary in language, length and task. We find that the hand-crafted classifiers outperformed the neural network classifier.

## Run instructions
1) Install the Python [requirements](requirements.txt) (Python 3.10.9).
2) Run the [train.py](train.py) file:
```commandline
python train.py
```
## Japanese Vowel (/ae/) Data Set

**Size**
- 640 speaker recordings
- 9 unique speakers
- Split:
    - Train: 270, 30 recordings per speaker
    - Test: 370, 24-88 recordings per speaker

**Parameters**
- LPCCs order of 12

**Source**
- https://archive.ics.uci.edu/ml/datasets/Japanese+Vowels

## Spoken Digit Data Set

- extract all zip files in [spoken_digits](spoken_digits/)
- one recording per `txt` file

**Size**
- 3000 speaker recordings
- 6 unique speakers
- 50 recordings *of each* digit per speaker

**Parameters**
- wav files with sample rate of 8kHz (pretty low)
- recordings are trimmed, almost no silence at start/end points

**Feature Extraction**

- **LPCC** *(Linear Predictive Cepstral Coefficients)*:
    - Using `matlab_speech_features` functions to extract LPCCs
    - Source: https://github.com/jameslyons/matlab_speech_features

- Parameters:
    - Window size of 0.030 (30 ms)
    - Window step size of 0.015 (15 ms)
    - 12 cepstral coefficients (order)

- Additional MATLAB dependencies required for audio importing and analysis:
    - [DSP System Toolbox](https://www.mathworks.com/products/dsp-system.html)
    - [Audio Toolbox](https://www.mathworks.com/products/audio.html)
    - [Signal Processing Toolbox](https://www.mathworks.com/products/signal.html)


**Source**
- https://github.com/Jakobovski/free-spoken-digit-dataset