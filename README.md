

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

- **MFCC** *(Mel Frequency Cepstral Coefficients)*:
    - Apparently can be better than LPCC

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