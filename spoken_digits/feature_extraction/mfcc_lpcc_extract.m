function [mfcc, lpcc] = mfcc_lpcc_extract(filename)
    order = 12;
    windowLength = 0.03;
    windowStep = 0.015;

    [audioIn, fs] = audioread(filename);

    S = stft(audioIn, "Window", hann(round(windowLength * fs), ...
        "periodic"), "OverlapLength", round(windowStep * fs), ...
        "FrequencyRange", "onesided");
    S = abs(S);
    filterBank = designAuditoryFilterBank(fs, 'FFTLength', round(windowLength * fs));
    melSpec = filterBank * S;
    mfcc = cepstralCoefficients(melSpec, 'NumCoeffs', order);

    lpcc = msf_lpcc(audioIn, fs, 'order', order, 'winlen', ...
        windowLength, 'winstep', windowStep);

    % plot(mfcc);
    % figure;
    % plot(lpcc);
    % figure
end