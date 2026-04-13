import numpy as np
import scipy
import external_functions as external_functions


def hamilton(data, samplerate, low = 10.5, high = 24, order = 2, windowwidth = 0.08):
    mindistance = 0.272
    # Bandpass filter
    nyquist_freq = samplerate / 2
    low_cutoff = low / nyquist_freq
    high_cutoff = high / nyquist_freq
    coeffs = scipy.signal.butter(order, [low_cutoff, high_cutoff], btype="band")
    filtered = scipy.signal.filtfilt(coeffs[0], coeffs[1], data)
    #Differentiation, Squaring and Movingaverage
    diffsignal = np.diff(filtered)
    squared = diffsignal ** 2
    movavg = np.convolve(squared, np.ones(int(windowwidth*samplerate)) / windowwidth*samplerate, mode="same")
    filtered = np.append(movavg, movavg[-1]) # Length Correctíon after differentiation
    # Find initial peaklocations
    peaklocations, _ = scipy.signal.find_peaks(filtered, distance=mindistance * samplerate)
    # External threshold function
    qrsindex = external_functions.sedghamiz_thresholding(filtered, peaklocations, samplerate).astype(int)
    foundpeaks = np.zeros((len(filtered)))
    for i in range(0, len(qrsindex)):
        if qrsindex[i] < len(filtered):
            foundpeaks[qrsindex[i]] = 1
    return foundpeaks
