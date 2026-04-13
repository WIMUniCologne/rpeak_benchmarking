import numpy as np
import scipy


def pantompkins(data, samplerate, low=10.5, high=24, order=6, minheight=3.5):
    #Window for the moving average window
    windowlength = 0.256
        #Cascadation of a highpass- and lowpassfilter (Realized using a bandpass-filter)
    nyquist = 0.5 * samplerate
    low = low / nyquist
    high = high / nyquist
    coeffs = scipy.signal.butter(order, [low, high], btype="band")
    filtered = scipy.signal.filtfilt(coeffs[0], coeffs[1], data)
    b = np.array([0, -1, -2, 2, 1])
    b = b / (8/samplerate)
    filtered = scipy.signal.filtfilt(b, 1, filtered)
        #Squaring and Moving integration of the signal over the specified window length
    filtered = filtered ** 2
    windowwidth = int(windowlength * samplerate)
    movingintegrated = np.zeros(len(filtered))
    window_sum = np.sum(filtered[:windowwidth])
    movingintegrated[windowwidth - 1] = window_sum / windowwidth
    for i in range(windowwidth, len(filtered)):
        window_sum += filtered[i] - filtered[i - windowwidth]
        movingintegrated[i] = window_sum / windowwidth
    mindistance = 0.272
        #Dynamic Noise Level Estimation
    noiselevelarray = np.zeros(len(data))
    peaklevel, _ = scipy.signal.find_peaks(filtered, height= np.mean(filtered))
    i = 0
    for peak in peaklevel:
        if peak > i:
            noiselevelarray[i:peak] = 0
        i = peak
        # Set Noise peaks to zero to exclude them from peak search
    filtered[filtered < noiselevelarray] = 0
        # Actual peak search
    peaks = np.zeros(len(filtered))
    peakpositions, _ = scipy.signal.find_peaks(filtered, height = minheight * np.mean(filtered), distance= mindistance * samplerate)
    peaks[peakpositions] = 1
    return peaks
