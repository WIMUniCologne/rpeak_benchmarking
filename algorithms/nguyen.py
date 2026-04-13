import numpy as np
import scipy


def nguyen(data, samplerate, low=9, high=11.5, order=3, windowlength=0.025, beta=2.5):
    #Cascading a highpass filter, the template filter defined by the paper and a low-pass filter
        #Highpass-Filter
    wl = int(samplerate * windowlength)
    cutoff = low
    nyquist = 0.5 * samplerate
    normalized_cutoff = cutoff / nyquist
    coeffs = scipy.signal.butter(order, normalized_cutoff, btype="high", analog=False, output="sos")
    filtered = scipy.signal.sosfiltfilt(coeffs, data)
        #Attenuation of the triangle template as defined by the paper
    templatefiltered = np.zeros_like(filtered)
    templatefiltered[wl :-wl ] = (filtered[wl :-wl ] - filtered[:-2 * wl ]) * (filtered[wl :-wl ] - filtered[wl  * 2:])
    templatefiltered[templatefiltered < 0] = 0
    cutoff = high
    normalized_cutoff = cutoff / nyquist
        #Lowpass-Filter
    coeffs = scipy.signal.butter(order, normalized_cutoff, btype="low", analog=False, output = "sos")
    mindistance = 0.272
    filtered = scipy.signal.sosfiltfilt(coeffs, templatefiltered)
        #Threshold-based peak identification
    foundpeaks, _ = scipy.signal.find_peaks(filtered, height = beta * np.mean(filtered),distance=int(samplerate * mindistance))
    peaks = np.zeros(len(filtered))
    peaks[foundpeaks] = 1
    return peaks
