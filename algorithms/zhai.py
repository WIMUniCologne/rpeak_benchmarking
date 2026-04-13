import numpy as np
import scipy


def zhai(data, samplerate, low=8, high=12, order=4, windowlength=0.038, windowwidth=0.44):
        #Bandpassfilter (By cascading a high- and lowpass-filter)
    windowlength = int(samplerate * windowlength)
    cutoff = low
    nyquist = 0.5 * samplerate
    normalized_cutoff = cutoff / nyquist
    coeffs = scipy.signal.butter(order, normalized_cutoff, btype="high", analog=False, output="sos")
    filtered = scipy.signal.sosfiltfilt(coeffs, data)
    templatefiltered = np.zeros_like(filtered)
    templatefiltered[windowlength:-windowlength] = (filtered[windowlength:-windowlength] - filtered[:-2 * windowlength]) * (filtered[windowlength:-windowlength] - filtered[windowlength * 2:])
    templatefiltered[templatefiltered < 0] = 0
    cutoff = high
    normalized_cutoff = cutoff / nyquist
    coeffs = scipy.signal.butter(order, normalized_cutoff, btype="low", analog=False, output = "sos")
    filtered = scipy.signal.sosfiltfilt(coeffs, templatefiltered)
    mindistance = 0.272
    windowwidth = int(samplerate*windowwidth)
        #First phase: Identifying areas with QRS-complexes
    li = filtered ** 2
    minimum = np.min(li)
    maximum = np.max(li)
    li = (li - minimum) / (maximum - minimum)
    limean = np.mean(li)
    lipeaks, _ = scipy.signal.find_peaks(li, height=limean, distance = mindistance * samplerate)
    qrsbegin = lipeaks.copy()
    qrsbegin = qrsbegin - int(windowwidth/2)
        #Creation of the template
    template = []
    for begin in qrsbegin:
        if begin >= 0 and begin + windowwidth < len(filtered):
            template.append(filtered[begin:begin+windowwidth])
    qrstemplate = np.asarray(template)
    qrstemplate = np.mean(qrstemplate, axis=0)
        #Cross-correlation of the filtered data with the template
    crosscorelation = scipy.signal.correlate(filtered, qrstemplate, mode="full")
    shift = int((len(crosscorelation) - len(filtered)) / 2)
    crosscorelation = crosscorelation[shift:len(filtered) + shift]
    minpeaknumber = int(len(crosscorelation) / (samplerate * 0.3))
    sortiert = np.sort(crosscorelation)[::-1]
    peakestimate = sortiert[minpeaknumber]
        #R-Peaks = Areas where the crosscorelation exceeds a certain threshold
    foundpeaks, _ = scipy.signal.find_peaks(crosscorelation, height=0.1*peakestimate, distance=mindistance * samplerate)
    peaks = np.zeros(len(filtered))
    peaks[foundpeaks] = 1
    return peaks
