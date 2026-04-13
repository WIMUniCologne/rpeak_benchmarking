import numpy as np
import scipy
import pywt
from sklearn.cluster import KMeans


def xia(data, samplerate, low = 9, high = 25, order = 3):
        #Preprocessing: Bandpassfilter
    nyquist = 0.5 * samplerate
    low = low / nyquist
    high = high / nyquist
    coeffs = scipy.signal.butter(order, [low, high], btype="band")
    filtered = scipy.signal.filtfilt(coeffs[0], coeffs[1], data)
        #Wavelet Transformation of the filtered signal
    coeffs = pywt.wavedec(filtered, "sym8", level = 10)
    ca8, cd1, cd2, cd3, *rest = coeffs
    ca8[:], cd1[:], cd2[:], cd3[:] = 0, 0, 0, 0
    coeffs = [ca8, cd1, cd2, cd3, *rest]
    filtered = pywt.waverec(coeffs, "sym8")
    mindistance = 0.275
        #Absolute value of the slope of the signal
    slope = np.diff(filtered)
    slope = np.append(slope, slope[-1])
    absslope = np.abs(slope)
        #Clustering the absolute value of the slope into two groups
    kmeans = KMeans(n_clusters=2, n_init=10)
    absslope = absslope.reshape(-1, 1)
    classification = np.zeros(len(absslope))
    kmeans.fit(absslope)
    labels = kmeans.labels_
    zeroamount = np.count_nonzero(labels == 0)
    oneamount = len(labels) - zeroamount
        #Identification of the group related to QRS-complexes (= necessarily the smaller group)
    if oneamount > zeroamount:
        classification[labels == 1] = 0
        classification[labels == 0] = 1
    else:
        classification[labels == 1] = 1
        classification[labels == 0] = 0
        #Moving average in order to smooth the clusters assignment of the signal
    cumulation = np.zeros(len(classification))
    cumulation[0] = classification[0]
    for i in range(1, len(classification)):
        if classification[i] == 0 and cumulation[i-1] > 0:
            cumulation[i] = cumulation[i-1] - 1
        elif classification[i] == 0 and cumulation[i-1] > 0:
            cumulation[i] = 0
        elif classification[i] == 1:
            cumulation[i] = cumulation[i - 1] + 1
        #Peak identification, Clusters = R-Peak
    allpeaks, _ = scipy.signal.find_peaks(cumulation, distance = samplerate * mindistance)
    peaks = np.zeros(len(filtered))
    peaks[allpeaks] = 1
    return peaks
