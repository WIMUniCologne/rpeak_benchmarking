import numpy as np
import scipy


def arteagaFalconi(data, samplerate, low = 6, high = 17.5, order = 5, lrfaktor=0.018, windowlength=0.34):
        #Bandpassfilter (assumed as not specified by the paper)
    nyquist = 0.5 * samplerate
    low = low / nyquist
    high = high / nyquist
    coeffs = scipy.signal.butter(order, [low, high], btype="band")
    filtered = scipy.signal.filtfilt(coeffs[0], coeffs[1], data)
    b = np.array([0, -1, -2, 2, 1])
    b = b / (8/samplerate)
    filtered = scipy.signal.filtfilt(b, 1, filtered)
        #Determination of the (inverted) second derivative
    firstdiff = np.diff(filtered)
    secondiff = np.diff(firstdiff)
    secondiff = np.append(secondiff, secondiff[-3:-1]) #Length correction of the array
    inverted = -secondiff
    numbers = np.arange(len(inverted))
        #Sort the values
    order = np.argsort(inverted)
    sortedecgdata = inverted[order]
    sortednumbers = numbers[order].astype(int)
        #Determination of the maximum possible points related to QRS-complexes
        #using the physiological max amount of contained R-Peaks as assumed by the paper (220):
    lr = int(lrfaktor * samplerate * (220 / 60) * (len(data) / samplerate))
        #Reconstruct the original position of these points (if the values hadnt been sorted)
    positions = sortednumbers[:lr]
    qrs = np.zeros(len(data))
    peaks = np.zeros(len(data))
    qrs[positions] = 1
    window = int(windowlength * samplerate)
    start_indices = np.maximum(0, positions - window)
    end_indices = np.minimum(len(inverted), positions + window)
    max_indices = [np.argmax(inverted[start:end]) + start for start, end in zip(start_indices, end_indices)]
    peaks[max_indices] = 1
    return peaks
