import numpy as np
import scipy


def xu(data, samplerate, lowcut = 8, highcut = 25, order = 3, windowwidthfaktor = 49):
    shannonwindowwidth = int(samplerate * windowwidthfaktor / 360)
        #Bandpassfilter if the signal
    filtercoeffs = scipy.signal.cheby1(order, 1, [lowcut, highcut], btype="band", fs=samplerate)
    filtereddata = scipy.signal.lfilter(filtercoeffs[0], filtercoeffs[1], data)
        #Differentiation
    diff = np.diff(filtereddata)
    diff = np.append(diff, diff[-1])    #Length correction of the array after differentation
    normdiff = diff / np.max(np.abs(diff))  #Normalization
        #Formula for the shannon energy as specified in the paper, followed by determining the peak envelope
    shannonenergy = (-(normdiff ** 2)) * np.log(normdiff ** 2)
    window = np.ones(shannonwindowwidth) / (shannonwindowwidth)
    see = np.convolve(shannonenergy, window, mode="same")
    filtercoeffs = scipy.signal.butter(1, 2 / (0.5 * samplerate), btype="low", analog=False, output="ba")
    see = scipy.signal.filtfilt(filtercoeffs[0], filtercoeffs[1], see)
        #Hilbert Transformation
    htsignal = np.angle(scipy.signal.hilbert(see))
        #Determining negative-to-positive zero-crossings
    peaks_bool = (htsignal[:-1] < 0) & (htsignal[1:] >= 0)
    peaks = np.zeros(len(htsignal))
    peaks[1:] = peaks_bool.astype(int)
    return peaks
