import numpy as np
import scipy
import external_functions as external_functions

def elgendi(data, samplerate, low = 9, high = 25, order = 3, w1faktor=0.2, w2faktor=0.82, beta=0.08):
        #Bandpass and Squaring
    nyquist = 0.5 * samplerate
    low = low / nyquist
    high = high / nyquist
    coeffs = scipy.signal.butter(order, [low, high], btype="band")
    filtered = scipy.signal.filtfilt(coeffs[0], coeffs[1], data)
    squared = filtered ** 2
        #Normalization
    filtered = (filtered - np.min(filtered)) / (np.max(filtered) - np.min(filtered))
    peaks = np.zeros(len(filtered))
    w1 = int(w1faktor * samplerate)
    w2 = int(w2faktor * samplerate)
    maqrs = np.convolve(squared, np.ones(w1), mode="same") / w1
    mabeat = np.convolve(squared, np.ones(w2), mode="same") / w2
    alpha = beta * np.mean(squared)
    thr1 = mabeat + alpha
        #Determination of Blocks of Interest
    blocksofinterest = maqrs > thr1
    blocksofinterest = np.append(blocksofinterest, False)
    boi = False
    for i, boi_val in enumerate(blocksofinterest):
        if boi_val and not boi:
            boi = True
            boiarea = i
        elif not boi_val and boi:
            boi = False
            peak = boiarea + np.argmax(filtered[boiarea:i])
            peaks[peak] = 1
    foundpeaks = np.where(peaks == 1)[0]
        #Peakcorrection
    fpeliminated = []
    fpeliminated.append(foundpeaks[0])
    tolerance = 0.4
    jumpnextone = False
    for i in range(0, len(foundpeaks)-1):
        if not jumpnextone:
            dist = foundpeaks[i+1] - fpeliminated[-1]
            if dist > tolerance * samplerate:
                fpeliminated.append(foundpeaks[i])
                jumpnextone = False
            else:
                nextmax = np.max([0, i - tolerance * samplerate]) + np.argmax(filtered[foundpeaks[max(0, i - int(tolerance * samplerate))]:foundpeaks[min(len(foundpeaks) - 1, i + int(tolerance * samplerate))]])
                if np.abs(nextmax - foundpeaks[i+1]) > np.abs(nextmax - foundpeaks[i]):
                    fpeliminated.append(foundpeaks[i])
                    jumpnextone = True
                else:
                    fpeliminated.append(foundpeaks[i+1])
    correctedpeaks = np.zeros(len(filtered))
    correctedpeaks[fpeliminated] = 1
    correctedpeaks = np.zeros(len(data))
    foundpeaks = np.where(peaks == 1)[0]
    searchwindow = int(0.13275*samplerate)
    absdata = np.abs(data)
    for i in foundpeaks:
        if i - searchwindow >= 0 and i + searchwindow < len(absdata):
            locmax = i - searchwindow + np.argmax(absdata[i - searchwindow:i + searchwindow])
            correctedpeaks[locmax] = 1
    return correctedpeaks