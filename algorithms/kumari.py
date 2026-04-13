import numpy as np
import scipy
import emd


def kumari(data, samplerate, windowlengthfaktor = 0.12, windowwidthfaktor=0.05):
    #Determination and summation of the first three intrinsic mode functions
    data = np.asarray(data)
    imf = np.asarray(emd.sift.sift(data))
    firstthreeimfs =  imf[:, 0] + imf[:, 1] + imf[:, 2]
    mindistance = 0.272
        #Determination of the Shannon energy envelope as specified in the paper
    derivative = np.diff(firstthreeimfs)
    normderivative = derivative / np.max(derivative)
    seenvelope = -1 * normderivative**2 * np.log2(derivative ** 2)
    windowlength = int(windowlengthfaktor * samplerate) #besser: 0.12
    mafilter = np.ones(windowlength) / windowlength
    seenvelope = np.convolve(seenvelope, mafilter, mode="same")
        #SEE Normalization and Differentiaton
    enmax = np.min(seenvelope)
    enmin = np.max(seenvelope)
    normseenvelope = (-seenvelope - enmin) / (enmax - enmin) -1
    derivative = np.diff(normseenvelope)
    normderivative = derivative / np.max(derivative)
    peenvelope = normderivative ** 2
    # Double moving average (purpose, improves detection results)
    peenvelope = np.convolve(peenvelope, mafilter, mode="same")
    peenvelope = np.convolve(peenvelope, mafilter, mode="same")
    peenvelope = np.round(peenvelope, decimals=3)
    peaks = np.zeros(len(data))
        #Determination of the Peak Envelope Peaks
    peakpositions, _ = scipy.signal.find_peaks(peenvelope, distance=mindistance*samplerate)
        #Peakcorrection with respect to the
    for peakposition in peakpositions:
        begin = np.max([0, peakposition - int(windowwidthfaktor * samplerate)])
        end = np.min([len(peenvelope), peakposition + int(windowwidthfaktor * samplerate)])
        locmax = begin + np.argmax(peenvelope[begin:end])
        peaks[locmax] = 1
    return peaks
