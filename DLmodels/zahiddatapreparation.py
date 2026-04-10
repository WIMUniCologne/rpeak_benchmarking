import scipy
import numpy as np


def cpsc():
    filenames = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    annotators = []
    filepath = "data/cpsc/"
    return filenames, annotators, filepath


def createtrainingdatacpsc():
    samplerate = 400
    x_set = np.asarray([])
    y_set = np.asarray([])
    filenames, annotators, filepath = cpsc()
    for i in range(0, len(filenames)):
        ecgdata = scipy.io.loadmat(filepath + "A" + filenames[i] + ".mat")
        ecgdata = ecgdata["ecg"]
        peakdata = scipy.io.loadmat(filepath + "RPN_" + filenames[i] + ".mat")
        peakdata = peakdata["R"]
        peaks = np.zeros(len(ecgdata))
        for j in range(0, len(peakdata)):
            peaks[peakdata[j]-2] = 1
            peaks[peakdata[j]-1] = 1
            peaks[peakdata[j]] = 1
            peaks[peakdata[j]+1] = 1
            peaks[peakdata[j]+2] = 1
        x_set = np.append(x_set, ecgdata)
        y_set = np.append(y_set, peaks)
    return x_set, y_set, samplerate


#createtrainingdatacpsc()
