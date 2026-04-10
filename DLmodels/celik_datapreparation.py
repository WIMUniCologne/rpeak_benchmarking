import pywt
import numpy as np
import scipy


def cpsc():
    filenames = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    filepath = "data/cpsc/"
    return filenames, filepath


def createtrainingdatacpsc():
    samplerate = 400
    windowlength = 4 * samplerate
    datasegments = []
    peaksegments = []
    filenames, filepath = cpsc()
    for i in filenames:
        ecgdata = scipy.io.loadmat(filepath + "A" + i + ".mat")
        ecgdata = ecgdata["ecg"]
        ecgdata = np.squeeze(ecgdata)
        peakdata = scipy.io.loadmat(filepath + "RPN_" + i + ".mat")
        peakdata = peakdata["R"]
        peakdata = np.squeeze(peakdata)
        peaks = np.zeros(len(ecgdata))
        peaks[peakdata] = 1
        length = len(ecgdata)
        ecgdata = ecgdata[0:int(0.5 * length)]
        peaks = peaks[0:int(0.5 * length)]
        for j in range(0, len(ecgdata), windowlength):
            segment = ecgdata[j: j + windowlength]
            peaksegment = peaks[j: j + windowlength]
            if len(segment) < windowlength:
                pad_length = windowlength - len(segment)
                segment = np.pad(segment, (0, pad_length), "constant")
                peaksegment = np.pad(peaksegment, (0, pad_length), "constant")
            downsizenumber = 512
            origsegmentlength = len(segment)
            downsampledecg = scipy.signal.resample(segment, downsizenumber)
            downsizedsegmentlength = len(downsampledecg)
            downsampledecg = wavelettransformation(downsampledecg, samplerate)
            oneindexe = np.where(peaksegment == 1)[0]
            oneindexe = (downsizedsegmentlength * oneindexe / origsegmentlength).astype(int)
            downsampledpeaks = np.zeros(512)
            for k in oneindexe:
                downsampledpeaks[k - 10:k + 10] = int(1)
            downsampledpeaks = np.tile(downsampledpeaks, (16, 1))
            datasegments.append(downsampledecg)
            peaksegments.append(downsampledpeaks)
    datasegments = np.asarray(datasegments)
    peaksegments = np.asarray(peaksegments)
    return datasegments, peaksegments


def wavelettransformation(data, samplerate):
    coefficients, frequencies = pywt.cwt(data, np.arange(1, 100, 0.8), "morl")
    frequencies = frequencies * samplerate
    filtered_coefficients = np.real(coefficients[(frequencies >= 16.66) & (frequencies <= 47.13)])
    maximum = np.max(filtered_coefficients)
    minimum = np.min(filtered_coefficients)
    if maximum - minimum != 0:
        filtered_coefficients = filtered_coefficients - minimum / (maximum - minimum)
    print(filtered_coefficients.shape)
    return filtered_coefficients
