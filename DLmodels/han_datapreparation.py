import numpy as np
import scipy

def cpsc():
    filenames = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    filepath = "data/cpsc/"
    return filenames, filepath


def preprocess_ecg(data, samplerate, low=5, high=12.5, order=2, windowlength=0.041): # 6 25 6 0.045
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
    return filtered


def createtrainingdatacpsc():
    samplerate = 400
    ecgsegments = []
    peaksegments = []
    filenames, filepath = cpsc()
    for i in filenames:
        print(i)
        ecgdata = scipy.io.loadmat(filepath + "A" + i + ".mat")
        ecgdata = ecgdata["ecg"]
        ecgdata = np.squeeze(ecgdata)
        peakdata = scipy.io.loadmat(filepath + "RPN_" + i + ".mat")
        peakdata = peakdata["R"]
        peakdata = np.squeeze(peakdata)
        peaks = np.zeros(len(ecgdata))
        for peakpoint in peakdata:
            peakpoint = int(peakpoint)
            peaks[peakpoint-2:peakpoint+2] = 1
        segmentamount = int(len(ecgdata) / 5000)
        for i in range(0, segmentamount):
            ecgpart = ecgdata[i * 5000:i * 5000 + 5000]
            peakpart = peaks[i * 5000:i * 5000 + 5000]
            if len(ecgpart) == 5000 and len(peakpart) == 5000:
                ecgsegments.append(ecgpart)
                peaksegments.append(peakpart)
            elif len(ecgpart) == len(peakpart) and len(ecgpart) < 5000:
                missingvalues = 5000 - len(ecgpart)
                ecgpart = np.pad(ecgpart, (0, missingvalues), "constant", constant_values=(0, 0))
                peakpart = np.pad(peakpart, (0, missingvalues), "constant", constant_values=(0, 0))
                ecgsegments.append(ecgpart)
                peaksegments.append(peakpart)
    ecgsegments = np.asarray(ecgsegments)
    peaksegments = np.asarray(peaksegments)
    return ecgsegments, peaksegments

