import numpy as np
import scipy


def cpsc():
    filenames = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    filepath = "data/cpsc/"
    return filenames, filepath


def downsample(ecgdata, peaks, samplerate):
    newsamplerate = 360
    times = np.arange(len(ecgdata)) / samplerate
    new_times = np.arange(0, len(ecgdata) / samplerate, 1 / newsamplerate)
    cs = scipy.interpolate.CubicSpline(times, ecgdata)
    newecgdata = cs(new_times)
    newpeaks = np.zeros(len(peaks))
    for i in range(0, len(peaks)):
        if (peaks[i] == 1):
            newpeaks[int(i * newsamplerate / samplerate)] = 1
    return newecgdata, newpeaks, newsamplerate


def preprocess_ecg(data, samplerate, ni=5):
    srd = np.diff(data)
    srd = np.append(srd, srd[-1])
    halfwindow = int(ni / 2)
    dataaveraged = data.copy()
    for i in range(ni, len(dataaveraged) - ni):
        datainwindow = data[i - halfwindow:i - halfwindow + ni]
        dataaveraged[i] = np.mean(datainwindow)
    sad = np.diff(dataaveraged)
    sad = np.append(sad, sad[-1])
    return srd, sad


def createtrainingdatacpsc():
    samplerate = 400
    window = 40
    ecgsrdsegments = []
    ecgsadsegments = []
    ispeaksegment = []
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
            peaks[peakpoint] = 1
        ecgdata, peaks, samplerate = downsample(ecgdata, peaks, samplerate)
        ecgdatasrd, ecgdatasad = preprocess_ecg(ecgdata, samplerate)
        j = 140
        nopeakamount = 0
        peakamount = 0
        shift = 10
        while j < len(ecgdata) - 140:
            begin = j
            end = begin + window
            ecgsrdpart = ecgdatasrd[j - 23:j + 33]
            maximum = np.max(ecgsrdpart)
            minimum = np.min(ecgsrdpart)
            ecgsrdpart = -1 + 2 * (ecgsrdpart - minimum) / (maximum - minimum + 0.0000001)
            ecgsadpart = ecgdatasad[j - 112:j + 168]
            maximum = np.max(ecgsadpart)
            minimum = np.min(ecgsadpart)
            ecgsadpart = -1 + 2 * (ecgsadpart - minimum) / (maximum - minimum + 0.0000001)
            if np.sum(peaks[j - shift:j + shift]) > 0:
                ispeak = 1
            else:
                ispeak = 0
            if ispeak == 0 and np.random.randint(1, 11) > 8:
                j += int(window / 2)
            else:
                if len(ecgsrdpart) == 56 and len(ecgsadpart) == 280:
                    if ispeak == 0:
                        nopeakamount += 1
                    else:
                        peakamount += 1
                    # plt.plot(ecgsrdpart)
                    # plt.plot(ecgsadpart)
                    # plt.show()
                    ecgsrdsegments.append(ecgsrdpart)
                    ecgsadsegments.append(ecgsadpart)
                    ispeaksegment.append(ispeak)
                j += int(window / 2)
        print(peakamount, nopeakamount)
    ecgsegmentsrawdiff = np.asarray(ecgsrdsegments)
    ecgsegmentsavgdiff = np.asarray(ecgsadsegments)
    peaksegments = np.asarray(ispeaksegment)
    return ecgsegmentsrawdiff, ecgsegmentsavgdiff, peaksegments

# createtrainingdatacpsc()