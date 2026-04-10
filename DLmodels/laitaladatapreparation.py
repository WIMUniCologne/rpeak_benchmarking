import scipy
import numpy as np

def cpsc():
    filenames = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    filepath = "data/cpsc/"
    return filenames, filepath

def downsample(ecgdata, peaks, samplerate):
    newsamplerate = 250
    times = np.arange(len(ecgdata)) / samplerate
    new_times = np.arange(0, len(ecgdata) / samplerate, 1 / newsamplerate)
    cs = scipy.interpolate.CubicSpline(times, ecgdata)
    newecgdata = cs(new_times)
    newpeaks = np.zeros(len(peaks))
    for i in range(0, len(peaks)):
        if(peaks[i] == 1):
            newpeaks[int(i * newsamplerate / samplerate)-1] = 1
            newpeaks[int(i * newsamplerate / samplerate)] = 1
            newpeaks[int(i * newsamplerate / samplerate)+1] = 1
    return newecgdata, newpeaks, newsamplerate


def createtrainingdatacpsc():
    samplerate = 400
    shift =250
    segmentlength = 1000
    ecgsegments = []
    peaksegments = []
    filenames, filepath = cpsc()
    for i in filenames:
        ecgdata = scipy.io.loadmat(filepath + "A" + i + ".mat")
        ecgdata = ecgdata["ecg"]
        peakdata = scipy.io.loadmat(filepath + "RPN_" + i + ".mat")
        peakdata = peakdata["R"]
        peaks = np.zeros(len(ecgdata))
        for j in range(0, len(peakdata)):
            peaks[peakdata[j]] = 1
        newecgdata, newpeaks, newsamplerate = downsample(ecgdata, peaks, samplerate)
        segmentamount = int(len(newecgdata) / segmentlength*(segmentlength/shift))
        for i in range(0, segmentamount):
            ecgpart = newecgdata[i * shift:i * shift + 1000]
            peakpart = newpeaks[i * shift:i * shift + 1000]
            if len(ecgpart) == 1000 and len(peakpart) == 1000:
                ecgsegments.append(ecgpart)
                peaksegments.append(peakpart)
    ecgsegments = np.asarray(ecgsegments)
    peaksegments = np.asarray(peaksegments)
    return ecgsegments, peaksegments, newsamplerate

#ecgsegments, peaksegments, newsamplerate = createtrainingdatacpsc()
#print("Shapes: ecgsegments={}, peaksegments={}".format(ecgsegments.shape, peaksegments.shape))
#x_set, y_set, newsamplerate = createtrainingdatacpsc()





