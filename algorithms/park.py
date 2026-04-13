import numpy as np
import scipy


def park(data, samplerate, low = 2, high = 30, order = 1, winsizefaktor1=30, winsizefaktor2=45, windowfaktor = 0.05, segmentlengthfaktor = 3):
        #Preprocessing using a Bandpass filter and the Wavelet Transformation
    nyquist = 0.5 * samplerate
    low = low / nyquist
    high = high / nyquist
    coeffs = scipy.signal.butter(order, [low, high], btype="band")
    data = scipy.signal.filtfilt(coeffs[0], coeffs[1], data)
    scales = np.arange(1, 100)
    wavelet_transform = scipy.signal.cwt(data, scipy.signal.ricker, scales)
    dc = wavelet_transform[0]
    sigma = np.median(np.real(dc)) / 0.6745
    t = sigma * np.sqrt(2 * np.log(len(dc)) / len(dc))
    dcdach = np.sign(dc) * np.maximum(np.abs(dc) - t, 0)
    filtered = np.sum(scipy.signal.cwt(dcdach, scipy.signal.ricker, scales), axis=0)
    winsize1 = int((winsizefaktor1/360)*samplerate)
    winsize2 = int((winsizefaktor2 / 360) * samplerate)
        #Differentation
    diff = np.diff(filtered)
    diff = np.append(diff, diff[-1])
    normdiff = diff / np.max(np.abs(diff))
        #Shannon energy of the squared differentiation
    sen = -1 * (normdiff ** 2) * np.log(diff ** 2)
    masen = np.convolve(sen, np.ones(winsize1), "same") / winsize1
    dns = np.diff(masen)
    dns = np.append(dns, dns[-1])
    normdns = dns / np.max(np.abs(dns))
        #Determination of the Peak Envelope of the Shannon energy
    pen = normdns ** 2
    mapen = np.convolve(pen, np.ones(winsize2), "same") / winsize2
    winsize = int(windowfaktor * samplerate)
    window = np.ones(winsize) / winsize
    mafiltered = np.convolve(mapen, window, mode="same")
    peakthreshold = np.zeros(len(mafiltered))
    segmentlength = segmentlengthfaktor * samplerate
        #Determining the Peak threshold
    for i in range(0,len(mafiltered),segmentlength):
        segment = mafiltered[i:i+segmentlength]
        peaks, _ = scipy.signal.find_peaks(segment)
        peaks = i + peaks
        peakthreshold[i:i+segmentlength] = np.mean(mafiltered[peaks])
    peakthreshold = np.convolve(peakthreshold, window, mode="same")
        #Sets values below the Peak threshold to 0, as they do not need to be considered in the search for R-Peaks
    mafiltered[mafiltered < peakthreshold] = 0
        #R-Peak determination
    peakcandidates, _ = scipy.signal.find_peaks(mafiltered, height=0.1*np.mean(mafiltered), distance=0.272 * samplerate)
    peaks = np.zeros(len(data))
    peaks[peakcandidates] = 1
    return peaks
