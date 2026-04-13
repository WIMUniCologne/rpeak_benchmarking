import numpy as np
import scipy


def shaik(data, samplerate, window_length = 75, step_distance = 10, windowfaktor1 = 0.3, windowfaktor2 = 0.2):
    origlength = len(data)
        #Filtering the signal using the Short-time Fourier Transformation
    f, t, comp = scipy.signal.stft(data, fs=samplerate, nperseg=window_length, noverlap=window_length - step_distance)
        #Selecting the components for the range 8 - 15 Hz
    freq_indices = np.where((f >= 8) & (f <= 15))[0]
    filtered = comp[freq_indices, :]
    #plt.show()
    sig1 = np.sqrt(np.real(filtered[0])**2)
    minimum = np.min(sig1)
    maximum = np.max(sig1)
    sig1 = (sig1 - minimum) / (maximum - minimum)   # Normalization
    #Set sig2 to sig1 if  the filtered signal contains just one frequency component
    #(an isssue that appeared for specific ecg recorcings)
    if len(filtered) == 1:
        sig2 = sig1
    else:
        sig2 = np.sqrt(np.real(filtered[1])**2)
    minimum = np.min(sig2)
    maximum = np.max(sig2)
    sig2 = (sig2 - minimum) / (maximum - minimum)
    sig1 = np.interp(np.linspace(0, len(sig1), origlength), np.arange(len(sig1)), sig1)
    sig2 = np.interp(np.linspace(0, len(sig2), origlength), np.arange(len(sig2)), sig2)
    filtereddata = [sig1, sig2]
    mindistance = 0.272
    filtered1 = filtereddata[0]
    filtered2 = filtereddata[1]
        #Peak threshold calulation for both signal components
    winsize = int(windowfaktor1 * samplerate)
    window = np.ones(winsize) / winsize
    filtered1 = np.convolve(filtered1, window, mode="same")
    peakthreshold1 = np.zeros(len(filtered1))
    segmentlength = 5 * samplerate
    for i in range(0,len(filtered1),segmentlength):
        segment = filtered1[i:i+segmentlength]
        peaks, _ = scipy.signal.find_peaks(segment)
        peaks = i + peaks
        peakthreshold1[i:i+segmentlength] = np.mean(filtered1[peaks])
    peakthreshold1 = np.convolve(peakthreshold1, window, mode="same")
    filtered2 = np.convolve(filtered2, window, mode="same")
    peakthreshold2 = np.zeros(len(filtered2))
    segmentlength = 5 * samplerate
    for i in range(0,len(filtered2),segmentlength):
        segment = filtered2[i:i+segmentlength]
        peaks, _ = scipy.signal.find_peaks(segment)
        peaks = i + peaks
        peakthreshold2[i:i+segmentlength] = np.mean(filtered2[peaks])
        #Aggregation of the signal components and the peak threshold
    peakthreshold = 0.8*peakthreshold1 + 0.8*peakthreshold2
    filtered = filtered1 + filtered2
    winsize = int(windowfaktor2 * samplerate)
    window = np.ones(winsize) / winsize
    peakthreshold = np.convolve(peakthreshold, window, mode="same")
        #Set values below threshold to 0 them from peak search
    filtered[filtered<peakthreshold] = 0
    returnedpeaks = np.zeros(len(filtered))
        #Determining the R-peaks in the filtered signal
    peakpositions, _ = scipy.signal.find_peaks(filtered, distance = mindistance*samplerate)
    returnedpeaks[peakpositions] = 1
    return returnedpeaks
