import numpy as np
import scipy
import emd
import pywt
from sklearn.cluster import KMeans
import external_functions as external_functions
import keras


'''
This file contains all implemented algorithms listed in chapter 4.2.1, based on the corresponding papers. 

The file further contains
- the MIT Long Term Database-specific configuration for these algorithms (related to chapter 4.3.2)
- the two voting ensemble approaches (related to chapter 4.3.3)
- the function that makes a noise-level dependent choice between a certain algorithm as default 
      and the HAN RNN which turned out the be the most noise-robust in this study     

- All functions take a numpy array data and an int value samplerate as required parameters, and return a 
      numpy array of the length of data that contain a one at spots where an R-Peak is detected and that
      is filled with zeros otherwise 
      
'''


### Classical Methods (Filter Stage + Decision Stage), by order of obtained overall performance:

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


def zhai(data, samplerate, low=8, high=12, order=4, windowlength=0.038, windowwidth=0.44):
        #Bandpassfilter (By cascading a high- and lowpass-filter)
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
    mindistance = 0.272
    windowwidth = int(samplerate*windowwidth)
        #First phase: Identifying areas with QRS-complexes
    li = filtered ** 2
    minimum = np.min(li)
    maximum = np.max(li)
    li = (li - minimum) / (maximum - minimum)
    limean = np.mean(li)
    lipeaks, _ = scipy.signal.find_peaks(li, height=limean, distance = mindistance * samplerate)
    qrsbegin = lipeaks.copy()
    qrsbegin = qrsbegin - int(windowwidth/2)
        #Creation of the template
    template = []
    for begin in qrsbegin:
        if begin >= 0 and begin + windowwidth < len(filtered):
            template.append(filtered[begin:begin+windowwidth])
    qrstemplate = np.asarray(template)
    qrstemplate = np.mean(qrstemplate, axis=0)
        #Cross-correlation of the filtered data with the template
    crosscorelation = scipy.signal.correlate(filtered, qrstemplate, mode="full")
    shift = int((len(crosscorelation) - len(filtered)) / 2)
    crosscorelation = crosscorelation[shift:len(filtered) + shift]
    minpeaknumber = int(len(crosscorelation) / (samplerate * 0.3))
    sortiert = np.sort(crosscorelation)[::-1]
    peakestimate = sortiert[minpeaknumber]
        #R-Peaks = Areas where the crosscorelation exceeds a certain threshold
    foundpeaks, _ = scipy.signal.find_peaks(crosscorelation, height=0.1*peakestimate, distance=mindistance * samplerate)
    peaks = np.zeros(len(filtered))
    peaks[foundpeaks] = 1
    return peaks


def xia(data, samplerate, low = 9, high = 25, order = 3):
        #Preprocessing: Bandpassfilter
    nyquist = 0.5 * samplerate
    low = low / nyquist
    high = high / nyquist
    coeffs = scipy.signal.butter(order, [low, high], btype="band")
    filtered = scipy.signal.filtfilt(coeffs[0], coeffs[1], data)
        #Wavelet Transformation of the filtered signal
    coeffs = pywt.wavedec(filtered, "sym8", level = 10)
    ca8, cd1, cd2, cd3, *rest = coeffs
    ca8[:], cd1[:], cd2[:], cd3[:] = 0, 0, 0, 0
    coeffs = [ca8, cd1, cd2, cd3, *rest]
    filtered = pywt.waverec(coeffs, "sym8")
    mindistance = 0.275
        #Absolute value of the slope of the signal
    slope = np.diff(filtered)
    slope = np.append(slope, slope[-1])
    absslope = np.abs(slope)
        #Clustering the absolute value of the slope into two groups
    kmeans = KMeans(n_clusters=2, n_init=10)
    absslope = absslope.reshape(-1, 1)
    classification = np.zeros(len(absslope))
    kmeans.fit(absslope)
    labels = kmeans.labels_
    zeroamount = np.count_nonzero(labels == 0)
    oneamount = len(labels) - zeroamount
        #Identification of the group related to QRS-complexes (= necessarily the smaller group)
    if oneamount > zeroamount:
        classification[labels == 1] = 0
        classification[labels == 0] = 1
    else:
        classification[labels == 1] = 1
        classification[labels == 0] = 0
        #Moving average in order to smooth the clusters assignment of the signal
    cumulation = np.zeros(len(classification))
    cumulation[0] = classification[0]
    for i in range(1, len(classification)):
        if classification[i] == 0 and cumulation[i-1] > 0:
            cumulation[i] = cumulation[i-1] - 1
        elif classification[i] == 0 and cumulation[i-1] > 0:
            cumulation[i] = 0
        elif classification[i] == 1:
            cumulation[i] = cumulation[i - 1] + 1
        #Peak identification, Clusters = R-Peak
    allpeaks, _ = scipy.signal.find_peaks(cumulation, distance = samplerate * mindistance)
    peaks = np.zeros(len(filtered))
    peaks[allpeaks] = 1
    return peaks


def arteagaFalconi(data, samplerate, low = 6, high = 17.5, order = 5, lrfaktor=0.018, windowlength=0.34):
        #Bandpassfilter (assumed as not specified by the paper)
    nyquist = 0.5 * samplerate
    low = low / nyquist
    high = high / nyquist
    coeffs = scipy.signal.butter(order, [low, high], btype="band")
    filtered = scipy.signal.filtfilt(coeffs[0], coeffs[1], data)
    b = np.array([0, -1, -2, 2, 1])
    b = b / (8/samplerate)
    filtered = scipy.signal.filtfilt(b, 1, filtered)
        #Determination of the (inverted) second derivative
    firstdiff = np.diff(filtered)
    secondiff = np.diff(firstdiff)
    secondiff = np.append(secondiff, secondiff[-3:-1]) #Length correction of the array
    inverted = -secondiff
    numbers = np.arange(len(inverted))
        #Sort the values
    order = np.argsort(inverted)
    sortedecgdata = inverted[order]
    sortednumbers = numbers[order].astype(int)
        #Determination of the maximum possible points related to QRS-complexes
        #using the physiological max amount of contained R-Peaks as assumed by the paper (220):
    lr = int(lrfaktor * samplerate * (220 / 60) * (len(data) / samplerate))
        #Reconstruct the original position of these points (if the values hadnt been sorted)
    positions = sortednumbers[:lr]
    qrs = np.zeros(len(data))
    peaks = np.zeros(len(data))
    qrs[positions] = 1
    window = int(windowlength * samplerate)
    start_indices = np.maximum(0, positions - window)
    end_indices = np.minimum(len(inverted), positions + window)
    max_indices = [np.argmax(inverted[start:end]) + start for start, end in zip(start_indices, end_indices)]
    peaks[max_indices] = 1
    return peaks


def nguyen(data, samplerate, low=9, high=11.5, order=3, windowlength=0.025, beta=2.5):
    #Cascading a highpass filter, the template filter defined by the paper and a low-pass filter
        #Highpass-Filter
    wl = int(samplerate * windowlength)
    cutoff = low
    nyquist = 0.5 * samplerate
    normalized_cutoff = cutoff / nyquist
    coeffs = scipy.signal.butter(order, normalized_cutoff, btype="high", analog=False, output="sos")
    filtered = scipy.signal.sosfiltfilt(coeffs, data)
        #Attenuation of the triangle template as defined by the paper
    templatefiltered = np.zeros_like(filtered)
    templatefiltered[wl :-wl ] = (filtered[wl :-wl ] - filtered[:-2 * wl ]) * (filtered[wl :-wl ] - filtered[wl  * 2:])
    templatefiltered[templatefiltered < 0] = 0
    cutoff = high
    normalized_cutoff = cutoff / nyquist
        #Lowpass-Filter
    coeffs = scipy.signal.butter(order, normalized_cutoff, btype="low", analog=False, output = "sos")
    mindistance = 0.272
    filtered = scipy.signal.sosfiltfilt(coeffs, templatefiltered)
        #Threshold-based peak identification
    foundpeaks, _ = scipy.signal.find_peaks(filtered, height = beta * np.mean(filtered),distance=int(samplerate * mindistance))
    peaks = np.zeros(len(filtered))
    peaks[foundpeaks] = 1
    return peaks


def pantompkins(data, samplerate, low=10.5, high=24, order=6, minheight=3.5):
    #Window for the moving average window
    windowlength = 0.256
        #Cascadation of a highpass- and lowpassfilter (Realized using a bandpass-filter)
    nyquist = 0.5 * samplerate
    low = low / nyquist
    high = high / nyquist
    coeffs = scipy.signal.butter(order, [low, high], btype="band")
    filtered = scipy.signal.filtfilt(coeffs[0], coeffs[1], data)
    b = np.array([0, -1, -2, 2, 1])
    b = b / (8/samplerate)
    filtered = scipy.signal.filtfilt(b, 1, filtered)
        #Squaring and Moving integration of the signal over the specified window length
    filtered = filtered ** 2
    windowwidth = int(windowlength * samplerate)
    movingintegrated = np.zeros(len(filtered))
    window_sum = np.sum(filtered[:windowwidth])
    movingintegrated[windowwidth - 1] = window_sum / windowwidth
    for i in range(windowwidth, len(filtered)):
        window_sum += filtered[i] - filtered[i - windowwidth]
        movingintegrated[i] = window_sum / windowwidth
    mindistance = 0.272
        #Dynamic Noise Level Estimation
    noiselevelarray = np.zeros(len(data))
    peaklevel, _ = scipy.signal.find_peaks(filtered, height= np.mean(filtered))
    i = 0
    for peak in peaklevel:
        if peak > i:
            noiselevelarray[i:peak] = 0
        i = peak
        # Set Noise peaks to zero to exclude them from peak search
    filtered[filtered < noiselevelarray] = 0
        # Actual peak search
    peaks = np.zeros(len(filtered))
    peakpositions, _ = scipy.signal.find_peaks(filtered, height = minheight * np.mean(filtered), distance= mindistance * samplerate)
    peaks[peakpositions] = 1
    return peaks


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

def hamilton(data, samplerate, low = 10.5, high = 24, order = 2, windowwidth = 0.08):
    mindistance = 0.272
    # Bandpass filter
    nyquist_freq = samplerate / 2
    low_cutoff = low / nyquist_freq
    high_cutoff = high / nyquist_freq
    coeffs = scipy.signal.butter(order, [low_cutoff, high_cutoff], btype="band")
    filtered = scipy.signal.filtfilt(coeffs[0], coeffs[1], data)
    #Differentiation, Squaring and Movingaverage
    diffsignal = np.diff(filtered)
    squared = diffsignal ** 2
    movavg = np.convolve(squared, np.ones(int(windowwidth*samplerate)) / windowwidth*samplerate, mode="same")
    filtered = np.append(movavg, movavg[-1]) # Length Correctíon after differentiation
    # Find initial peaklocations
    peaklocations, _ = scipy.signal.find_peaks(filtered, distance=mindistance * samplerate)
    # External threshold function
    qrsindex = external_functions.sedghamiz_thresholding(filtered, peaklocations, samplerate).astype(int)
    foundpeaks = np.zeros((len(filtered)))
    for i in range(0, len(qrsindex)):
        if qrsindex[i] < len(filtered):
            foundpeaks[qrsindex[i]] = 1
    return foundpeaks


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


### Models that apply deep learning


def zahid(data, samplerate, segmentsize = 5000, overlap = 2, probabilityfaktor = 0.1, winsizefaktor = 0.1):
    model = keras.models.load_model("DLmodels/zahidmodell_withf1.h5", custom_objects={"F1Score": external_functions.F1Score})
    origlength = len(data)
    if samplerate != 400:   #Samplerate Correction with respect to the samplerate of the trianing data
        newsamplerate = 400
        times = np.arange(len(data)) / samplerate
        new_times = np.arange(0, len(data) / samplerate, 1 / newsamplerate)
        cs = scipy.interpolate.CubicSpline(times, data)
        data = cs(new_times)
    newlength = len(data)
    peaks = np.zeros(origlength)
    def arraysegmenter(data, segmentsize, overlap):
        ecgparts = []
        shift = int(segmentsize / overlap)
        i = 0
        while i < len(data):
            part = data[i:i + segmentsize]
            if len(part) < segmentsize:
                zeros = segmentsize - len(part)
                part = np.concatenate((part, np.zeros(zeros)))
            maximum = np.max(part)
            minimum = np.min(part)
            normalized = -1 + 2 * (part - minimum) / (maximum - minimum)
            ecgparts.append(normalized)
            i = i + shift
        return ecgparts, shift
    segments, shift = arraysegmenter(data, segmentsize, overlap)
    segments = np.asarray(segments)
    peaksegments = model.predict(segments)
    segmentaddition = np.zeros(newlength)
    alreadydone = np.zeros(newlength)
    for i in range(0, len(peaksegments)):
        peaksegment = peaksegments[i][0:np.min([len(peaksegments[i]), segmentsize])].flatten()
        peaksegmentsize = len(peaksegment)
        if i*shift+peaksegmentsize < len(segmentaddition):
            segmentaddition[i*shift:i*shift+peaksegmentsize] += peaksegment
            alreadydone[i*shift:i*shift+peaksegmentsize] += 1
        else:
            allowedlength = len(segmentaddition) - i*shift
            segmentaddition[i * shift:i * shift + allowedlength] += peaksegment[0:allowedlength]
            alreadydone[i * shift:i * shift + allowedlength] += 1
    alreadydone[alreadydone == overlap] = -1
    alreadydone = np.reciprocal(alreadydone) * overlap
    alreadydone[alreadydone < 0] = 1
    segmentaddition = segmentaddition * alreadydone
    winsize = int(winsizefaktor * samplerate) #0.15
    window = np.ones(winsize) / winsize
    segmentaddition = np.convolve(segmentaddition, window, mode="same")
    minpeaknumber = int(len(segmentaddition) / (samplerate * 0.3))
    sortiert = np.sort(segmentaddition)[::-1]
    peakestimate = sortiert[minpeaknumber]
    foundpeaks, _ = scipy.signal.find_peaks(segmentaddition, height = probabilityfaktor * peakestimate, distance = 0.272 * samplerate)
    # Correction of the output with respect to the original length of the data
    foundpeaks = (origlength * foundpeaks / len(segmentaddition)).astype(int)
    peaks[foundpeaks] = 1
    return peaks


def laitala(data, samplerate, low = 8, high = 15, order = 1, segmentsize = 1000, overlap = 2, probabilityfaktor = 0.1):
    model = keras.models.load_model("DLmodels/laitalamodell_withf1.h5", custom_objects={"F1Score": external_functions.F1Score})
    nyquist = 0.5 * samplerate
    low = low / nyquist
    high = high / nyquist
    coeffs = scipy.signal.butter(order, [low, high], btype="band")
    prepareddata = scipy.signal.filtfilt(coeffs[0], coeffs[1], data)
    origlength = len(data)
    newsamplerate = 250
    times = np.arange(len(prepareddata)) / samplerate
    new_times = np.arange(0, len(prepareddata) / samplerate, 1 / newsamplerate)
    cs = scipy.interpolate.CubicSpline(times, prepareddata)
    prepareddata = cs(new_times)
    newlength = len(prepareddata)
    peaks = np.zeros(origlength)
    def arraysegmenter(data, segmentsize, overlap):
        ecgparts = []
        shift = int(segmentsize / overlap)
        i = 0
        while i < len(data):
            part = data[i:i + segmentsize]
            if len(part) < segmentsize:
                zeros = segmentsize - len(part)
                part = np.concatenate((part, np.zeros(zeros)))
            ecgparts.append(part)
            i = i + shift
        return ecgparts, shift
    segments, shift = arraysegmenter(prepareddata, segmentsize, overlap)
    segments = np.asarray(segments)
    peaksegments = model.predict(segments)
    segmentaddition = np.zeros(newlength)
    alreadydone = np.zeros(newlength)
    for i in range(0, len(peaksegments)):
        peaksegment = peaksegments[i][0:np.min([len(peaksegments[i]), segmentsize])].flatten()
        peaksegmentsize = len(peaksegment)
        if i*shift+peaksegmentsize < len(segmentaddition):
            segmentaddition[i*shift:i*shift+peaksegmentsize] += peaksegment
            alreadydone[i*shift:i*shift+peaksegmentsize] += 1
        else:
            allowedlength = len(segmentaddition) - i*shift
            segmentaddition[i * shift:i * shift + allowedlength] += peaksegment[0:allowedlength]
            alreadydone[i * shift:i * shift + allowedlength] += 1
    alreadydone[alreadydone == overlap] = -1
    alreadydone = np.reciprocal(alreadydone) * overlap
    alreadydone[alreadydone < 0] = 1
    segmentaddition = segmentaddition * alreadydone
    winsize = int(0.1*newsamplerate)
    window = np.ones(winsize) / winsize
    segmentaddition = np.convolve(segmentaddition, window, mode="same")
    minimum = np.min(segmentaddition)
    maximum = np.max(segmentaddition)
    segmentaddition = (segmentaddition - minimum) / (maximum - minimum)
        # Determination of the minimum peak number
    minpeaknumber = int(len(segmentaddition)/(newsamplerate * 0.3))
    #print(minpeaknumber)
    sortiert = np.sort(segmentaddition)[::-1]
        # Estimation of the probability threshold
    peakestimate = sortiert[minpeaknumber]
    #height = 1 ohne movingaverage
    foundpeaks, _ = scipy.signal.find_peaks(segmentaddition, height = probabilityfaktor*peakestimate, distance = 0.272 * newsamplerate)
    # Correction of the output with respect to the original length of the data
    foundpeaks = (samplerate * foundpeaks / newsamplerate).astype(int)
    peaks[foundpeaks] = 1
    return peaks


def xiang(data, samplerate, center = 20, segmentsize = 56, overlap = 15, probabilityfaktor = 0.8):
    model = keras.models.load_model("DLmodels/xiangmodell_withf1_13.h5", custom_objects={"F1Score": external_functions.F1Score})
    windowsize = 10#int(samplerate / 10)
        #Preprocessing
    origlength = len(data)
        #Samplerate correction, 360 as specified by the paper (training data was therefore downsampled)
        #in order to keep the window length values specified by the paper
    if samplerate != 360:
        newsamplerate = 360
        times = np.arange(origlength) / samplerate
        new_times = np.arange(0, origlength / samplerate, 1 / newsamplerate)
        cs = scipy.interpolate.CubicSpline(times, data)
        data = cs(new_times)
        #Differentation of the filtered data = srd
    #windowsize = int(samplerate/10)
    srd = np.diff(data)
    srd = np.append(srd, srd[-1])   #Length correction after differentiation
    window = np.ones(windowsize) / windowsize
        #Differentation of the moving averaged filtered data = sad
    dataaveraged = np.convolve(data, window, mode="same")
    sad = np.diff(dataaveraged)
    sad = np.append(sad, sad[-1])   #Length correction after differentiation
    def arraysegmenter(srd, sad, segmentsize, overlap):
        srdparts = []
        sadparts = []
        shift = int(segmentsize / overlap)
        i = 112
        while i < len(data)-168:
            srdpart = srd[i-23:i + 33]
            sadpart = sad[i-112:i + 168]
            if len(srdpart) < 56:
                zeros = 56 - len(srdpart)
                srdpart = np.concatenate((srdpart, np.zeros(zeros)))
            if len(sadpart) < 280:
                zeros = 280 - len(srdpart)
                sadpart = np.concatenate((sadpart, np.zeros(zeros)))
            maximum = np.max(srdpart)
            minimum = np.min(srdpart)
            normalized = -1 + 2 * (srdpart - minimum) / (maximum - minimum + 0.0000001)
            srdparts.append(normalized)
            maximum = np.max(sadpart)
            minimum = np.min(sadpart)
            normalized = -1 + 2 * (sadpart - minimum) / (maximum - minimum + 0.0000001)
            sadparts.append(normalized)
            i = i + shift
        return srdparts, sadparts, shift
        #Segmentation into Object-Level and Part-Level
    srdparts, sadparts, shift = arraysegmenter(srd, sad, segmentsize, overlap)
    srdparts = np.asarray(srdparts)
    sadparts = np.asarray(sadparts)
    peaksegments = model.predict([srdparts, sadparts]).flatten()
    segmentaddition = np.zeros(len(data))
    for i in range(0, len(peaksegments)):
        if i*shift+center < len(segmentaddition):
            segmentaddition[i*shift+112] = peaksegments[i]
    window = np.ones(40) / 40
    segmentaddition = np.convolve(segmentaddition, window, mode="same")
    print(len(segmentaddition))
        #Meanvalue for probability threshold
    meanvalue = np.mean(segmentaddition)
    peaks = np.zeros(origlength)
    print(len(peaks))
    foundpeaks, _ = scipy.signal.find_peaks(segmentaddition, height = probabilityfaktor * meanvalue ,distance = 0.272 * 360)
        # Correction of the output with respect to the original length of the data
    foundpeaks = (foundpeaks * origlength / len(data)).astype(int)
    peaks[foundpeaks] = 1
    return peaks


def celik(data, samplerate, low = 8, high = 50, order = 1, overlap = 2, probabilityfaktor = 0.01):
    origlength = len(data)
    mindistance = 0.275
    def wavelettransformation(data, samplerateadjustment = 0.8):
        coefficients, frequencies = pywt.cwt(data, np.arange(1, 100, samplerateadjustment), "morl")
        frequencies = frequencies * 400
        filtered_coefficients = np.real(coefficients[(frequencies >= 16.66) & (frequencies <= 47.13)])
        maximum = np.max(filtered_coefficients)
        minimum = np.min(filtered_coefficients)
        if maximum - minimum != 0:
            filtered_coefficients = filtered_coefficients - minimum / (maximum - minimum)
        return filtered_coefficients
    #Samplerate correction with respect to the samplerate of the training data (not if dataspecific)
    newsamplerate = 400
    times = np.arange(len(data)) / samplerate
    new_times = np.arange(0, len(data) / samplerate, 1 / newsamplerate)
    cs = scipy.interpolate.CubicSpline(times, data)
    data = cs(new_times)
        #Bandpassfilter
    nyquist = 0.5 * samplerate
    low = low / nyquist
    high = high / nyquist
    coeffs = scipy.signal.butter(order, [low, high], btype="band")
    prepareddata = scipy.signal.filtfilt(coeffs[0], coeffs[1], data)
    model = keras.models.load_model("DLmodels/celikmodell_withf1.h5", custom_objects={"F1Score": external_functions.F1Score})
        #Downsizing as specified in the paper
    downsizenumber = int(0.32 * len(prepareddata))
    downsampledecg = scipy.signal.resample(prepareddata, downsizenumber)
    newlength = len(downsampledecg)
    lengthfaktor = origlength / newlength
    def arraysegmenter(data, overlap):
        segmentsize = 512
        ecgparts = []
        shift = int(segmentsize / overlap)
        i = 0
        while i < len(data):
            part = data[i:i + segmentsize]
            if len(part) < segmentsize:
                zeros = segmentsize - len(part)
                part = np.concatenate((part, np.zeros(zeros)))
            ecgparts.append(wavelettransformation(part))
            i = i + shift
        ecgparts = np.asarray(ecgparts)
        return ecgparts, shift
        #Creating 2d segments containing wavelet coefficients in the range 16.66 - 47.13
    ecgparts, shift = arraysegmenter(downsampledecg, overlap)
    peaksegments = model.predict(ecgparts)
    segmentaddition = np.zeros(newlength)
    alreadydone = np.zeros(newlength)
    for i in range(0, len(peaksegments)):
        peaksegment = np.sum(peaksegments[i], axis=0) / 16
        peaksegment = np.squeeze(peaksegment)
        peaksegmentsize = len(peaksegment)
        if i*shift+peaksegmentsize < len(segmentaddition):
            segmentaddition[i*shift:i*shift+peaksegmentsize] += peaksegment
            alreadydone[i*shift:i*shift+peaksegmentsize] += 1
        else:
            allowedlength = len(segmentaddition) - i*shift
            segmentaddition[i * shift:i * shift + allowedlength] += peaksegment[0:allowedlength]
            alreadydone[i * shift:i * shift + allowedlength] += 1
    alreadydone[alreadydone == overlap] = -1
    alreadydone = np.reciprocal(alreadydone) * overlap
    alreadydone[alreadydone < 0] = 1
    segmentaddition = segmentaddition * alreadydone
    foundpeaks, _ = scipy.signal.find_peaks(segmentaddition, height = probabilityfaktor * np.mean(segmentaddition), distance = int((mindistance / lengthfaktor) * samplerate))
    # Correction of the output with respect to the original length of the data
    foundpeaks = (foundpeaks*lengthfaktor).astype(int)
    peaks = np.zeros(origlength)
    peaks[foundpeaks] = 1
    return peaks


def han_preprocessing_all(data, samplerate, low=6, high=25, order=6, windowlength=0.035): 
        #Preprocessing stage for both Han models, additional use of the triangle filter by nguyen (due to better results)
    origlength = len(data)
    newsamplerate = samplerate
    wl = int(samplerate * windowlength)
    cutoff = low
    nyquist = 0.5 * samplerate
    normalized_cutoff = cutoff / nyquist
    coeffs = scipy.signal.butter(order, normalized_cutoff, btype="high", analog=False, output="sos")
    hpf = scipy.signal.sosfiltfilt(coeffs, data)
    #hpf = np.zeros_like(hpf)
    hpf[wl:-wl] = (hpf[wl:-wl] - hpf[:-2 * wl]) * (hpf[wl:-wl] - hpf[wl * 2:])
    hpf[hpf < 0] = 0
    cutoff = high
    normalized_cutoff = cutoff / nyquist
    coeffs = scipy.signal.butter(order, normalized_cutoff, btype="low", analog=False, output = "sos")
    filtered = scipy.signal.sosfiltfilt(coeffs, hpf)
    #return data, origlength, newsamplerate
        #Samplerate correction
    if samplerate != 400:
        newsamplerate = 400
        times = np.arange(len(filtered)) / samplerate
        new_times = np.arange(0, len(filtered) / samplerate, 1 / newsamplerate)
        cs = scipy.interpolate.CubicSpline(times, filtered)
        filtered = cs(new_times)
    return filtered, origlength, newsamplerate


def han_cnn(data, samplerate, segmentsize = 5000, overlap = 2, decisionrulefaktor = 0.1, rawdata = False):
    prepareddata, origlength, newsamplerate = han_preprocessing_all(data, samplerate)
    model = keras.models.load_model("DLmodels/hancnnmodell_withf1_2.h5", custom_objects={"F1Score": external_functions.F1Score})
    peaks = np.zeros(origlength)
    def arraysegmenter(data, segmentsize, overlap):
        ecgparts = []
        shift = int(segmentsize / overlap)
        i = 0
        while i < len(data):
            part = data[i:i + segmentsize]
            if len(part) < segmentsize:
                zeros = segmentsize - len(part)
                part = np.concatenate((part, np.zeros(zeros)))
            maximum = np.max(part)
            minimum = np.min(part)
            normalized = -1 + 2 * (part - minimum) / (maximum - minimum+0.000001)
            ecgparts.append(normalized)
            i = i + shift
        return ecgparts, shift
    datalength = len(prepareddata)
    segments, shift = arraysegmenter(prepareddata, segmentsize, overlap)
    segments = np.asarray(segments)
    peaksegments = model.predict(segments)
    segmentaddition = np.zeros(datalength)
    alreadydone = np.zeros(datalength)
    for i in range(0, len(peaksegments)):
        peaksegment = peaksegments[i][0:np.min([len(peaksegments[i]), segmentsize])].flatten()
        peaksegmentsize = len(peaksegment)
        if i*shift+peaksegmentsize < len(segmentaddition):
            segmentaddition[i*shift:i*shift+peaksegmentsize] += peaksegment
            alreadydone[i*shift:i*shift+peaksegmentsize] += 1
        else:
            allowedlength = len(segmentaddition) - i*shift
            segmentaddition[i * shift:i * shift + allowedlength] += peaksegment[0:allowedlength]
            alreadydone[i * shift:i * shift + allowedlength] += 1
    alreadydone[alreadydone == overlap] = -1
    alreadydone = np.reciprocal(alreadydone) * overlap
    alreadydone[alreadydone < 0] = 1
    segmentaddition = segmentaddition * alreadydone
    #newsamplerate = samplerate
        #Determination of the R-Peaks or Return of the raw data for the ensemble approach
    if not rawdata:
        winsize = int(0.3 * newsamplerate)
        window = np.ones(winsize) / winsize
        segmentaddition = np.convolve(segmentaddition, window, mode="same")
        minimum = np.min(segmentaddition)
        maximum = np.max(segmentaddition)
        segmentaddition = (segmentaddition - minimum) / (maximum - minimum)
        minpeaknumber = int(len(segmentaddition) / (newsamplerate * 0.3))
        sortiert = np.sort(segmentaddition)[::-1]
        peakestimate = sortiert[minpeaknumber]
        foundpeaks, _ = scipy.signal.find_peaks(segmentaddition, height=decisionrulefaktor * peakestimate, distance=0.272 * newsamplerate)
        # Correction of the output with respect to the original length of the data
        foundpeaks = (origlength * foundpeaks / len(segmentaddition)).astype(int)
        peaks[foundpeaks] = 1
        return peaks
    else:
        return segmentaddition, newsamplerate


def han_rnn(data, samplerate, segmentsize = 5000, overlap = 2, decisionrulefaktor = 0.01, rawdata = False):
    prepareddata, origlength, newsamplerate = han_preprocessing_all(data, samplerate)
    model = keras.models.load_model("DLmodels/hanrnnmodell_withf1_2.h5", custom_objects={"F1Score": external_functions.F1Score})
    peaks = np.zeros(origlength)
    def arraysegmenter(data, segmentsize, overlap):
        ecgparts = []
        shift = int(segmentsize / overlap)
        i = 0
        while i < len(data):
            part = data[i:i + segmentsize]
            if len(part) < segmentsize:
                zeros = segmentsize - len(part)
                part = np.concatenate((part, np.zeros(zeros)))
            maximum = np.max(part)
            minimum = np.min(part)
            normalized = -1 + 2 * (part - minimum) / (maximum - minimum)
            ecgparts.append(normalized)
            i = i + shift
        return ecgparts, shift
    datalength = len(prepareddata)
        #Data segmentation
    segments, shift = arraysegmenter(prepareddata, segmentsize, overlap)
    segments = np.asarray(segments)
    peaksegments = model.predict(segments)
    segmentaddition = np.zeros(datalength)
    alreadydone = np.zeros(datalength)
    for i in range(0, len(peaksegments)):
        peaksegment = peaksegments[i][0:np.min([len(peaksegments[i]), segmentsize])].flatten()
        peaksegmentsize = len(peaksegment)
        if i*shift+peaksegmentsize < len(segmentaddition):
            segmentaddition[i*shift:i*shift+peaksegmentsize] += peaksegment
            alreadydone[i*shift:i*shift+peaksegmentsize] += 1
        else:
            allowedlength = len(segmentaddition) - i*shift
            segmentaddition[i * shift:i * shift + allowedlength] += peaksegment[0:allowedlength]
            alreadydone[i * shift:i * shift + allowedlength] += 1
    alreadydone[alreadydone == overlap] = -1
    alreadydone = np.reciprocal(alreadydone) * overlap
    alreadydone[alreadydone < 0] = 1
    segmentaddition = segmentaddition * alreadydone
        #Either Determination of the R-Peaks or return of the raw data for the Ensemble
    if not rawdata:
        winsize = int(0.3 * newsamplerate)
        window = np.ones(winsize) / winsize
        segmentaddition = np.convolve(segmentaddition, window, mode="same")
        minimum = np.min(segmentaddition)
        maximum = np.max(segmentaddition)
        segmentaddition = (segmentaddition - minimum) / (maximum - minimum)
        minpeaknumber = int(len(segmentaddition) / (newsamplerate * 0.3))
        sortiert = np.sort(segmentaddition)[::-1]
        peakestimate = sortiert[minpeaknumber]
        foundpeaks, _ = scipy.signal.find_peaks(segmentaddition, height=decisionrulefaktor * peakestimate, distance=0.272 * newsamplerate)
        #Correction of the output with respect to the original length of the data
        foundpeaks = (origlength * foundpeaks / len(segmentaddition)).astype(int)
        peaks[foundpeaks] = 1
        return peaks
    else:
        return segmentaddition, newsamplerate
