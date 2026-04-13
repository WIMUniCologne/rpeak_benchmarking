import numpy as np
import scipy
import keras
import os
from cpsc import cpsc
from keras.losses import BinaryCrossentropy
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split  
from keras.optimizers import Adam
from external_functions import F1Score

dirname = os.path.dirname(__file__)


def laitala(modelname, data, samplerate, low = 8, high = 15, order = 1, segmentsize = 1000, overlap = 2, probabilityfaktor = 0.1):
    model = keras.models.load_model(dirname + "/trained_DLmodels/" + modelname, custom_objects={"F1Score": F1Score})
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


def model(lr=0.001):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, activation="tanh", return_sequences=True), input_shape=(1000, 1)))
    model.add(Bidirectional(LSTM(64, activation="tanh", return_sequences=True)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=Adam(learning_rate=lr), loss=BinaryCrossentropy(), metrics=[F1Score()])
    return model


def train_and_save_model(modelname, epochs=250, batch_size=16, validation_split=0.1):
    ecg_segments, peak_segments, samplerate = createtrainingdatacpsc()
    ecg_segments_np = np.array(ecg_segments)
    peak_segments_np = np.array(peak_segments)
    ecg_segments_np = ecg_segments_np.reshape((-1, 1000, 1))
    X_train, X_val, y_train, y_val = train_test_split(ecg_segments_np, peak_segments_np, test_size=validation_split)
    early_stopping = EarlyStopping(monitor="val_loss", patience=20, verbose=1, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(dirname + "/trained_DLmodels/" + modelname, monitor="val_loss", save_best_only=True, verbose=1)
    model = model(lr=0.001)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping, model_checkpoint])
    return history


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