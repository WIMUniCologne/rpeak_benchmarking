import numpy as np
import scipy
import keras
import os
from cpsc import cpsc
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
from keras.layers import Input, Conv1D, BatchNormalization, Activation, MaxPooling1D, UpSampling1D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from external_functions import F1Score

dirname = os.path.dirname(__file__)


def han_cnn(modelname, data, samplerate, segmentsize = 5000, overlap = 2, decisionrulefaktor = 0.1, rawdata = False):
    prepareddata, origlength, newsamplerate = han_preprocessing_all(data, samplerate)
    model = keras.models.load_model(dirname + "/trained_DLmodels/" + modelname, custom_objects={"F1Score": F1Score})
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
    

def han_rnn(modelname, data, samplerate, segmentsize = 5000, overlap = 2, decisionrulefaktor = 0.01, rawdata = False):
    prepareddata, origlength, newsamplerate = han_preprocessing_all(data, samplerate)
    model = keras.models.load_model(dirname + "/trained_DLmodels/" + modelname, custom_objects={"F1Score": F1Score})
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


def traincnnmodel(modelname):
    ecgsegments, peaksegments = createtrainingdatacpsc()
    ecgsegments = ecgsegments.reshape((-1, 5000, 1))
    peaksegments = peaksegments.reshape((-1, 5000, 1))
    X_train, X_test, y_train, y_test = train_test_split(ecgsegments, peaksegments, test_size=0.1)
    model = cnn_structure()
    adam_optimizer = Adam(lr=0.001)
    model.compile(optimizer=adam_optimizer, loss="binary_crossentropy", metrics=[F1Score()])
    early_stopping = EarlyStopping(monitor="val_loss", patience=20, verbose=1, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(dirname + "/trained_DLmodels/" + modelname, monitor="val_loss", save_best_only=True, verbose=1)
    history = model.fit(X_train, y_train, epochs=250, batch_size=64,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping, model_checkpoint])
    return history


def trainrnnmodel(modelname):
    ecgsegments, peaksegments = createtrainingdatacpsc()
    ecgsegments = ecgsegments.reshape((-1, 5000, 1))
    peaksegments = peaksegments.reshape((-1, 5000, 1))
    X_train, X_test, y_train, y_test = train_test_split(ecgsegments, peaksegments, test_size=0.1)
    print(X_train.shape)
    print(y_train.shape)
    model = rnn_structure()
    adam_optimizer = Adam(lr=0.001)
    model.compile(optimizer=adam_optimizer, loss="binary_crossentropy", metrics=[F1Score()])
    early_stopping = EarlyStopping(monitor="val_loss", patience=20, verbose=1, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(dirname + "/trained_DLmodels/" + modelname, monitor="val_loss", save_best_only=True, verbose=1)
    history = model.fit(X_train, y_train, epochs=250, batch_size=64,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping, model_checkpoint])
    return history


def cnn_structure():
    input_layer = Input(shape=(5000, 1))
    x1 = Conv1D(16, kernel_size=31, padding="same")(input_layer)
    x2 = Conv1D(16, kernel_size=31, padding="same")(x1)
    x2 = BatchNormalization()(x2)
    x2 = Activation("relu")(x2)
    x2 = MaxPooling1D(pool_size=2, padding="same")(x2)
    x3 = Conv1D(32, kernel_size=25, padding="same")(x2)
    x4 = Conv1D(32, kernel_size=25, padding="same")(x3)
    x4 = BatchNormalization()(x4)
    x4 = Activation("relu")(x4)
    x4 = MaxPooling1D(pool_size=2, padding="same")(x4)
    x5 = Conv1D(64, kernel_size=19, padding="same")(x4)
    x6 = Conv1D(64, kernel_size=19, padding="same")(x5)
    x6 = BatchNormalization()(x6)
    x6 = Activation("relu")(x6)
    x6 = MaxPooling1D(pool_size=2, padding="same")(x6)
    x7 = Conv1D(128, kernel_size=13, padding="same")(x6)
    x8 = Conv1D(128, kernel_size=13, padding="same")(x7)
    x8 = BatchNormalization()(x8)
    x8 = Activation("relu")(x8)
    x9 = UpSampling1D(size=2)(x8)
    x9 = Conv1D(64, kernel_size=19, padding="same")(x9)
    x10 = Conv1D(64, kernel_size=19, padding="same")(x9)
    x10 = BatchNormalization()(x10)
    x10 = Activation("relu")(x10)
    x11 = Conv1D(64, kernel_size=19, padding="same")(x10)
    x11 = BatchNormalization()(x11)
    x11 = Activation("relu")(x11)
    x12 = UpSampling1D(size=2)(x11)
    x12 = Conv1D(32, kernel_size=25, padding="same")(x12)
    x13 = Conv1D(32, kernel_size=25, padding="same")(x12)
    x13 = BatchNormalization()(x13)
    x13 = Activation("relu")(x13)
    x14 = Conv1D(32, kernel_size=25, padding="same")(x13)
    x14 = BatchNormalization()(x14)
    x14 = Activation("relu")(x14)
    x15 = UpSampling1D(size=2)(x14)
    x15 = Conv1D(16, kernel_size=31, padding="same")(x15)
    x16 = Conv1D(16, kernel_size=31, padding="same")(x15)
    x16 = BatchNormalization()(x16)
    x16 = Activation("relu")(x16)
    x17 = Conv1D(16, kernel_size=31, padding="same")(x16)
    x17 = BatchNormalization()(x17)
    x17 = Activation("relu")(x17)
    x18 = Conv1D(1, kernel_size=1, padding="same")(x17)
    output_layer = Activation("sigmoid")(x18)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def rnn_structure():
    input_layer = Input(shape=(5000, 1))
    x1 = Conv1D(16, kernel_size=31, padding="same")(input_layer)
    x2 = Conv1D(16, kernel_size=31, padding="same")(x1)
    x2 = BatchNormalization()(x2)
    x2 = Activation("relu")(x2)
    x2 = MaxPooling1D(pool_size=2, padding="same")(x2)
    x3 = Conv1D(32, kernel_size=25, padding="same")(x2)
    x4 = Conv1D(32, kernel_size=25, padding="same")(x3)
    x4 = BatchNormalization()(x4)
    x4 = Activation("relu")(x4)
    x4 = MaxPooling1D(pool_size=2, padding="same")(x4)
    x5 = Conv1D(64, kernel_size=19, padding="same")(x4)
    x6 = Conv1D(64, kernel_size=19, padding="same")(x5)
    x6 = BatchNormalization()(x6)
    x6 = Activation("relu")(x6)
    x6 = MaxPooling1D(pool_size=2, padding="same")(x6)
    # Use of LSTM in the bottom layers as described in the paper:
    x7 = LSTM(128, return_sequences=True)(x6)
    x8 = LSTM(128, return_sequences=True)(x7)
    x9 = UpSampling1D(size=2)(x8)
    x9 = Conv1D(64, kernel_size=19, padding="same")(x9)
    x10 = Conv1D(64, kernel_size=19, padding="same")(x9)
    x10 = BatchNormalization()(x10)
    x10 = Activation("relu")(x10)
    x11 = Conv1D(64, kernel_size=19, padding="same")(x10)
    x11 = BatchNormalization()(x11)
    x11 = Activation("relu")(x11)
    x12 = UpSampling1D(size=2)(x11)
    x12 = Conv1D(32, kernel_size=25, padding="same")(x12)
    x13 = Conv1D(32, kernel_size=25, padding="same")(x12)
    x13 = BatchNormalization()(x13)
    x13 = Activation("relu")(x13)
    x14 = Conv1D(32, kernel_size=25, padding="same")(x13)
    x14 = BatchNormalization()(x14)
    x14 = Activation("relu")(x14)
    x15 = UpSampling1D(size=2)(x14)
    x15 = Conv1D(16, kernel_size=31, padding="same")(x15)
    x16 = Conv1D(16, kernel_size=31, padding="same")(x15)
    x16 = BatchNormalization()(x16)
    x16 = Activation("relu")(x16)
    x17 = Conv1D(16, kernel_size=31, padding="same")(x16)
    x17 = BatchNormalization()(x17)
    x17 = Activation("relu")(x17)
    x18 = Conv1D(1, kernel_size=1, padding="same")(x17)
    output_layer = Activation("sigmoid")(x18)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model


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