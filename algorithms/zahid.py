import numpy as np
import scipy
import keras
import os
from cpsc import cpsc
from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, Activation, MaxPooling1D, UpSampling1D
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.initializers import initializers_v1
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from external_functions import F1Score

dirname = os.path.dirname(__file__)


def zahid(modelname, data, samplerate, segmentsize = 5000, overlap = 2, probabilityfaktor = 0.1, winsizefaktor = 0.1):
    model = keras.models.load_model(dirname + "/trained_DLmodels/" + modelname, custom_objects={"F1Score": F1Score})
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


def model():
    input_layer = Input(shape=(None, 1))
    x = Conv1D(16, kernel_size=9, padding="same", kernel_initializer=initializers_v1.RandomUniform(minval=-0.1, maxval=0.1))(input_layer)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2, padding="same")(x)
    x = Conv1D(16, kernel_size=9, padding="same", kernel_initializer=initializers_v1.RandomUniform(minval=-0.1, maxval=0.1))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2, padding="same")(x)
    x = Conv1D(32, kernel_size=6, padding="same", kernel_initializer=initializers_v1.RandomUniform(minval=-0.1, maxval=0.1))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2, padding="same")(x)
    x = Conv1D(32, kernel_size=6, padding="same", kernel_initializer=initializers_v1.RandomUniform(minval=-0.1, maxval=0.1))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2, padding="same")(x)
    x = Conv1D(64, kernel_size=3, padding="same", kernel_initializer=initializers_v1.RandomUniform(minval=-0.1, maxval=0.1))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2, padding="same")(x)
    x = Conv1D(64, kernel_size=3, padding="same", kernel_initializer=initializers_v1.RandomUniform(minval=-0.1, maxval=0.1))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2, padding="same")(x)
    x = Conv1D(64, kernel_size=3, padding="same", kernel_initializer=initializers_v1.RandomUniform(minval=-0.1, maxval=0.1))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = UpSampling1D(size=2)(x)
    x = Conv1D(64, kernel_size=3, padding="same", kernel_initializer=initializers_v1.RandomUniform(minval=-0.1, maxval=0.1))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = UpSampling1D(size=2)(x)
    x = Conv1D(32, kernel_size=6, padding="same", kernel_initializer=initializers_v1.RandomUniform(minval=-0.1, maxval=0.1))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = UpSampling1D(size=2)(x)
    x = Conv1D(32, kernel_size=6, padding="same", kernel_initializer=initializers_v1.RandomUniform(minval=-0.1, maxval=0.1))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = UpSampling1D(size=2)(x)
    x = Conv1D(16, kernel_size=9, padding="same", kernel_initializer=initializers_v1.RandomUniform(minval=-0.1, maxval=0.1))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = UpSampling1D(size=2)(x)
    x = Conv1D(16, kernel_size=9, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = UpSampling1D(size=2)(x)
    output_layer = Conv1D(1, kernel_size=3, padding="same", activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

def prepare_data():
    x_data, y_data, samplerate = createtrainingdatacpsc()
    setlength = 20 * samplerate
    x_set = []
    y_set = []
    numberofsets = int(len(x_data) / setlength)
    for i in range(0, numberofsets):
        x = x_data[setlength * i : setlength * i + setlength]
        y = y_data[setlength * i : setlength * i + setlength]
        x_set.append(x)
        y_set.append(y)
    x_set = np.asarray(x_set)
    y_set = np.asarray(y_set)
    return x_set, y_set

def train_model(modelname, x_set, y_set, epochs=250, batch_size=8, validation_split=0.1):
    x_set, y_set = prepare_data()
    model = model()
    X_train, X_test, y_train, y_test = train_test_split(x_set, y_set, test_size=validation_split)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=BinaryCrossentropy(), metrics=[F1Score()])
    early_stopping = EarlyStopping(monitor="val_loss", patience=20, verbose=1, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(dirname + "/trained_DLmodels/" + modelname, monitor="val_loss", save_best_only=True, verbose=1)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[early_stopping, model_checkpoint])
    return history


def createtrainingdatacpsc():
    samplerate = 400
    x_set = np.asarray([])
    y_set = np.asarray([])
    filenames, filepath = cpsc()
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