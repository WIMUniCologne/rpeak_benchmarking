import numpy as np
import scipy
import keras
import os
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, concatenate, BatchNormalization
from cpsc import cpsc
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from external_functions import F1Score

dirname = os.path.dirname(__file__)


def xiang(modelname, data, samplerate, center = 20, segmentsize = 56, overlap = 15, probabilityfaktor = 0.8):
    model = keras.models.load_model(dirname + "/trained_DLmodels/" + modelname, custom_objects={"F1Score": F1Score})
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
    #print(len(segmentaddition))
        #Meanvalue for probability threshold
    meanvalue = np.mean(segmentaddition)
    peaks = np.zeros(origlength)
    #print(len(peaks))
    foundpeaks, _ = scipy.signal.find_peaks(segmentaddition, height = probabilityfaktor * meanvalue ,distance = 0.272 * 360)
        # Correction of the output with respect to the original length of the data
    foundpeaks = (foundpeaks * origlength / len(data)).astype(int)
    peaks[foundpeaks] = 1
    return peaks


def model():
    #Part-Level
    input_part_level = Input(shape=(56, 1), name="input_part_level")
    part_level_cnn = Conv1D(5, kernel_size=5, activation="relu")(input_part_level)
    part_level_cnn = BatchNormalization()(part_level_cnn)
    part_level_cnn = MaxPooling1D(pool_size=2)(part_level_cnn)
    part_level_cnn = Flatten()(part_level_cnn)
    #Object-Level
    input_object_level = Input(shape=(280, 1), name="input_object_level")
    object_level_cnn = Conv1D(5, kernel_size=5, activation="relu")(input_object_level)
    object_level_cnn = BatchNormalization()(object_level_cnn)
    object_level_cnn = MaxPooling1D(pool_size=2)(object_level_cnn)
    object_level_cnn = Conv1D(5, kernel_size=5, activation="relu")(object_level_cnn)
    object_level_cnn = BatchNormalization()(object_level_cnn)
    object_level_cnn = MaxPooling1D(pool_size=2)(object_level_cnn)
    object_level_cnn = Flatten()(object_level_cnn)
    # MLP
    merged_layers = concatenate([part_level_cnn, object_level_cnn])
    mlp_hidden_1 = Dense(20, activation="relu")(merged_layers)
    mlp_hidden_2 = Dense(1, activation="sigmoid", name="output_layer")(mlp_hidden_1)
    model = Model(inputs=[input_part_level, input_object_level], outputs=mlp_hidden_2)
    return model


def train_cnn_model(modelname):
    ecg_segments_raw_diff, ecg_segments_avg_diff, peak_segments = createtrainingdatacpsc()
    ecg_segments_raw_diff = ecg_segments_raw_diff.reshape((-1, 56, 1))
    ecg_segments_avg_diff = ecg_segments_avg_diff.reshape((-1, 280, 1))
    peak_segments = peak_segments.reshape((-1, 1))
    X_train_raw_diff, X_test_raw_diff, X_train_avg_diff, X_test_avg_diff, y_train, y_test = train_test_split(
        ecg_segments_raw_diff, ecg_segments_avg_diff, peak_segments, test_size=0.1)
    model = model()
    adam_optimizer = Adam(lr=0.0001)
    model.compile(optimizer=adam_optimizer, loss="binary_crossentropy", metrics=[F1Score()])
    early_stopping = EarlyStopping(monitor="val_loss", patience=20, verbose=1, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(dirname + "/trained_DLmodels/" + modelname, monitor="val_loss", save_best_only=True, verbose=1)
    history = model.fit(
        [X_train_raw_diff, X_train_avg_diff],
        y_train,
        epochs=250,
        batch_size=64,
        validation_data=([X_test_raw_diff, X_test_avg_diff], y_test),
        callbacks=[early_stopping, model_checkpoint]
    )
    return history


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
        #print(i)
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
        #print(peakamount, nopeakamount)
    ecgsegmentsrawdiff = np.asarray(ecgsrdsegments)
    ecgsegmentsavgdiff = np.asarray(ecgsadsegments)
    peaksegments = np.asarray(ispeaksegment)
    return ecgsegmentsrawdiff, ecgsegmentsavgdiff, peaksegments
