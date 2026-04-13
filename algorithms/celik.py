import numpy as np
import scipy
import pywt
import keras
import os
from cpsc import cpsc
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Concatenate
from keras.models import Model
from external_functions import F1Score

dirname = os.path.dirname(__file__)


def celik(modelname, data, samplerate, low = 8, high = 50, order = 1, overlap = 2, probabilityfaktor = 0.01):
    origlength = len(data)
    mindistance = 0.275
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
    model = keras.models.load_model(dirname + "/trained_DLmodels/" + modelname, custom_objects={"F1Score": F1Score})
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
            ecgparts.append(wavelettransformation(part, newsamplerate))
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


def model(input_shape=(16, 512, 1)):
    inputs = Input(input_shape)

    conv1 = Conv2D(32, 3, activation="relu", padding="same")(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, 3, activation="relu", padding="same")(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, activation="relu", padding="same")(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, 3, activation="relu", padding="same")(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, activation="relu", padding="same")(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, 3, activation="relu", padding="same")(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, activation="relu", padding="same")(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, 3, activation="relu", padding="same")(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, 3, activation="relu", padding="same")(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, 3, activation="relu", padding="same")(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = Concatenate()([UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = Conv2D(256, 3, activation="relu", padding="same")(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, 3, activation="relu", padding="same")(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Concatenate()([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = Conv2D(128, 3, activation="relu", padding="same")(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, 3, activation="relu", padding="same")(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Concatenate()([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = Conv2D(64, 3, activation="relu", padding="same")(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, 3, activation="relu", padding="same")(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Concatenate()([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Conv2D(32, 3, activation="relu", padding="same")(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, 3, activation="relu", padding="same")(conv9)
    conv9 = BatchNormalization()(conv9)

    outputs = Conv2D(1, 1, activation="sigmoid")(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def training(modelname, learnrate=0.001):
    X_train, y_train = createtrainingdatacpsc()
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train , test_size=0.1
    )
    model = model()
    adam_optimizer = Adam(lr=learnrate)
    model.compile(optimizer=adam_optimizer, loss="binary_crossentropy", metrics=[F1Score()])
    early_stopping = EarlyStopping(monitor="val_loss", patience=20, verbose=1, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(dirname + "/trained_DLmodels/" + modelname, monitor="val_loss", save_best_only=True, verbose=1)
    history = model.fit(X_train, y_train,
        epochs=250,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, model_checkpoint]
    )
    return history


def createtrainingdatacpsc():
    samplerate = 400
    windowlength = 4 * samplerate
    datasegments = []
    peaksegments = []
    filenames, filepath = cpsc()
    for i in filenames:
        ecgdata = scipy.io.loadmat(filepath + "A" + i + ".mat")
        ecgdata = ecgdata["ecg"]
        ecgdata = np.squeeze(ecgdata)
        peakdata = scipy.io.loadmat(filepath + "RPN_" + i + ".mat")
        peakdata = peakdata["R"]
        peakdata = np.squeeze(peakdata)
        peaks = np.zeros(len(ecgdata))
        peaks[peakdata] = 1
        length = len(ecgdata)
        ecgdata = ecgdata[0:int(0.5 * length)]
        peaks = peaks[0:int(0.5 * length)]
        for j in range(0, len(ecgdata), windowlength):
            segment = ecgdata[j: j + windowlength]
            peaksegment = peaks[j: j + windowlength]
            if len(segment) < windowlength:
                pad_length = windowlength - len(segment)
                segment = np.pad(segment, (0, pad_length), "constant")
                peaksegment = np.pad(peaksegment, (0, pad_length), "constant")
            downsizenumber = 512
            origsegmentlength = len(segment)
            downsampledecg = scipy.signal.resample(segment, downsizenumber)
            downsizedsegmentlength = len(downsampledecg)
            downsampledecg = wavelettransformation(downsampledecg, samplerate)
            oneindexe = np.where(peaksegment == 1)[0]
            oneindexe = (downsizedsegmentlength * oneindexe / origsegmentlength).astype(int)
            downsampledpeaks = np.zeros(512)
            for k in oneindexe:
                downsampledpeaks[k - 10:k + 10] = int(1)
            downsampledpeaks = np.tile(downsampledpeaks, (16, 1))
            datasegments.append(downsampledecg)
            peaksegments.append(downsampledpeaks)
    datasegments = np.asarray(datasegments)
    peaksegments = np.asarray(peaksegments)
    return datasegments, peaksegments


def wavelettransformation(data, samplerate, samplerateadjustment=0.8):
    coefficients, frequencies = pywt.cwt(data, np.arange(1, 100, samplerateadjustment), "morl")
    frequencies = frequencies * samplerate
    filtered_coefficients = np.real(coefficients[(frequencies >= 16.66) & (frequencies <= 47.13)])
    maximum = np.max(filtered_coefficients)
    minimum = np.min(filtered_coefficients)
    if maximum - minimum != 0:
        filtered_coefficients = filtered_coefficients - minimum / (maximum - minimum)
    #print(filtered_coefficients.shape)
    return filtered_coefficients
