import zahiddatapreparation
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, Activation, MaxPooling1D, UpSampling1D
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.initializers import initializers_v1
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from ma_external_functions import F1Score

def cnnstructure():
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

def prepare_data(x_data, y_data, samplerate):
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

def train_model(model, x_set, y_set, epochs=250, batch_size=8, validation_split=0.1):
    X_train, X_test, y_train, y_test = train_test_split(x_set, y_set, test_size=validation_split)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=BinaryCrossentropy(), metrics=[F1Score()])
    early_stopping = EarlyStopping(monitor="val_loss", patience=20, verbose=1, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint("zahidmodell_withf1.h5", monitor="val_loss", save_best_only=True, verbose=1)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[early_stopping, model_checkpoint])
    return history


x_data, y_data, samplerate = zahiddatapreparation.createtrainingdatacpsc()
x_set, y_set = prepare_data(x_data, y_data, samplerate)
model = cnnstructure()
history = train_model(model, x_set, y_set)