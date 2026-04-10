from keras.layers import LSTM
from sklearn.model_selection import train_test_split
import han_datapreparation
from keras.layers import Input, Conv1D, BatchNormalization, Activation, MaxPooling1D, UpSampling1D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from ma_external_functions import F1Score


def traincnnmodel():
    ecgsegments, peaksegments = han_datapreparation.createtrainingdatacpsc()
    ecgsegments = ecgsegments.reshape((-1, 5000, 1))
    peaksegments = peaksegments.reshape((-1, 5000, 1))
    X_train, X_test, y_train, y_test = train_test_split(ecgsegments, peaksegments, test_size=0.1)
    model = cnn_structure()
    adam_optimizer = Adam(lr=0.001)
    model.compile(optimizer=adam_optimizer, loss="binary_crossentropy", metrics=[F1Score()])
    early_stopping = EarlyStopping(monitor="val_loss", patience=20, verbose=1, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint("hancnnmodell_withf1_2.h5", monitor="val_loss", save_best_only=True, verbose=1)
    history = model.fit(X_train, y_train, epochs=250, batch_size=64,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping, model_checkpoint])
    return history


def trainrnnmodel():
    ecgsegments, peaksegments = han_datapreparation.createtrainingdatacpsc()
    ecgsegments = ecgsegments.reshape((-1, 5000, 1))
    peaksegments = peaksegments.reshape((-1, 5000, 1))
    X_train, X_test, y_train, y_test = train_test_split(ecgsegments, peaksegments, test_size=0.1)
    print(X_train.shape)
    print(y_train.shape)
    model = rnn_structure()
    adam_optimizer = Adam(lr=0.001)
    model.compile(optimizer=adam_optimizer, loss="binary_crossentropy", metrics=[F1Score()])
    early_stopping = EarlyStopping(monitor="val_loss", patience=20, verbose=1, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint("hanrnnmodell_withf1_2.h5", monitor="val_loss", save_best_only=True, verbose=1)
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


traincnnmodel()
# trainrnnmodel()