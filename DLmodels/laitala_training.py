import numpy as np
from keras.losses import BinaryCrossentropy
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split  # Assuming you have sklearn installed
import laitaladatapreparation
from keras.optimizers import Adam
from ma_external_functions import F1Score


def lstm_model(lr):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, activation="tanh", return_sequences=True), input_shape=(1000, 1)))
    model.add(Bidirectional(LSTM(64, activation="tanh", return_sequences=True)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=Adam(learning_rate=lr), loss=BinaryCrossentropy(), metrics=[F1Score()])
    return model


def train_and_save_model(model, ecg_segments, peak_segments, epochs, batch_size, validation_split, filename):
    ecg_segments_np = np.array(ecg_segments)
    peak_segments_np = np.array(peak_segments)
    ecg_segments_np = ecg_segments_np.reshape((-1, 1000, 1))
    X_train, X_val, y_train, y_val = train_test_split(ecg_segments_np, peak_segments_np, test_size=validation_split)
    early_stopping = EarlyStopping(monitor="val_loss", patience=20, verbose=1, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(filename, monitor="val_loss", save_best_only=True, verbose=1)

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping, model_checkpoint])
    model.save("laitalamodell_final_2.h5")
    return history

epochs = 250
validation_split = 0.1
model = lstm_model(0.001)
ecg_segments, peak_segments, samplerate = laitaladatapreparation.createtrainingdatacpsc()
train_and_save_model(model, ecg_segments, peak_segments, epochs=epochs, batch_size=16, validation_split=validation_split, filename = "laitalamodell_withf1.h5")
