from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, concatenate, BatchNormalization
import xiang_datapreparation
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from ma_external_functions import F1Score


def modelstructure():
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

#ecgsegmentsrawdiff, ecgsegmentsavgdiff, peaksegments = xiang_datapreparation.createtrainingdatacpsc()

def train_cnn_model():
    ecg_segments_raw_diff, ecg_segments_avg_diff, peak_segments = xiang_datapreparation.createtrainingdatacpsc()
    ecg_segments_raw_diff = ecg_segments_raw_diff.reshape((-1, 56, 1))
    ecg_segments_avg_diff = ecg_segments_avg_diff.reshape((-1, 280, 1))
    peak_segments = peak_segments.reshape((-1, 1))
    X_train_raw_diff, X_test_raw_diff, X_train_avg_diff, X_test_avg_diff, y_train, y_test = train_test_split(
        ecg_segments_raw_diff, ecg_segments_avg_diff, peak_segments, test_size=0.1)
    model = modelstructure()
    adam_optimizer = Adam(lr=0.0001)
    model.compile(optimizer=adam_optimizer, loss="binary_crossentropy", metrics=[F1Score()])
    early_stopping = EarlyStopping(monitor="val_loss", patience=20, verbose=1, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint("xiangmodell_withf1_13.h5", monitor="val_loss", save_best_only=True, verbose=1)
    history = model.fit(
        [X_train_raw_diff, X_train_avg_diff],
        y_train,
        epochs=250,
        batch_size=64,
        validation_data=([X_test_raw_diff, X_test_avg_diff], y_test),
        callbacks=[early_stopping, model_checkpoint]
    )
    return history

train_cnn_model()
model = modelstructure()
model.summary()