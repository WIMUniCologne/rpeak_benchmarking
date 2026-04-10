from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import celik_datapreparation
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Concatenate
from keras.models import Model
from ma_external_functions import F1Score

def celik(input_shape=(16, 512, 1)):
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


def training(learnrate, filename):
    X_train, y_train = celik_datapreparation.createtrainingdatacpsc()
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train , test_size=0.1
    )
    model = celik()
    adam_optimizer = Adam(lr=learnrate)
    model.compile(optimizer=adam_optimizer, loss="binary_crossentropy", metrics=[F1Score()])
    early_stopping = EarlyStopping(monitor="val_loss", patience=20, verbose=1, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(filename, monitor="val_loss", save_best_only=True, verbose=1)
    history = model.fit(X_train, y_train,
        epochs=250,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, model_checkpoint]
    )
    return history


training(learnrate = 0.001, filename = "celikmodell_withf1.h5")
