from functools import partial

from tensorflow import keras


def build(input_shape=(28, 28, 1), n_classes=10, dropout=0.5):
    DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, activation="relu", padding="SAME")
    return keras.models.Sequential([
        keras.layers.Input(shape=input_shape),
        DefaultConv2D(filters=64, kernel_size=7),
        keras.layers.MaxPooling2D(pool_size=2),
        DefaultConv2D(filters=128),
        DefaultConv2D(filters=128),
        keras.layers.MaxPooling2D(pool_size=2),
        DefaultConv2D(filters=256),
        DefaultConv2D(filters=256),
        keras.layers.MaxPooling2D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(units=128, activation="relu"),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(units=64, activation="relu"),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(units=n_classes, activation="softmax"),
    ])
