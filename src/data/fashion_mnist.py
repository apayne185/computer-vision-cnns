import numpy as np
import tensorflow as tf
from tensorflow import keras

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def load(resize_to=None):
    """Return (X_train, y_train), (X_valid, y_valid), (X_test, y_test).

    resize_to: optional (H, W) tuple — upscales images and normalises to [0, 1].
    Without it, images are standardised per-pixel using training-set statistics.
    """
    (X_full, y_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

    X_full = X_full[..., np.newaxis].astype("float32")
    X_test = X_test[..., np.newaxis].astype("float32")

    if resize_to is not None:
        X_full = tf.image.resize(X_full, resize_to).numpy() / 255.0
        X_test = tf.image.resize(X_test, resize_to).numpy() / 255.0
    else:
        X_mean = X_full[:-5000].mean(axis=0, keepdims=True)
        X_std = X_full[:-5000].std(axis=0, keepdims=True) + 1e-7
        X_full = (X_full - X_mean) / X_std
        X_test = (X_test - X_mean) / X_std

    X_train, X_valid = X_full[:-5000], X_full[-5000:]
    y_train, y_valid = y_full[:-5000], y_full[-5000:]

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)
