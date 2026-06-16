import numpy as np
import tensorflow as tf
from tensorflow import keras


def central_crop(image):
    shape = tf.shape(image)
    min_dim = tf.reduce_min([shape[0], shape[1]])
    top = (shape[0] - min_dim) // 2
    left = (shape[1] - min_dim) // 2
    return image[top : top + min_dim, left : left + min_dim]


def random_crop(image):
    shape = tf.shape(image)
    min_dim = tf.reduce_min([shape[0], shape[1]]) * 90 // 100
    return tf.image.random_crop(image, [min_dim, min_dim, 3])


def preprocess(image, label, randomize=False):
    """Crop, resize to 224×224, and apply Xception preprocessing."""
    if randomize:
        image = random_crop(image)
        image = tf.image.random_flip_left_right(image)
    else:
        image = central_crop(image)
    image = tf.image.resize(image, [224, 224])
    image = keras.applications.xception.preprocess_input(image)
    return image, label


def normalize(X):
    """Scale tensor values to [0, 1]."""
    return (X - tf.reduce_min(X)) / (tf.reduce_max(X) - tf.reduce_min(X))
