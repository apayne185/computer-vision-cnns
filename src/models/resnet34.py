from functools import partial

from tensorflow import keras

_DefaultConv2D = partial(
    keras.layers.Conv2D, kernel_size=3, strides=1, padding="SAME", use_bias=False
)


class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            _DefaultConv2D(filters, strides=strides),
            keras.layers.BatchNormalization(),
            keras.layers.Activation(activation),
            _DefaultConv2D(filters),
            keras.layers.BatchNormalization(),
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                _DefaultConv2D(filters, kernel_size=1, strides=strides),
                keras.layers.BatchNormalization(),
            ]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)


def build(input_shape=(224, 224, 1), n_classes=10):
    model = keras.models.Sequential([keras.layers.Input(shape=input_shape)])
    model.add(_DefaultConv2D(64, kernel_size=7, strides=2))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"))

    prev_filters = 64
    for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
        strides = 1 if filters == prev_filters else 2
        model.add(ResidualUnit(filters, strides=strides))
        prev_filters = filters

    model.add(keras.layers.GlobalAvgPool2D())
    model.add(keras.layers.Dense(n_classes, activation="softmax"))
    return model
