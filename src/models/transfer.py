from tensorflow import keras

_SUPPORTED = ("xception", "resnet50")


def build(n_classes: int, base: str = "xception"):
    if base == "xception":
        base_model = keras.applications.xception.Xception(weights="imagenet", include_top=False)
    elif base == "resnet50":
        base_model = keras.applications.resnet50.ResNet50(weights="imagenet", include_top=False)
    else:
        raise ValueError(f"base must be one of {_SUPPORTED}, got '{base}'")

    avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = keras.layers.Dense(n_classes, activation="softmax")(avg)
    model = keras.models.Model(inputs=base_model.input, outputs=output)
    return model, base_model
