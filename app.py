"""Hugging Face Spaces entry point.

Identical to demo.py but calls launch() unconditionally so HF Spaces
picks it up automatically.
"""
import os

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

_model = tf.keras.models.load_model(
    os.environ.get("MODEL_PATH", "saved_models/custom_cnn.keras")
)


def predict(image):
    img = Image.fromarray(image).convert("L").resize((28, 28))
    arr = np.array(img, dtype="float32")[..., np.newaxis] / 255.0
    probs = _model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
    return {name: float(p) for name, p in zip(CLASS_NAMES, probs)}


gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload a clothing image"),
    outputs=gr.Label(num_top_classes=5, label="Predictions"),
    title="Fashion-MNIST Clothing Classifier",
    description=(
        "Classify clothing images into one of 10 categories using a CNN trained on Fashion-MNIST.\n\n"
        "**Best results:** plain white/grey background, item centred, no people. "
        "Real-world photos with complex backgrounds or lighting may be less accurate — "
        "the model was trained on 28×28 thumbnail sketches, not photographs."
    ),
    flagging_mode="never",
).launch()
