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

_MODEL_PATH = os.environ.get("MODEL_PATH", "saved_models/custom_cnn.keras")
_model = tf.keras.models.load_model(_MODEL_PATH)

_norm_stats = None
_stats_path = _MODEL_PATH.replace(".keras", "_norm_stats.npz")
if os.path.exists(_stats_path):
    _d = np.load(_stats_path)
    _norm_stats = (_d["X_mean"], _d["X_std"])


def predict(image):
    img = Image.fromarray(image).convert("L").resize((28, 28))
    arr = np.array(img, dtype="float32")[..., np.newaxis]
    if _norm_stats is not None:
        arr = (arr - _norm_stats[0]) / _norm_stats[1]
    else:
        arr = arr / 255.0
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
