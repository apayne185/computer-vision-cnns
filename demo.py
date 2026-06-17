"""Gradio web demo for the Fashion-MNIST clothing classifier.

Usage:
    python demo.py
    make demo
"""
import os

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

from src.data.fashion_mnist import CLASS_NAMES

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
    arr = np.expand_dims(arr, axis=0)
    probs = _model.predict(arr, verbose=0)[0]
    return {name: float(p) for name, p in zip(CLASS_NAMES, probs)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload a clothing image"),
    outputs=gr.Label(num_top_classes=5, label="Predictions"),
    title="Fashion-MNIST Clothing Classifier",
    description=(
        "Upload a clothing image to classify it into one of 10 categories: "
        + ", ".join(CLASS_NAMES) + "."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
