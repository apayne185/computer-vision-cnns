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


def predict(image):
    img = Image.fromarray(image).convert("L").resize((28, 28))
    arr = np.array(img, dtype="float32")[..., np.newaxis] / 255.0
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
