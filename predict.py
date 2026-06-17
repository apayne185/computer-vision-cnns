"""Run inference on a single image.

Usage:
    python predict.py --model saved_models/custom_cnn.keras --image my_shirt.png
"""
import argparse
import os

import numpy as np
import tensorflow as tf
from PIL import Image

from src.data.fashion_mnist import CLASS_NAMES


def _load_norm_stats(model_path: str):
    stats_path = model_path.replace(".keras", "_norm_stats.npz")
    if os.path.exists(stats_path):
        data = np.load(stats_path)
        return data["X_mean"], data["X_std"]
    return None, None


def _load_image(path: str, norm_mean=None, norm_std=None, target_size=(28, 28)) -> np.ndarray:
    img = Image.open(path).convert("L").resize(target_size)
    arr = np.array(img, dtype="float32")[..., np.newaxis]
    if norm_mean is not None and norm_std is not None:
        arr = (arr - norm_mean) / norm_std
    else:
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to a saved Keras model")
    parser.add_argument("--image", required=True, help="Path to an input image")
    args = parser.parse_args()

    norm_mean, norm_std = _load_norm_stats(args.model)
    model = tf.keras.models.load_model(args.model)
    img = _load_image(args.image, norm_mean=norm_mean, norm_std=norm_std)
    probs = model.predict(img, verbose=0)[0]

    top_idx = int(np.argmax(probs))
    print(f"Prediction: {CLASS_NAMES[top_idx]}  ({probs[top_idx] * 100:.1f}%)\n")
    print("All classes:")
    for name, p in sorted(zip(CLASS_NAMES, probs), key=lambda x: -x[1]):
        bar = "#" * int(p * 30)
        print(f"  {name:<15} {p * 100:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
