"""Evaluate a saved model on the Fashion-MNIST test set.

Usage:
    python evaluate.py --model saved_models/custom_cnn
    python evaluate.py --model saved_models/custom_cnn --confusion-matrix
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.data.fashion_mnist import CLASS_NAMES, load


def _plot_confusion_matrix(cm, class_names, save_path=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved confusion matrix to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to a saved .keras model")
    parser.add_argument("--confusion-matrix", action="store_true")
    parser.add_argument("--save-plot", default=None, help="Path to save the confusion matrix image")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    _, _, (X_test, y_test) = load()

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy : {accuracy:.4f}")
    print(f"Test Loss     : {loss:.4f}\n")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    if args.confusion_matrix:
        cm = confusion_matrix(y_test, y_pred)
        _plot_confusion_matrix(cm, CLASS_NAMES, save_path=args.save_plot)


if __name__ == "__main__":
    main()
