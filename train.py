"""CLI entrypoint: python train.py --config configs/custom_cnn.yaml"""
import argparse

import yaml
from tensorflow import keras

from src.data import fashion_mnist, flowers
from src.models import custom_cnn, resnet34, transfer


def _build_optimizer(cfg: dict):
    kind = cfg.get("type", "adam").lower()
    if kind == "sgd":
        return keras.optimizers.SGD(
            learning_rate=cfg.get("learning_rate", 0.01),
            momentum=cfg.get("momentum", 0.0),
            nesterov=cfg.get("nesterov", False),
            decay=cfg.get("decay", 0.0),
        )
    if kind == "nadam":
        return keras.optimizers.Nadam(learning_rate=cfg.get("learning_rate", 1e-3))
    if kind == "adam":
        return keras.optimizers.Adam(learning_rate=cfg.get("learning_rate", 1e-3))
    raise ValueError(f"Unknown optimizer type: {kind!r}")


def train_custom_cnn(cfg: dict):
    mp = cfg["model_params"]
    tc = cfg["training"]

    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = fashion_mnist.load()

    model = custom_cnn.build(
        input_shape=tuple(mp["input_shape"]),
        n_classes=mp["n_classes"],
        dropout=mp.get("dropout", 0.5),
    )
    model.compile(loss=tc["loss"], optimizer=tc.get("optimizer", "nadam"), metrics=tc["metrics"])
    history = model.fit(X_train, y_train, epochs=tc["epochs"], validation_data=(X_valid, y_valid))

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc:.4f}  |  Test loss: {loss:.4f}")
    return model, history


def train_resnet34(cfg: dict):
    mp = cfg["model_params"]
    tc = cfg["training"]
    resize = tuple(mp.get("resize_to", [224, 224]))

    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = fashion_mnist.load(resize_to=resize)

    model = resnet34.build(input_shape=tuple(mp["input_shape"]), n_classes=mp["n_classes"])
    model.compile(loss=tc["loss"], optimizer=tc.get("optimizer", "adam"), metrics=tc["metrics"])
    history = model.fit(
        X_train, y_train,
        epochs=tc["epochs"],
        batch_size=tc.get("batch_size", 32),
        validation_data=(X_valid, y_valid),
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc:.4f}  |  Test loss: {loss:.4f}")
    return model, history


def train_xception(cfg: dict):
    tc = cfg["training"]
    train_set, valid_set, _, n_classes, dataset_size, _ = flowers.load(batch_size=tc["batch_size"])

    model, base_model = transfer.build(n_classes=n_classes, base=cfg.get("base", "xception"))

    # Phase 1: train only the new head
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(
        loss=tc["loss"],
        optimizer=_build_optimizer(tc["freeze_optimizer"]),
        metrics=tc["metrics"],
    )
    model.fit(
        train_set,
        steps_per_epoch=int(0.75 * dataset_size / tc["batch_size"]),
        validation_data=valid_set,
        validation_steps=int(0.15 * dataset_size / tc["batch_size"]),
        epochs=tc["freeze_epochs"],
    )

    # Phase 2: unfreeze and fine-tune the whole network
    for layer in base_model.layers:
        layer.trainable = True
    model.compile(
        loss=tc["loss"],
        optimizer=_build_optimizer(tc["finetune_optimizer"]),
        metrics=tc["metrics"],
    )
    history = model.fit(
        train_set,
        steps_per_epoch=int(0.75 * dataset_size / tc["batch_size"]),
        validation_data=valid_set,
        validation_steps=int(0.15 * dataset_size / tc["batch_size"]),
        epochs=tc["finetune_epochs"],
    )
    return model, history


_TRAINERS = {
    "custom_cnn": train_custom_cnn,
    "resnet34": train_resnet34,
    "xception": train_xception,
}


def main():
    parser = argparse.ArgumentParser(description="Train a CNN model")
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    parser.add_argument("--save", default=None, help="Directory to save the trained model")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]
    if model_name not in _TRAINERS:
        raise ValueError(f"Unknown model {model_name!r}. Choose from: {list(_TRAINERS)}")

    print(f"Training {model_name} ...")
    model, _ = _TRAINERS[model_name](cfg)

    if args.save:
        model.save(args.save)
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
