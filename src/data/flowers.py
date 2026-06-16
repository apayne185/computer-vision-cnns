from functools import partial

import tensorflow_datasets as tfds

from src.utils.preprocessing import preprocess


def load(batch_size: int = 32):
    """Return train/valid/test tf.data pipelines plus dataset metadata."""
    _, info = tfds.load("tf_flowers", as_supervised=True, with_info=True)
    n_classes = info.features["label"].num_classes
    dataset_size = info.splits["train"].num_examples
    class_names = info.features["label"].names

    test_raw, valid_raw, train_raw = tfds.load(
        "tf_flowers",
        split=["train[:10%]", "train[10%:25%]", "train[25%:]"],
        as_supervised=True,
    )

    train_set = (
        train_raw.shuffle(1000)
        .repeat()
        .map(partial(preprocess, randomize=True))
        .batch(batch_size)
        .prefetch(1)
    )
    valid_set = valid_raw.map(preprocess).batch(batch_size).prefetch(1)
    test_set = test_raw.map(preprocess).batch(batch_size).prefetch(1)

    return train_set, valid_set, test_set, n_classes, dataset_size, class_names
