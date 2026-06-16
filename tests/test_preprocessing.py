import numpy as np
import tensorflow as tf

from src.utils.preprocessing import central_crop, normalize


def test_normalize_bounds():
    X = tf.constant([[0.0, 5.0], [10.0, 3.0]])
    result = normalize(X).numpy()
    assert result.min() >= 0.0
    assert result.max() <= 1.0 + 1e-6


def test_normalize_known_values():
    X = tf.constant([0.0, 10.0])
    result = normalize(X).numpy()
    np.testing.assert_allclose(result, [0.0, 1.0], atol=1e-6)


def test_central_crop_is_square():
    image = tf.zeros([100, 150, 3])
    cropped = central_crop(image)
    shape = tf.shape(cropped).numpy()
    assert shape[0] == shape[1]


def test_central_crop_landscape_and_portrait():
    for h, w in [(200, 100), (100, 200)]:
        image = tf.zeros([h, w, 3])
        cropped = central_crop(image)
        s = tf.shape(cropped).numpy()
        assert s[0] == s[1]
