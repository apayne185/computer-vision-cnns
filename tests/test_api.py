"""API tests using FastAPI's TestClient (no live server needed)."""
import io
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

os.environ.setdefault("MODEL_PATH", "saved_models/custom_cnn.keras")
os.environ.setdefault("MODEL_TYPE", "fashion_mnist")


@pytest.fixture(scope="module")
def client():
    from api import app
    with TestClient(app) as c:
        yield c


def _make_png_bytes(size=(28, 28)) -> bytes:
    arr = (np.random.rand(*size) * 255).astype("uint8")
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_returns_class_and_confidence(client):
    resp = client.post("/predict", files={"file": ("test.png", _make_png_bytes(), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert "class" in body
    assert "confidence" in body
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_scores_sum_to_one(client):
    resp = client.post("/predict", files={"file": ("test.png", _make_png_bytes(), "image/png")})
    scores = resp.json()["scores"]
    assert abs(sum(scores.values()) - 1.0) < 1e-3


def test_predict_rejects_non_image(client):
    resp = client.post("/predict", files={"file": ("data.csv", b"a,b,c", "text/csv")})
    assert resp.status_code == 400
