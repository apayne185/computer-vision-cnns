# CNN Models for Image Classification

Convolutional neural network implementations in TensorFlow/Keras, structured as a runnable Python package with a CLI training interface, YAML configs, and evaluation tooling.

## Models

| Model | Architecture | Dataset | Notes |
|---|---|---|---|
| `custom_cnn` | Conv2D → MaxPool → Dense | Fashion-MNIST | Built from scratch |
| `resnet34` | 34-layer residual network | Fashion-MNIST | Custom `ResidualUnit` layer |
| `xception` | Xception (ImageNet pretrained) | tf_flowers | Two-phase transfer learning |

## Project structure

```
src/
  models/         # model definitions (custom_cnn, resnet34, transfer)
  data/           # dataset loaders (fashion_mnist, flowers)
  utils/          # preprocessing and metrics helpers
configs/          # YAML hyperparameter configs per model
notebooks/        # original exploration notebooks
tests/            # pytest unit tests
train.py          # CLI training entrypoint
evaluate.py       # evaluation + confusion matrix
predict.py        # single-image inference
```

## Setup

> Requires Python 3.10 — TensorFlow does not support 3.11+.

```bash
pip install -r requirements-dev.txt
```

## Training

```bash
# Train the custom CNN
python train.py --config configs/custom_cnn.yaml --save saved_models/custom_cnn

# Train ResNet-34 from scratch
python train.py --config configs/resnet34.yaml --save saved_models/resnet34

# Transfer learning with Xception on tf_flowers
python train.py --config configs/xception.yaml --save saved_models/xception

# Or via Make
make train-cnn
```

Edit any `configs/*.yaml` file to change epochs, optimizer, learning rate, etc. without touching the code.

## Evaluation

```bash
python evaluate.py --model saved_models/custom_cnn --confusion-matrix
```

Prints per-class precision/recall/F1 and optionally renders a confusion matrix.

## Inference

```bash
python predict.py --model saved_models/custom_cnn --image my_image.png
```

## API (FastAPI)

Start the inference server locally:

```bash
make serve
# or: uvicorn api:app --reload
```

`POST /predict` — upload an image, get back a class label and confidence scores:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@my_image.png"
# {"class":"Sneaker","confidence":0.9821,"scores":{...}}
```

`GET /health` — liveness check.

## Docker

```bash
# Build and run
make docker-up

# Equivalent to:
docker compose up
```

The model is volume-mounted from `./saved_models` so you don't need to rebuild the image when you retrain. Set `MODEL_PATH` and `MODEL_TYPE` in `docker-compose.yml` to switch models.

## Tests

```bash
pytest tests/ -v
# or
make test
```

CI runs automatically on every push via GitHub Actions.

## Requirements

- tensorflow >= 2.12
- tensorflow-datasets
- scikit-learn, matplotlib, seaborn
- fastapi, uvicorn, python-multipart
- pyyaml, pillow
