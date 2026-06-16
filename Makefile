.PHONY: train-cnn train-resnet train-xception evaluate serve docker-build docker-up docker-down test lint

train-cnn:
	python train.py --config configs/custom_cnn.yaml --save saved_models/custom_cnn.keras

train-resnet:
	python train.py --config configs/resnet34.yaml --save saved_models/resnet34.keras

train-xception:
	python train.py --config configs/xception.yaml --save saved_models/xception.keras

evaluate:
	python evaluate.py --model saved_models/custom_cnn.keras --confusion-matrix

serve:
	uvicorn api:app --reload

docker-build:
	docker build -t cv-cnns-api .

docker-up:
	docker compose up

docker-down:
	docker compose down

test:
	pytest tests/ -v

lint:
	ruff check src/ train.py evaluate.py predict.py
