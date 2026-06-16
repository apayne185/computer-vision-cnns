.PHONY: train-cnn train-resnet train-xception evaluate test lint

train-cnn:
	python train.py --config configs/custom_cnn.yaml --save saved_models/custom_cnn

train-resnet:
	python train.py --config configs/resnet34.yaml --save saved_models/resnet34

train-xception:
	python train.py --config configs/xception.yaml --save saved_models/xception

evaluate:
	python evaluate.py --model saved_models/custom_cnn --confusion-matrix

test:
	pytest tests/ -v

lint:
	ruff check src/ train.py evaluate.py predict.py
