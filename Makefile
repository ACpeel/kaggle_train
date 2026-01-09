.PHONY: dataset train help

help:
	@echo "Targets:"
	@echo "  dataset  - download dataset into kaggle_train/datasets"
	@echo "  train    - train ConvNeXtV2 classifier"

dataset:
	python scripts/get_dataset.py

train:
	python scripts/train.py

