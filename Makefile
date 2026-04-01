.PHONY: install train infer

install:
	pip install -e .

train:
	python -m scripts.train

infer:
	python -m scripts.infer

