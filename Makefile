.PHONY: install train infer lint clean

install:
	pip install -e .

train:
	python -m scripts.train

infer:
	python -m scripts.infer

