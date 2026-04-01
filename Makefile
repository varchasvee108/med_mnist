.PHONY: install train infer lint clean

install:
	pip install -e .

train:
	python -m scripts.train

infer:
	python -m scripts.infer

lint:
	black .
	isort .

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info
