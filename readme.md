# Medical Image Classification for MedMNIST

This project implements a Vision Transformer (ViT) model to classify medical images, specifically targeting the PathMNIST dataset from the MedMNIST collection. It provides a modularized codebase for training, evaluating, and inferring on medical image data.

## Project Structure

- `core/`: Core configuration and factory abstractions for model creation.
- `data/`: Dataset and dataloader implementations for MedMNIST.
- `model/`: Vision Transformer (ViT) architecture components.
- `scripts/`: Execution scripts for training and inference.
- `config/`: TOML configuration files for model and training hyperparameters.

## Quick Start

### Installation

Clone the repository and install the dependencies:

```bash
pip install -e .
```

### Configuration

Modify `config/config.toml` to adjust the training parameters, model architecture, or dataset settings.

### Training

Start the training process with Weights & Biases (wandb) logging. You can use either `make` or the standard `python` command:

**Using Make:**
```bash
make train
```

**Using Python:**
```bash
python -m scripts.train
```

### Inference

Generate predictions using a trained model's weights:

**Using Make:**
```bash
make infer
```

**Using Python:**
```bash
python -m scripts.infer
```

### Code Formatting and Linting

Maintain code quality with `black` and `isort`:

**Using Make:**
```bash
make lint
```

**Using Python:**
```bash
black . && isort .
```

## Features

- Modular ViT implementation with customizable patch sizes, embedding dimensions, and attention heads.
- Integrated training pipeline with Mixed Precision (AMP) and gradient clipping.
- Real-time logging and visualization with Weights & Biases.
- Support for multiple hardware backends (CPU, CUDA, MPS).

## Dataset

This project currently uses **PathMNIST**, a collection of histological images for colorectal cancer classification. The dataset is automatically downloaded and processed using the `medmnist` library.

## Requirements

- Python 3.11 or later
- PyTorch
- MedMNIST
- Weights & Biases
- TOML support (built-in for Python 3.11+)
