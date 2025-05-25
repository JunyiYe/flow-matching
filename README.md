# Handwritten Image Generation with Flow Matching

This project implements a Conditional U-Net trained using a flow matching objective on the MNIST dataset.

## Features

- Conditional embedding using class labels and continuous time
- U-Net inspired encoder-decoder architecture
- Flow-based training mechanism for image generation

## Usage

### Train

```bash
python train.py
```

### Inference

```bash
python infer.py
```

## Requirements

```bash
pip install -r requirements.txt
```
