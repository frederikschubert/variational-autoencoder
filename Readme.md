# Variational Autoencoder

This repository contains some experimental implementations of the Variational Autoencoder.

## Installation

```bash
conda env create -f environment.yml
```

## Usage

```bash
# Start a VAE on mnist using fully connected neural networks as encoder/decoder
python vae.py

# Start a VAE on mnist using convolutional neural networks as encoder/decoder
python vae.py with cnn

# Start a convolutional VAE on mnist with batch size 128
python vae.py with cnn "batch_size=128"
```
