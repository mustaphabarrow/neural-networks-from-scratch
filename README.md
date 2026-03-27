# Neural Networks from Scratch (NumPy)

## Project Overview
This project implements neural networks from scratch using NumPy, without relying on high-level deep learning frameworks.

It covers the full pipeline:
- Forward propagation
- Activation functions
- Loss computation
- Backpropagation
- Training loop with multiple optimizers

## Project Structure
neural-networks-from-scratch/
│
├── nnfs/ # Core implementation
│ ├── init.py # Module initializer
│ ├── layers.py # Dense (fully connected) layers
│ ├── activations.py # ReLU and Softmax activation functions
│ ├── losses.py # Categorical cross-entropy loss
│ └── optimizers.py # Optimizers (SGD, Momentum, Adam, etc.)
│
├── notebooks/ # Step-by-step learning notebooks
│ ├── 01_forward_pass.ipynb
│ ├── 02_layers_and_numpy.ipynb
│ ├── 03_activation.ipynb
│ ├── 04_loss.ipynb
│ ├── 05_training.ipynb
│ ├── 06_backpropagation.ipynb
│ └── full_code.ipynb
│
├── examples/ # Applied classification tasks
│ ├── spiral_classification.py
│ └── vertical_classification.py
│
├── requirements.txt # Project dependencies
└── README.md # This file

text

## Key Features
- Dense (fully connected) layers with forward and backward passes
- ReLU and Softmax activation functions
- Categorical Cross-Entropy loss
- Manual backpropagation implementation using chain rule
- Multiple optimizers: SGD, Momentum, Adagrad, RMSprop, Adam

## Installation

```bash
pip install numpy matplotlib nnfs