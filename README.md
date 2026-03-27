# Neural Networks from Scratch (NumPy)

## Project Overview
This project implements neural networks from scratch using NumPy, without relying on high-level deep learning frameworks.

It covers the full pipeline:
- Forward propagation
- Activation functions
- Loss computation
- Backpropagation
- Training loop with multiple optimizers

---

## Project Structure

- **nnfs/** - Core implementation
  - `__init__.py` - Module initializer
  - `layers.py` - Dense (fully connected) layers
  - `activations.py` - ReLU and Softmax activation functions
  - `losses.py` - Categorical cross-entropy loss
  - `optimizers.py` - Optimizers (SGD, Momentum, Adam, etc.)

- **notebooks/** - Step-by-step learning notebooks
  - `01_forward_pass.ipynb`
  - `02_layers_and_numpy.ipynb`
  - `03_activation.ipynb`
  - `04_loss.ipynb`
  - `05_training.ipynb`
  - `06_backpropagation.ipynb`
  - `full_code.ipynb`

- **examples/** - Applied classification tasks
  - `spiral_classification.py`
  - `vertical_classification.py`

- `requirements.txt` - Project dependencies
- `README.md` - This file

---

## Key Features

- Dense (fully connected) layers with forward and backward passes
- ReLU and Softmax activation functions
- Categorical Cross-Entropy loss
- Manual backpropagation implementation using chain rule
- Multiple optimizers: SGD, Momentum, Adagrad, RMSprop, Adam

---

## Installation

```bash
pip install numpy matplotlib nnfs
Quick Start
python
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from nnfs import Layer_Dense, Activation_ReLU, Activation_Softmax
from nnfs.losses import Loss_CategoricalCrossentropy
from nnfs.optimizers import Optimizer_Adam

nnfs.init()

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Build model
dense1 = Layer_Dense(2, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 3)
activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossentropy()
optimizer = Optimizer_Adam(learning_rate=0.02)

# Training loop
for epoch in range(10001):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.outputs)
    dense2.forward(activation1.outputs)
    activation2.forward(dense2.outputs)
    
    # Calculate loss and accuracy
    loss = loss_function.calculate(activation2.outputs, y)
    predictions = np.argmax(activation2.outputs, axis=1)
    accuracy = np.mean(predictions == y)
    
    if epoch % 1000 == 0:
        print(f"epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}")
    
    # Backward pass
    loss_function.backward(activation2.outputs, y)
    dense2.backward(loss_function.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    # Update weights
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
Example Output
text
epoch: 0, acc: 0.280, loss: 1.098
epoch: 1000, acc: 0.820, loss: 0.450
epoch: 2000, acc: 0.920, loss: 0.230
epoch: 3000, acc: 0.950, loss: 0.150
epoch: 4000, acc: 0.970, loss: 0.100
epoch: 5000, acc: 0.980, loss: 0.070
What I Learned
How neural networks operate mathematically at the fundamental level

How backpropagation applies the chain rule to compute gradients

How different optimizers (SGD, Momentum, Adam) affect convergence

How to implement ML systems without high-level frameworks

How to structure machine learning code properly for reusability

Future Improvements
Add dropout regularization to prevent overfitting

Implement batch normalization for faster convergence

Add visualization for loss curves to track training progress

Save and load model weights for reuse

Add more activation functions (Leaky ReLU, Tanh, Sigmoid)

Add learning rate scheduling for better optimization

Author
Mustapha Barrow
Electrical Engineering Student
Universitas Muhammadiyah Yogyakarta