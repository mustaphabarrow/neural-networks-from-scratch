"""
Neural Networks from Scratch (NumPy)
A complete implementation of neural networks using only NumPy.
"""

from .layers import Layer_Dense
from .activations import Activation_ReLU, Activation_Softmax
from .losses import Loss, Loss_CategoricalCrossentropy
from .optimizers import (
    Optimizer_SGD,
    Optimizer_Adagrad,
    Optimizer_RMSprop,
    Optimizer_Adam
)

__version__ = "1.0.0"

__all__ = [
    'Layer_Dense',
    'Activation_ReLU',
    'Activation_Softmax',
    'Loss',
    'Loss_CategoricalCrossentropy',
    'Optimizer_SGD',
    'Optimizer_Adagrad',
    'Optimizer_RMSprop',
    'Optimizer_Adam'
]