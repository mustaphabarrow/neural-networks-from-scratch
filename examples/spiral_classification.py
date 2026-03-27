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