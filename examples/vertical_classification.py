import numpy as np
from nnfs.datasets import vertical_data

X, y = vertical_data(samples=100, classes=3)

from nnfs.layers import Layer_Dense
from nnfs.activations import Activation_ReLU, Activation_Softmax
from nnfs.losses import Loss_CategoricalCrossentropy
from nnfs.optimizers import Optimizer_SGD

nnfs.init()

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create model
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossentropy()
optimizer = Optimizer_SGD(learning_rate=0.1)

# Training loop
for epoch in range(1000):

    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.outputs)

    dense2.forward(activation1.outputs)
    activation2.forward(dense2.outputs)

    loss = loss_function.calculate(activation2.outputs, y)

    predictions = np.argmax(activation2.outputs, axis=1)
    accuracy = np.mean(predictions == y)

    if epoch % 100 == 0:
        print(f"epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}")

    # Backward pass
    loss_function.backward(activation2.outputs, y)
    dense2.backward(loss_function.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)