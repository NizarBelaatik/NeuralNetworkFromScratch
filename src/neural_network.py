import numpy as np
from layers import FullyConnectedLayer
from activation_functions import Sigmoid
from loss_functions import MeanSquaredError

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.fc1 = FullyConnectedLayer(input_size, hidden_size)
        self.activation = Sigmoid()
        self.fc2 = FullyConnectedLayer(hidden_size, output_size)
        self.loss_function = MeanSquaredError()

    def forward(self, X):
        self.z1 = self.fc1.forward(X)
        self.a1 = self.activation.forward(self.z1)
        self.z2 = self.fc2.forward(self.a1)
        return self.z2

    def backward(self, X, y):
        loss_derivative = self.loss_function.backward(y, self.z2)
        dz2 = self.fc2.backward(loss_derivative)
        da1 = self.activation.backward(dz2)
        self.fc1.backward(da1)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y)
            # Update weights here (not shown for brevity)
