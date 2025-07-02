import numpy as np

class Sigmoid:
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, output_gradient):
        return output_gradient * (self.output * (1 - self.output))
