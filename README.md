# Neural Network from Scratch

This project is an implementation of a neural network from scratch to solve the XOR problem, a classic problem in machine learning that demonstrates the capabilities of neural networks to learn non-linear decision boundaries.

## Directory Structure
   - Project structure:
     ```
     NeuralNetworkFromScratch/
     ├── src/
     │   ├── layer.py
     │   ├── fc_layer.py
     │   ├── activation_layer.py
     │   ├── activations.py
     │   ├── losses.py
     │   └── network.py
     ├── tests/
     │   └── test_neural_network.py
     ├── README.md
     └── requirements.txt
     ```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/NizarBelaatik/NeuralNetworkFromScratch
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To use the neural network, follow these steps:

```python
from src.network import Network
from src.fc_layer import FCLayer
from src.activation_layer import ActivationLayer
from src.activations import tanh, tanh_prime
from src.losses import mse, mse_prime

# Training data (XOR)
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

# Create network
net = Network()
net.add(FCLayer(2, 3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(tanh, tanh_prime))

# Compile network
net.use(mse, mse_prime)

# Train
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# Test
out = net.predict(x_train)
print(out)
```

## Running Tests

To run the tests, execute:
```
python -m unittest discover tests/
```
