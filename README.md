# Neural Network from Scratch

This project implements a neural network from scratch to solve the XOR problem, a classic problem in machine learning that demonstrates the capabilities of neural networks to learn non-linear decision boundaries.


## Directory Structure
   - Project structure:
     ```
     NeuralNetworkFromScratch/
     ├── src/
     │   ├── neural_network.py
     │   ├── layers.py
     │   ├── activation_functions.py
     │   └── loss_functions.py
     ├── data/
     │   └── xor_data.py
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

# Usage
To use the neural network, follow these steps:

```python
    from src.neural_network import NeuralNetwork
    from data.xor_data import get_xor_data

    # Load XOR data
    X, y = get_xor_data()

    # Create an instance of the NeuralNetwork
    nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1)

    # Train the network
    nn.train(X, y, epochs=1000, learning_rate=0.01)

    # Make predictions
    predictions = nn.forward(X)
    print(predictions)
```

## Running Tests

To run the tests, execute:
```
python -m unittest discover tests/
```
