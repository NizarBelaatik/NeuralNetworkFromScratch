import unittest
from neural_network import NeuralNetwork
from data.xor_data import get_xor_data

class TestNeuralNetwork(unittest.TestCase):
    def test_forward(self):
        X, y = get_xor_data()
        nn = NeuralNetwork(2, 2, 1)
        output = nn.forward(X)
        self.assertEqual(output.shape, (4, 1))

if __name__ == '__main__':
    unittest.main()
