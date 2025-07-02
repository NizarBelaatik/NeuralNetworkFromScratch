import unittest
import numpy as np
from src.network import Network
from src.fc_layer import FCLayer
from src.activation_layer import ActivationLayer
from src.activations import tanh, tanh_prime
from src.losses import mse, mse_prime

class TestNeuralNetwork(unittest.TestCase):

    def setUp(self):
        # Set up a simple network for testing
        self.net = Network()
        self.net.add(FCLayer(2, 3))
        self.net.add(ActivationLayer(tanh, tanh_prime))
        self.net.add(FCLayer(3, 1))
        self.net.add(ActivationLayer(tanh, tanh_prime))
        self.net.use(mse, mse_prime)

    def test_forward_propagation(self):
        # Test forward propagation
        x = np.array([[0, 0]])
        output = self.net.predict(x)
        self.assertEqual(output[0].shape, (1, 1))

    def test_loss_function(self):
        # Test loss function
        y_true = np.array([[0]])
        y_pred = np.array([[0.1]])
        loss = mse(y_true, y_pred)
        self.assertAlmostEqual(loss, 0.01)

    def test_training(self):
        # Test training process
        x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_train = np.array([[0], [1], [1], [0]])
        self.net.fit(x_train, y_train, epochs=1, learning_rate=0.1)
        output = self.net.predict(x_train)
        self.assertEqual(len(output), 4)

if __name__ == '__main__':
    unittest.main()
