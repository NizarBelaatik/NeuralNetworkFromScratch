class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    
    def forward_propagation(self, input):
        """Computes the output Y of a layer for a given input X"""
        raise NotImplementedError

    def backward_propagation(self, output_error, learning_rate):
        """Computes dE/dX for a given dE/dY"""
        raise NotImplementedError
