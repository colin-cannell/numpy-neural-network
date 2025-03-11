import numpy as np
from layer import Layer

class Dense(Layer):
    """
    Dense layer is a fully connected layer that connects every neuron in the previous layer to every neuron in the next layer
    @param input_size: number of input neurons
    @param output_size: number of output neurons
    """
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(output_size)

    def forward(self, input):
        self.input_shape = input.shape  # Save original shape for backward pass
        return input.reshape(input.shape[0], -1)  # Keep batch size
    
    def backward(self, output_gradient, learning_rate):
        return output_gradient.reshape(self.input_shape)  # Restore original shape
    
