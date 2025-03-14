import numpy as np
from layer import Layer

class Dense(Layer):
    """
    Dense layer is a fully connected layer that connects every neuron in the previous layer to every neuron in the next layer
    @param input_size: number of input neurons
    @param output_size: number of output neurons
    """
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) 
        self.biases = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input

        if self.input.ndim == 1:
            self.input = self.input.reshape(-1, 1)
        
        output = np.dot(self.weights, self.input) + self.biases
        return output
        
    def backward(self, output_gradient, learning_rate):        
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)

        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * np.sum(output_gradient, axis=1, keepdims=True)

        return input_gradient


