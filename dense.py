import numpy as np
from layer import Layer

class Dense(Layer):
    """
    Dense layer is a fully connected layer that connects every neuron in the previous layer to every neuron in the next layer
    @param input_size: number of input neurons
    @param output_size: number of output neurons
    """
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.input_shape = np.reshape((input_size,)[0], -1)

        self.output_shape = (output_size, )

    def forward(self, input):
        self.input = input

        output = np.dot(self.input, self.weights) + self.biases
        return output
        
    def backward(self, output_gradient, learning_rate):
        # Compute gradients for weights and bias
        input_reshaped = self.input.reshape(self.input_shape)  # Flatten the input
        
        weights_gradient = np.dot(input_reshaped.T, output_gradient)
        input_gradient = np.dot(output_gradient, self.weights.T)

        # Update weights and biases
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * np.sum(output_gradient, axis=0)

        return input_gradient


