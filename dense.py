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
        self.input = input
        self.input_shape = input.shape  # Save original shape for backward pass
        # Flatten the input to be a 2D array (batch_size, input_size)
        flattened_input = input.reshape(input.shape[0], -1)
        # Perform matrix multiplication (input * weights) + bias
        output = np.dot(flattened_input, self.weights) + self.bias
        print(f"Dense layer output shape: {output.shape}")
        return output
        
    def backward(self, output_gradient, learning_rate):
        # Compute gradients for weights and bias
        
        weights_gradient = np.dot(self.input.T, output_gradient)  # Derivative w.r.t weights
        input_gradient = np.dot(output_gradient, self.weights.T)  # Derivative w.r.t input

        # Update weights and biases
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * np.sum(output_gradient, axis=0)

        return input_gradient


