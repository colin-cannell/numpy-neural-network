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
        self.bias = np.random.randn(output_size)

    def forward(self, input):
        self.input = input
        print(f"Input shape to Dense: {self.input.shape}")
        self.input_shape = input.shape

        reshaped_input = input.reshape(input.shape[0], -1)  # Flatten each input image into a vector
        reshaped_input = reshaped_input.T  # Transpose to shape (1, 9216)

        try:
            output = np.dot(reshaped_input, self.weights) + self.bias
        except:
            output = np.dot(reshaped_input.T, self.weights) + self.bias

        print(f"Output shape from Dense: {output.shape}")
        return output
        
    def backward(self, output_gradient, learning_rate):
        # Compute gradients for weights and bias
        
        self.input = self.input.reshape(self.input_shape[0], -1)  # Flatten each input image into a vector

        try:
            weights_gradient = np.dot(self.input, output_gradient)  # Derivative w.r.t weights
        except:
            weights_gradient = np.dot(self.input.T, output_gradient)  # Derivative w.r.t weights

        try:
            input_gradient = np.dot(output_gradient, self.weights)  # Derivative w.r.t input
        except:
            input_gradient = np.dot(output_gradient, self.weights.T)  # Derivative w.r.t input
        

        # Update weights and biases
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * np.sum(output_gradient, axis=0)

        return input_gradient


