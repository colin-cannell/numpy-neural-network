import numpy as np
from layer import Layer

"""
    Flatten layer flattens the input image into a 1D array while preserving batch size.
 """
class Flatten(Layer):
    """
    Forward pass of the Flatten layer
    @param input: input image
    @return: flattened output
    """
    def forward(self, input):
        self.input_shape = input.shape  # Save original shape for backprop
        self.output = input.reshape(input.shape[0], -1)  # Flatten input
        return self.output


    """
    Backward pass of the Flatten layer
    @param output_gradient: gradient of the loss with respect to the output
    @param learning_rate: learning rate
    @return: gradient of the input
    """
    def backward(self, output_gradient, learning_rate):
        # Gradient with respect to the weights and input
        flattened_input = self.input_shape.reshape(self.input_shape[0], -1)
        weights_gradient = np.dot(flattened_input.T, output_gradient)  # Derivative w.r.t weights
        input_gradient = np.dot(output_gradient, self.weights.T)  # Derivative w.r.t input

        # Update weights and biases
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * np.sum(output_gradient, axis=0)

        return input_gradient