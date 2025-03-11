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
        return input.reshape(input.shape[0], -1)  # Preserve batch size


    """
    Backward pass of the Flatten layer
    @param output_gradient: gradient of the loss with respect to the output
    @param learning_rate: learning rate
    @return: gradient of the input
    """
    def backward(self, output_gradient, learning_rate):
        return output_gradient.reshape(self.input_shape)  # Restore original shape
