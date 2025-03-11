import numpy as np
from layer import Layer

"""
Activation layer applies an activation function to the input
"""
class Activation(Layer):
    """
    @param activation: activation function
    @param derivative: derivative of the activation function
    """
    def __init__(self, activation, derivative):
        self.activation = activation
        self.derivative = derivative

    """
    Forward pass of the Activation layer
    @param input: input to the layer
    @return: output of the activation layer
    """
    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.derivative(self.input))