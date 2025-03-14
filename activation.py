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

        # Check if activation is callable (i.e., a function)
        if not callable(self.activation):
            raise TypeError(f"The activation function should be callable, but got {type(self.activation)}")


    """
    Forward pass of the Activation layer
    @param input: input to the layer
    @return: output of the activation layer
    """
    def forward(self, input):
        self.input = input
        output = self.activation(input)
        return output
    
    def backward(self, output_gradient, learning_rate=None):
        return output_gradient * self.derivative(self.input)
