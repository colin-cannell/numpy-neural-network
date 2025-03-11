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
        self.output = self.activation(input)
        return self.output
    
    def backward(self, output_error, learning_rate):
        return output_error * self.derivative(self.input)