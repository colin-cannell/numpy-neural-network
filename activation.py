import numpy as np
from layer import Layer

"""
Activation layer applies an activation function to the input
@param activation: activation function
@param derivative: derivative of the activation function
"""
class Activation(Layer):
    def __init__(self, activation, derivative):
        self.activation = activation
        self.derivative = derivative
