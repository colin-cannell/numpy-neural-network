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

"""
Relu activation function introduces non-linearity in the model
@ param x: input
"""
def relu(x):
    return np.maximum(0, x)

"""
Sigmoid activation function takes in any real number and returns the output between 0 and 1
@ param x: input
"""
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

"""
Tanh activation function takes in any real number and returns the output between -1 and 1
@ param x: input
"""
def tanh(x):
    return np.tanh(x)

"""
Softmax activation function takes in any real number and returns the output between 0 and 1
@ param x: input
"""
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

"""
Cross entropy loss function
@param y_true: true labels
@param y_pred: predicted labels
"""
def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-10)) / y_true.shape[0]

"""
Cross entropy loss derivative
@param y_true: true labels
@param y_pred: predicted labels
"""
def cross_entropy_loss_derivative(y_true, y_pred):
    return -y_true / (y_pred + 1e-10)

"""
Mean squared error loss function
@param y_true: true labels
@param y_pred: predicted labels
"""
def mean_squared_error_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

"""
Mean squared error loss derivative
@param y_true: true labels
@param y_pred: predicted labels
"""
def mean_squared_error_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size