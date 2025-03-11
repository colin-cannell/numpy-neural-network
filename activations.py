import numpy as np
from layer import Layer
from activation import Activation

class Relu:
    def relu(self, x):
        return np.maximum(0, x)

    def relu_prime(self, x):
        return np.where(x > 0, 1, 0)


class Sigmoid:
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        x = 1 / (1 + np.exp(-x))
        return x * (1 - x)


class Tanh:
    def tanh(self, x):
        return np.tanh(x)

    def tanh_prime(self, x):
        return 1 - np.tanh(x) ** 2


class Softmax(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.input = input
        exp_input = np.exp(input - np.max(input, axis=-1, keepdims=True))  # For numerical stability
        self.output = exp_input / np.sum(exp_input, axis=-1, keepdims=True)
        return self.output

    def backward(self, output_gradient, learning_rate):
        n = self.output.shape[1]
        jacobian = np.diagflat(self.output) - np.outer(self.output, self.output)
        return np.dot(jacobian, output_gradient)


class CrossEntropyLoss(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        # Clip values to avoid log(0) issues
        self.y_true = y_true
        self.y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        loss = -np.sum(y_true * np.log(self.y_pred)) / y_true.shape[0]
        return loss

    def backward(self, output_gradient, learning_rate):
        # Gradient of cross-entropy loss with respect to y_pred
        return (self.y_pred - self.y_true) / self.y_true.shape[0]


class MeanSquaredErrorLoss(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        loss = np.mean((y_true - y_pred) ** 2)
        return loss

    def backward(self, output_gradient, learning_rate):
        return 2 * (self.y_pred - self.y_true) / self.y_true.shape[0]
