import numpy as np
from layer import Layer
from activation import Activation

class Relu(Activation):
    def __init__(self):
        def relu(x):
            np.maximum(0, x)

        def relu_prime(x):
            return np.where(x > 0, 1, 0)

        super().__init__(relu, relu_prime)


class Sigmoid(Activation):
   def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            x = sigmoid(x)
            return x * (1 - x)
        
        super().__init__(sigmoid, sigmoid_prime)


class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2
        
        super().__init__(tanh, tanh_prime)


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


