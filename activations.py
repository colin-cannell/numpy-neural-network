import numpy as np
from layer import Layer
from activation import Activation

class Relu:
    def relu(self, x):
        return np.maximum(0, x)

    def relu_prime(self, x):
        return x > 0


class Sigmoid:
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)


class Tanh:
    def tanh(self, x):
        return np.tanh(x)

    def tanh_prime(self, x):
        return 1 - np.tanh(x) ** 2


class Softmax(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        input_stable = input - np.max(input, axis=0, keepdims=True)
        tmp = np.exp(input_stable)
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward(self, output_gradient, learning_rate=None):
        num_classes, num_neurons = output_gradient.shape
        softmax_grid = np.zeros_like(self.output)

        for i in range(num_neurons):
           s = self.output[:, i].reshape(-1, 1)
           jacobian = np.diagflat(s) - np.dot(s, s.T)
           softmax_grid[:, i] = np.dot(jacobian, output_gradient[:, i])

        return softmax_grid

class CrossEntropyLoss:
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-15

    def forward(self, y_true, y_pred):
        y_true = y_true.reshape(-1, 1)
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

    def backward(self, y_true, y_pred):
        y_true = y_true.reshape(-1, 1)
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

class MeanSquaredErrorLoss(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        loss = np.mean((y_true - y_pred) ** 2)
        return loss

    def backward(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.shape[0]
