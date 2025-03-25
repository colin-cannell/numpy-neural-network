import numpy as np
from layer import Layer

class Relu(Layer):
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x, learning_rate=None):
        return x > 0
    
class LeakyRelu(Layer):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        return np.where(x > 0, x, x * self.alpha)
    
    def backward(self, x, learning_rate=None):
        return np.where(x > 0, 1, self.alpha)

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x, learning_rate=None):
        s = self.sigmoid(x)
        return s * (1 - s)


class tanh:
    def __init__(self):
        super().__init__()

    def fowrard(self, x):
        return np.tanh(x)

    def backward(self, x, learning_rate=None):
        return 1 - np.tanh(x) ** 2


class Softmax:
    def __init__(self):
        super().__init__()

    def forward(self, input):
        input_stable = input - np.max(input, axis=0, keepdims=True)
        tmp = np.exp(input_stable)
        self.output = tmp / np.sum(tmp, axis=0, keepdims=True)
        return self.output

    def backward(self, output_gradient, learning_rate=None):
        output_neurons = output_gradient.shape[0]

        s = self.output.reshape(output_neurons, 1)
        jacobian = np.diagflat(s) - np.dot(s, s.T)
        output = np.dot(jacobian, output_gradient)

        return output

