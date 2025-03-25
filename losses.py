import numpy as np
from layer import Layer

def safe_log(x):
    # Replace any values that are 0 or negative with a small positive value to prevent log(0) or log(negative)
    x = np.clip(x, 1e-10, None)  # Clip to prevent log(0) or log(negative values)
    return np.log(x)

class CategoricalCrossEntropyLoss:
    def __init__(self):
        self.epsilon = 1e-15

    def forward(self, y_true, y_pred):
        y_pred = y_pred.T
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)

        y_true = y_true.reshape(-1, 1)  # Shape becomes (10, 1)
        y_pred = y_pred.reshape(-1, 1)  # Shape becomes (10, 1)

        correct_class_prob = safe_log(y_pred[np.arange(y_true.shape[0]), y_true.flatten().astype(int)])
        output = -np.mean(correct_class_prob)
        return output

    def backward(self, y_true, y_pred):
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)  # Prevent log(0)
        
        grad = np.zeros_like(y_pred)  # Initialize gradient as zero
        
        # Since we have 10 samples (batch size = 1), we compute the gradient for each class
        for i in range(y_true.shape[0]):  # Loop through each class
            grad[i] = -y_true[i] / y_pred[i]  # Update gradient for the correct class

        return grad  # Gradient with respect to y_pred



class SparseCategoricalCrossEntropyLoss:
    def __init__(self):
        self.epsilon = 1e-15

    def forward(self, y_true, y_pred):
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return -np.mean(np.log(y_pred[np.arange(len(y_true)), y_true]))

    def backward(self, y_true, y_pred):
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        y_pred[np.arange(len(y_true)), y_true] -= 1
        return y_pred / len(y_true)

class BinaryCrossEntropyLoss:
    def __init__(self):
        self.epsilon = 1e-15

    def forward(self, y_true, y_pred):
        y_true = y_true.reshape(-1, 1)
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

    def backward(self, y_true, y_pred):
        y_true = y_true.reshape(-1, 1)
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

class MeanSquaredErrorLoss:
    def forward(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        loss = np.mean((y_true - y_pred) ** 2)
        return loss

    def backward(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.shape[0]
