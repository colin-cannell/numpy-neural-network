import numpy as np
from layer import Layer

def safe_log(x):
    # Replace any values that are 0 or negative with a small positive value to prevent log(0) or log(negative)
    x = np.clip(x, 1e-10, None)  # Clip to prevent log(0) or log(negative values)
    return np.log(x)

class CategoricalCrossEntropyLoss:
    def __init__(self):
        self.epsilon = 1e-15

    def forward(self, y_pred, y_true):
        # Ensure y_pred is in the right shape
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)  # Clip to prevent log(0)

        # Assuming y_true is a one-hot encoded vector (or batch)
        # Correct class probabilities are selected using y_true as index
        correct_class_prob = np.sum(y_true * safe_log(y_pred), axis=0)

        # Calculate the loss (Mean Cross-Entropy Loss)
        output = -np.mean(correct_class_prob)
        return output

    def backward(self, y_pred, y_true):
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)  # Clip to prevent log(0)

        # Compute the gradient for each class
        grad = np.zeros_like(y_pred)  # Initialize gradient

        # Backpropagation: Gradient calculation for each sample
        grad = -y_true / y_pred  # For one-hot encoded labels, we divide by y_pred for the correct class
        
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
