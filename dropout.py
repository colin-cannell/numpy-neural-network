import numpy as np
from layer import Layer
from visualize import dropout

class Dropout(Layer):
    """
    Dropout layer randomly sets a fraction of the input to 0 during training to provent over fitting
    @param rate: fraction of the input to set to 0 
    """
    def __init__(self, rate=0.5, training=True):
        if not 0 <= rate <= 1:
            raise ValueError("Dropout rate must be between 0 and 1")
        
        self.rate = rate
        self.mask = None
    
    def forward(self, input, training=True):
        if training:
            self.mask = (np.random.rand(*input.shape) < (1 - self.rate))
            output = input * self.mask
            # dropout.dropout_effect(input, output, "Dropout Layer Forward Pass")
            return output
        else:
            return input
    
    def backward(self, output_gradient, learning_rate):
        input_gradient = output_gradient * self.mask
        return input_gradient
    

"""
value encountered in subtract
input_stable = input - np.max(input, axis=0, keepdims=True)

/Users/colincannell/NumpyNeuralNetwork/losses.py:15: RuntimeWarning: invalid value encountered in cast
  correct_class_prob = np.log(y_pred[np.arange(y_true.shape[0]), y_true.flatten().astype(int)])


"""

