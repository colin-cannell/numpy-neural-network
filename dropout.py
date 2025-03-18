import numpy as np
from layer import Layer

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
            self.dropout_effect(self.input, output, "Dropout Layer Forward Pass")
            return output
        else:
            return input
    
    # output from the dropout layer is currently (3200,3200) when it should just be 3200, 1
    # i think it is because of the way the mask is being applied
    # might have to rethink how backpropogation works in this layer
    def backward(self, output_gradient, learning_rate):
        input_gradient = output_gradient * self.mask
        return input_gradient

