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
        self.training = training  # Flag to indicate if the layer is in training mode
        self.mask = None
    
    def forward(self, input):
        if self.training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=input.shape)
            output = input * self.mask / (1 - self.rate)
            return output
        else:
            return input
    
    # output from the dropout layer is currently (3200,3200) when it should just be 3200, 1
    # i think it is because of the way the mask is being applied
    # might have to rethink how backpropogation works in this layer
    def backward(self, output_gradient, learning_rate):
        if self.training:
            self.mask = self.mask.reshape(output_gradient.shape)
            output = output_gradient * self.mask / (1 - self.rate)
            return output

        else:
            return output_gradient

