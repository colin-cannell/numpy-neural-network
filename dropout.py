import numpy as np
from layer import Layer

class Dropout(Layer):
    """
    Dropout layer randomly sets a fraction of the input to 0 during training to provent over fitting
    @param rate: fraction of the input to set to 0 
    """
    def __init__(self, rate=0.5):
        if not 0 <= rate <= 1:
            raise ValueError("Dropout rate must be between 0 and 1")
        self.rate = rate
        self.training = True
    
    def forward(self, input):
        if not self.training:
            return input

        # Create dropout mask (1s where values are kept, 0s where they are dropped)
        self.mask = np.random.binomial(1, 1 - self.rate, size=input.shape)

        # Scale the remaining values to maintain expected sum
        return input * self.mask / (1 - self.rate)
    
    def training(self, mode=True):
        self.training = mode
