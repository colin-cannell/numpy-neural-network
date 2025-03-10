import numpy as np
from layer import Layer

class Flatten(Layer):
    """
    Flatten layer flattens the input image into a 1D array
    @param input: input image
    """
    def forward(self, input):
        return input.flatten()
