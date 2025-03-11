import numpy as np
from layer import Layer

"""
MaxPool layer reduces the size of the image by taking the maximum value in each region
"""
class MaxPool(Layer):
    """
    @param pool_size: size of the pooling region
    @param strides: strides of the pooling
    """
    def __init__(self, pool_size=2, stride=1):
        self.pool_size = pool_size
        self.strides = stride
    
    """
    Forward pass of the MaxPool layer
    @param image: input image
    @return: output of the MaxPool layer
    """
    def forward(self, input):
        self.input = input
        
        # Input dimensions
        input_C, input_H, input_W = input.shape
        
        # Output dimensions after pooling
        output_H = (input_H - self.pool_size) // self.strides + 1
        output_W = (input_W - self.pool_size) // self.strides + 1
        
        # Output array shape (channels, height, width)
        output = np.zeros((input_C, output_H, output_W))

        for c in range(input_C):  # Iterate over channels
            for i in range(0, output_H):
                for j in range(0, output_W):
                    region = self.input[c, i*self.strides:i*self.strides+self.pool_size, j*self.strides:j*self.strides+self.pool_size]
                    output[c, i, j] = np.max(region)

        return output

    