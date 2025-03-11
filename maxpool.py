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
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.strides = stride
    
    """
    Forward pass of the MaxPool layer
    @param image: input image
    @return: output of the MaxPool layer
    """
    def forward(self, input):
        self.input = input
        print(f"🔎 Input shape to MaxPool: {self.input.shape}")
        
        # Input dimensions
        input_C, input_H, input_W = input.shape
        
        # Output dimensions after pooling
        output_H = (input_H - self.pool_size) // self.strides + 1
        output_W = (input_W - self.pool_size) // self.strides + 1
        
        # Output array shape (channels, height, width)
        output = np.zeros((input_C, output_H, output_W))

        for c in range(input_C):  # Iterate over channels
            for y in range(output_H):
                for x in range(output_W):
                    # print(f"🔎 Region: C:{c}, Y:{y}, X:{x}")
                    region = self.input[c, y*self.strides:y*self.strides+self.pool_size, x*self.strides:x*self.strides+self.pool_size]
                    output[c, y, x] = np.max(region)
                    
        return output

    