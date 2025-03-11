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
        # print(f"🔎 Input shape to MaxPool: {self.input.shape}")
        
        # Input dimensions
        input_H, input_W, input_C, = input.shape
        
        # Output dimensions after pooling
        output_H = (input_H - self.pool_size) // self.strides + 1
        output_W = (input_W - self.pool_size) // self.strides + 1
        
        # Output array shape (channels, height, width)
        output = np.zeros((output_H, output_W, input_C))

        for c in range(input_C):  # Iterate over channels
            for y in range(output_H):
                for x in range(output_W):
                    # print(f"🔎 Region: C:{c}, Y:{y}, X:{x}")
                    region = self.input[y*self.strides:y*self.strides+self.pool_size, x*self.strides:x*self.strides+self.pool_size, c]
                    output[y, x, c] = np.max(region)

        return output

    def backward(self, output_gradient):
        # Initialize the gradient with zeros
        input_gradient = np.zeros_like(self.input)

        input_H, input_W, input_C = self.input.shape
        output_H, output_W, _ = output_gradient.shape

        for c in range(input_C):  # Iterate over channels
            for y in range(output_H):
                for x in range(output_W):
                    # Get the region from the forward pass
                    region = self.input[y*self.strides:y*self.strides+self.pool_size, x*self.strides:x*self.strides+self.pool_size, c]
                    
                    # Find the location of the max value in the region
                    max_index = np.unravel_index(np.argmax(region), region.shape)
                    
                    # Set the gradient at the max location to the corresponding gradient from the output
                    input_gradient[y*self.strides + max_index[0], x*self.strides + max_index[1], c] = output_gradient[y, x, c]
        
        return input_gradient
