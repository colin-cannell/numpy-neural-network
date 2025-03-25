import numpy as np
import math
from layer import Layer
from visualize import maxpool

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
        self.stride = stride
    
    """
    Forward pass of the MaxPool layer
    @param image: input image
    @return: output of the MaxPool layer
    """
    def forward(self, input):        
        self.input = input
        
        # Input dimensions
        self.input_H, self.input_W, self.input_C, = input.shape
        
        # Output dimensions after pooling
        self.output_H = math.ceil(self.input_H / self.pool_size)  # Ensure floor division
        self.output_W = math.ceil(self.input_W / self.pool_size)
        
        # Output array shape (channels, height, width)
        output = np.zeros((self.output_H, self.output_W, self.input_C))

        self.max_indices_y = np.zeros((self.output_H, self.output_W, self.input_C))
        self.max_indices_x = np.zeros((self.output_H, self.output_W, self.input_C))

        for y in range(self.output_H):
            for x in range(self.output_W):
                for c in range(self.input_C):
                    y_start = y * self.stride
                    y_end = y_start + self.pool_size
                    x_start = x * self.stride
                    x_end = x_start + self.pool_size

                    region = input[y_start:y_end, x_start:x_end, c]

                    max_val = np.max(region)
                    max_index = np.unravel_index(np.argmax(region), region.shape)

                    output[y, x, c] = max_val

                    self.max_indices_y[y, x, c] = min(y * self.stride + max_index[0], self.input_H - 1)  # Prevent overflow
                    self.max_indices_x[y, x, c] = min(x * self.stride + max_index[1], self.input_W - 1)  # Prevent overflow

                    
        self.output = output 

        # maxpool.maxpool_pooled_feature_maps(self.input, self.output, layer_name="MaxPool Layer")
        # maxpool.maxpool_activation_distribution(self.input, self.output, layer_name="MaxPool Layer")
           
        return output

    def backward(self, output_gradient, learning_rate=None):        
        input_gradient = np.zeros_like(self.input)

        for y in range(self.output_H):
            for x in range(self.output_W):
               for c in range(self.input_C):
                    max_y = int(self.max_indices_y[y, x, c])
                    max_x = int(self.max_indices_x[y, x, c])

                    if 0 <= max_y < self.input_H and 0 <= max_x < self.input_W:
                        input_gradient[max_y, max_x, c] += output_gradient[y, x, c]
        

        # Return the computed input gradients
        return input_gradient

