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
        self.input_H, self.input_W, self.input_C, = input.shape
        
        # Output dimensions after pooling
        output_H = self.input_H // self.pool_size
        output_W = self.input_W // self.pool_size
        
        # Output array shape (channels, height, width)
        output = np.zeros((output_H, output_W, self.input_C))

        for c in range(self.input_C):  # Iterate over channels
            for y in range(output_H):
                for x in range(output_W):
                    # print(f"🔎 Region: C:{c}, Y:{y}, X:{x}")
                    region = self.input[y*self.strides:y*self.strides+self.pool_size, x*self.strides:x*self.strides+self.pool_size, c]
                    output[y, x, c] = np.max(region)

        return output

    def backward(self, output_gradient, learning_rate=None):
        # Initialize the gradient with zeros
        input_gradient = np.zeros_like(self.input)


        output_H, output_W, output_C = output_gradient.shape

        for c in range(self.input_C):  # Iterate over channels
            for y in range(output_H):
                for x in range(output_W):
                    # Ensure we don't go out of bounds when slicing
                    y_start = y * self.strides
                    x_start = x * self.strides
                    y_end = min(y_start + self.pool_size, self.input_H)
                    x_end = min(x_start + self.pool_size, self.input_W)

                    # Get the region from the forward pass
                    region = self.input[y_start:y_end, x_start:x_end, c]

                    if region.size > 0:
                        max_index = np.unravel_index(np.argmax(region), region.shape)
                        h_index = y_start + max_index[0]
                        w_index = x_start + max_index[1]

                        if h_index < self.input_H and w_index < self.input_W:
                            # Ensure correct accumulation across all channels
                            input_gradient[h_index, w_index, c] += output_gradient[y, x, c]

        # Return the computed input gradients
        return input_gradient

