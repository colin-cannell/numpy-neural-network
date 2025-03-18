import numpy as np
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
        self.strides = stride
    
    """
    Forward pass of the MaxPool layer
    @param image: input image
    @return: output of the MaxPool layer
    """
    def forward(self, input):        
        self.input = input
        
        # Input dimensions
        self.batch_size, self.input_H, self.input_W, self.input_C, = input.shape
        
        # Output dimensions after pooling
        self.output_H = self.input_H // self.pool_size + 1
        self.output_W = self.input_W // self.pool_size + 1
        
        # Output array shape (channels, height, width)
        output = np.zeros((self.batch_size, self.output_H, self.output_W, self.input_C))

        self.max_indicies = np.zeros_like(output)

        for y in range(self.output_H):
            for x in range(self.output_W):
                slice = self.input[:, y*self.strides:y*self.strides+self.pool_size, x*self.strides:x*self.strides+self.pool_size, :]
                for c in range(self.input_C):
                    print(slice[:, :, :, c])
                    max_vals = np.max(slice[:, :, :, c], axis=(1, 2))
                    output[:, y, x, c] = max_vals

                    max_indicies = np.argmax(slice[:, :, :, c], axis=(1, 2))
                    self.max_indicies[:, y, x, c] = max_indicies
        
        self.output = output 

        maxpool.maxpool_pooled_feature_maps(self.input, self.output, layer_name="MaxPool Layer")
        maxpool.maxpool_activation_distribution(self.input, self.output, layer_name="MaxPool Layer")
           
        return output

    def backward(self, output_gradient, learning_rate=None):
        batch_size, output_H, output_W, output_C = output_gradient.shape
        input_gradient = np.zeros_like(self.input)


        for y in range(output_H):
            for x in range(output_W):
                for c in range(output_C):
                    # Get the indices of the maximum values
                    max_indices = self.max_indicies[:, y, x, c]
                    input_gradient[:, y*self.strides:y*self.strides+self.pool_size, x*self.strides:x*self.strides+self.pool_size, c] = 0
                    input_gradient[np.arange(batch_size), max_indices, y * self.stride + max_indices % self.pool_size[0], 
                                    x * self.stride + max_indices // self.pool_size[0], c] = output_gradient[:, y, x, c]
        
        # Return the computed input gradients
        return input_gradient

