import numpy as np
import math
from layer import Layer
from activations import Relu

"""
Conv2D layer performs a 2D convolution on the input image
"""
class Conv2D(Layer):
    def __init__(self, input_shape, kernel_size, filters, stride=1, padding=1, activation=Relu(), visualize=None, id=0):
        self.filters = filters
        self.stride = stride
        self.padding = padding

        self.input_H, self.input_W, self.input_C = input_shape
        self.input_shape = input_shape

         # The kernel shape is important because it defines the dimensions of the filters applied to the input image, ensuring that:
        self.kernel_size = kernel_size

        self.kernel_shape = (self.kernel_size, 
                            self.kernel_size,
                            self.input_C,
                            self.filters)

        h_out = math.ceil((self.input_H - self.kernel_size + 2 * padding) / stride + 1)
        w_out = math.ceil((self.input_W - self.kernel_size + 2 * padding) / stride + 1)

        # output shape is calculated by subtracting the kernel size from the input size and adding 1
        self.feature_map_shape = (h_out, w_out, filters)
        
        # returns a sample (or samples) from the “standard normal” distribution.
        # kernels is a 4D array of shape (filters, input depth, kernerl hight, kernel width)
        # select random values from a normal distribution with mean 0 and standard deviation 1 to initialize the kernels

        # he initialization of kernels for relu activations
        self.kernels = np.random.randn(self.kernel_size, self.kernel_size, self.input_C, self.filters)
        self.bias = np.zeros((self.filters, 1))

        self.activation = activation

        self.visualizer = visualize

        self.id = id
        self.name = f"Conv{id}"

    def pad_input(self, input):
        pad = ((self.padding, self.padding), 
               (self.padding, self.padding), 
               (0, 0))

        return np.pad(input, pad, mode='constant')

    def forward(self, input):
        padded_input = self.pad_input(input)
        self.input = padded_input       
        
        feature_map = np.zeros(self.feature_map_shape)

        for y in range(self.feature_map_shape[0]):
            for x in range(self.feature_map_shape[1]):
                x_start = x * self.stride
                y_start = y * self.stride
                x_end = x_start + self.kernel_size
                y_end = y_start + self.kernel_size

                region = padded_input[y_start:y_end, x_start:x_end, :]

                for f in range(self.feature_map_shape[2]):
                    feature_map[y, x, f] = np.sum(region * self.kernels[:, :, :, f]) + self.bias[f]

        feature_map = self.activation.forward(feature_map)

        if self.visualizer:
            self.visualizer.update_feature_maps(self.id, feature_map)

        return feature_map
    
    def backward(self, output_gradient, learning_rate):
        # Apply activation gradient first
        output_gradient = self.activation.backward(output_gradient)

        output_height, output_width, num_filters = output_gradient.shape

        # Initialize gradients with respect to the filters, input, and biases
        kernel_gradient = np.zeros_like(self.kernels, dtype=np.float64)
        input_gradient = np.zeros_like(self.input, dtype=np.float64)
        bias_gradient = np.zeros_like(self.bias, dtype=np.float64)

        # Compute gradients with respect to the filters and biases
        for y in range(output_height):
            for x in range(output_width):
                # Extract the input slice for each position of the filter
                y_start = y * self.stride
                y_end = y_start + self.kernel_size
                x_start = x * self.stride
                x_end = x_start + self.kernel_size

                region = self.input[y_start:y_end, x_start:x_end, :]
                
                # Calculate the gradient of the loss with respect to the filters
                for f in range(num_filters):
                    grad_scale = output_gradient[y, x, f]
                    
                    # Compute filter gradient
                    for c in range(region.shape[2]):  # Iterate over input channels
                        # Ensure consistent shapes
                        channel_region = region[:, :, c]
                        
                        # Elementwise multiplication with broadcast
                        kernel_gradient[:, :, c, f] += channel_region * grad_scale
                    
                    # Compute input gradient
                    input_gradient[y_start:y_end, x_start:x_end, :] += (
                        self.kernels[:, :, :, f] * grad_scale
                    )
                    
                    # Bias gradient
                    bias_gradient[f] += grad_scale

        self.bias_gradient = bias_gradient
        self.kernel_gradient = kernel_gradient

        # Update parameters
        self.kernels -= learning_rate * kernel_gradient
        self.bias -= learning_rate * bias_gradient.reshape(self.bias.shape)
        
        if self.visualizer:
            self.visualizer.update_kernels_bias(self.id, self.kernels, self.bias)
            self.visualizer.update_gradients(self.name, input_gradient)   

        return input_gradient
                