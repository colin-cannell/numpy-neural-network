import numpy as np
import math
from layer import Layer
from activations import Relu
from visualize import conv

"""
Conv2D layer performs a 2D convolution on the input image
"""
class Conv2D(Layer):
    def __init__(self, input_shape, kernel_size, filters, stride=1, padding=0, activation=Relu()):
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

    def forward(self, input):
        self.input = input       
        
        feature_map = np.zeros(self.feature_map_shape)

        for y in range(self.feature_map_shape[0]):
            for x in range(self.feature_map_shape[1]):
                x_start = x * self.stride
                y_start = y * self.stride
                x_end = x_start + self.kernel_size
                y_end = y_start + self.kernel_size
                region = input[y_start:y_end, x_start:x_end, :]
                for f in range(self.feature_map_shape[2]):
                    feature_map[y, x, f] = np.sum(region * self.kernels[:, :, :, f]) + self.bias[f]

        feature_map = self.activation.forward(feature_map)

        # conv.conv_feature_maps(feature_map, layer_name="Conv Layer")
        # conv.conv_kernels(self.kernels, layer_name="Conv Layer")

        return feature_map
    
    def backward(self, output_gradient, learning_rate):
        output_gradient = self.activation.backward(output_gradient)

        output_height, output_width, num_filters = output_gradient.shape

        # Initialize gradients with respect to the filters, input, and biases
        kernel_gradient = np.zeros_like(self.kernels, dtype=np.float64)
        input_gradient = np.zeros_like(self.input, dtype=np.float64)
        bias_gradient = np.zeros_like(self.bias, dtype=np.float64)

        print("kernel gradient", kernel_gradient.shape)
        print("input gradient", input_gradient.shape)
        print("bias gradient", bias_gradient.shape)
        print("output gradient", output_gradient.shape)

        # Compute gradients with respect to the filters and biases
        for y in range(output_height):
            for x in range(output_width):
                # Extract the input slice for each position of the filter
                region = self.input[y * self.stride:y * self.stride + self.kernel_size, 
                                   x * self.stride:x * self.stride + self.kernel_size, :]
                

                # Calculate the gradient of the loss with respect to the filters
                for f in range(num_filters):
                    result = region * output_gradient[y, x, f]
                    try:
                        kernel_gradient[:, :, :, f] += result
                    except ValueError as e:
                        print("result", result.shape)

                    input_gradient[y * self.stride:y * self.stride + self.kernel_size, 
                                   x * self.stride:x * self.stride + self.kernel_size, :] += self.kernels[:, :, :, f] * output_gradient[y, x, f]
                    bias_gradient[f] += output_gradient[y, x, f] 

            

        self.bias_gradient = bias_gradient
        self.kernel_gradient = kernel_gradient

        self.kernels -= learning_rate * kernel_gradient
        self.bias -= learning_rate * bias_gradient.reshape(self.bias.shape)

        # conv.conv_gradients(self, layer_name="Conv Layer")

        return input_gradient
    
    