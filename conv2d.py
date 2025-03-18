import numpy as np
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

        self.batch_size, self.input_H, self.input_W, self.input_C = input_shape
        self.input_shape = input_shape

         # The kernel shape is important because it defines the dimensions of the filters applied to the input image, ensuring that:
        self.kernel_size = kernel_size

        self.kernel_shape = (self.kernel_size, 
                            self.kernel_size,
                            self.input_C,
                            self.filters)


        # output shape is calculated by subtracting the kernel size from the input size and adding 1
        self.feature_map_shape = (self.batch_size,
                                 (self.input_H - self.kernel_size + 2 * padding) // stride + 1,
                                 (self.input_W - self.kernel_size + 2 * padding) // stride + 1,
                                 filters)
        
        # returns a sample (or samples) from the “standard normal” distribution.
        # kernels is a 4D array of shape (filters, input depth, kernerl hight, kernel width)
        # select random values from a normal distribution with mean 0 and standard deviation 1 to initialize the kernels

        # he initialization of kernels for relu activations
        self.kernels = np.random.randn(self.kernel_size, self.kernel_size, self.input_C, self.filters)
        print(f"Kernel shape: {self.kernels.shape}")
        self.bias = np.zeros((self.filters, 1))

        self.activation = activation

    def forward(self, input):
        self.input = input       
        
        feature_map = np.zeros(self.feature_map_shape)

        for y in range(self.feature_map_shape[1]):
            for x in range(self.feature_map_shape[2]):
                x_start = x * self.stride
                y_start = y * self.stride
                x_end = x_start + self.kernel_size
                y_end = y_start + self.kernel_size
                region = input[:, y_start:y_end, x_start:x_end, :]
                for f in range(self.feature_map_shape[3]):
                    feature_map[:, y, x, f] = np.sum(region * self.kernels[:, :, :, f]) + self.bias[f]

        feature_map = self.activation.forward(feature_map)

        conv.conv_feature_maps(feature_map, layer_name="Conv Layer")
        conv.conv_kernels(self.kernels, layer_name="Conv Layer")

        return feature_map
    
    def backward(self, output_gradient, learning_rate):
        output_gradient = self.activation.backward(output_gradient)

        batch_size, output_height, output_width, num_filters = output_gradient.shape
        
        # Initialize gradients with respect to the filters, input, and biases
        kernel_gradient = np.zeros_like(self.filters)
        input_gradient = np.zeros_like(self.input)
        bias_gradient = np.zeros_like(self.biases)

        # Compute gradients with respect to the filters and biases
        for i in range(output_height):
            for j in range(output_width):
                # Extract the input slice for each position of the filter
                slice = self.input[:, i * self.stride:i * self.stride + self.kernel_size, j * self.stride:j * self.stride + self.kernel_size, :]

                # Calculate the gradient of the loss with respect to the filters
                for k in range(self.num_filters):
                    kernel_gradient[k] += np.sum(slice * output_gradient[:, i, j, k][:, None, None, None], axis=0)
                    input_gradient[:, i * self.stride:i * self.stride + self.filter_height, j * self.stride:j * self.stride + self.filter_width, :] += self.filters[k] * output_gradient[:, i, j, k][:, None, None, None]


                # Calculate the gradient of the loss with respect to the biases
                bias_gradient += np.sum(output_gradient[:, i, j, :], axis=0)[:, None]

        conv.conv_gradients(self, layer_name="Conv Layer")

        return input_gradient
    
    