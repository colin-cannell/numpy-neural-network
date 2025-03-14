import numpy as np
from layer import Layer

"""
Conv2D layer performs a 2D convolution on the input image
"""
class Conv2D(Layer):
    """
    @param image_shape: shape of the input image
    @param kernel_size: size of the kernel
    @param depth: number of filters
    """
    def __init__(self, input_shape, kernel_size, filters, stride=1, padding=0):
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


        # output shape is calculated by subtracting the kernel size from the input size and adding 1
        self.output_shape = ((self.input_H - self.kernel_size + 2 * padding) // stride + 1,
                            (self.input_W - self.kernel_size + 2 * padding) // stride + 1,
                            filters)
        
        # returns a sample (or samples) from the “standard normal” distribution.
        # kernels is a 4D array of shape (filters, input depth, kernerl hight, kernel width)
        # select random values from a normal distribution with mean 0 and standard deviation 1 to initialize the kernels
        self.kernels = np.random.randn(*self.kernel_shape) * 0.01
        self.biases = np.zeros((1, 1, self.filters))

    """
    Forward pass of the Conv2D layer
    @param image: input image
    @return: output of the Conv2D layer
    """
    def forward(self, input):
        self.input = input       
        self.output = np.zeros(self.output_shape)

        for i in range(self.filters):
            for j in range(self.input_C):
                self.output[:, :, i] = self.correlate(self.input[:, :, j], self.kernels[:, :, j, i]) 

        self.output += self.biases

        return self.output
    
    def correlate(self, input, kernel):
        output = np.zeros(self.output_shape[:2])

        for x in range(self.input_H):
            for y in range(self.input_W):
                    region = input[x * self.stride : x * self.stride + kernel.shape[0], 
                                y * self.stride : y * self.stride + kernel.shape[1]]
                    if region.shape == kernel.shape:
                        output[x // self.stride, y // self.stride] = np.sum(region * kernel)

                    
        return output

    """
    Backward pass of the Conv2D layer
    @param output_gradient: gradient of the output
    @param learning_rate: learning rate
    @return: gradient of the input
    """
    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels.shape)
        input_gradient = np.zeros(output_gradient.shape)
        # print("kernel shape", self.kernels.shape)
        # print("input shape", self.input.shape)
        # print("output shape", output_gradient.shape)  

        for i in range(self.filters):
            for j in range(self.input_C):
                x = self.correlate(self.input[:, :, j], output_gradient[:, :, i])
                input_gradient[:, :, j] = self.correlate(output_gradient[:, :, i], self.kernels[:, :, j, i])


        # Update the kernels and biases
        self.biases -= learning_rate * np.sum(output_gradient, axis=(0, 1), keepdims=True)
        
        self.kernels -= learning_rate * kernels_gradient

        return input_gradient
    
    