import numpy as np
from scipy import signal
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
    def __init__(self, image_shape, kernel_size, depth):
        image_D, image_H, image_W = image_shape
        self.image_shape = image_shape
        self.image_D = image_D
        self.image_H = image_H
        self.image_W = image_W
        self.kernel_H = kernel_size[0]
        self.kernel_W = kernel_size[1]
        self.output_shape = (depth, image_H - self.kernel_H + 1, image_W - self.kernel_W + 1)
        self.depth = depth
        self.kernel_shape = (depth, image_D, self.kernel_H, self.kernel_W)
        self.kernels = np.random.randn(*self.kernel_shape)
        self.biases = np.random.randn(depth)

    """
    Forward pass of the Conv2D layer
    @param image: input image
    @return: output of the Conv2D layer
    """
    def forward(self, image):
        self.input = image
        self.output = np.copy(self.biases)

        for i in range(self.depth):
            for j in range(self.image_D):
                self.output[i] += signal.correlate2d(self.image_shape[j], self.kernels[i, j], "valid")

        return self.output

    """
    Backward pass of the Conv2D layer
    @param output_gradient: gradient of the output
    @param learning_rate: learning rate
    @return: gradient of the input
    """
    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels.shape)
        image_gradient = np.zeros(self.image_shape)

        for i in range(self.depth):
            for j in range(self.image_D):
                kernels_gradient[i, j] = signal.correlate2d(self.image[j], output_gradient[i], "valid")
                image_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")
        
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * np.sum(output_gradient, axis=(1, 2))

        return image_gradient
