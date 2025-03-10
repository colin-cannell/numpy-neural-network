import numpy as np
from layer import Layer
from scipy import signal

class Conv2D(Layer):
    def __init__(self, image_shape, kernel_size, depth):
        image_D, image_H, image_W = image_shape
        self.image_shape = image_shape
        self.image_D = image_D
        self.kernel_H = kernel_size[0]
        self.kernel_W = kernel_size[1]
        self.output_shape = (depth, image_H - self.kernel_H + 1, image_W - self.kernel_W + 1)
        self.depth = depth
        self.kernel_shape = (depth, image_D, self.kernel_H, self.kernel_W)
        self.kernels = np.random.randn(*self.kernel_shape)
        self.biases = np.random.randn(depth)
        

    def forward(self, image):
        self.image = image
        self.output = np.copy(self.biases)

        for i in range(self.depth):
            for j in range(self.image_D):
                    self.output[i] += signal.correlate2d(self.image[j], self.kernels[i, j], "valid")

        return self.output        
        
    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels.shape)
        image_gradient = np.zeros(self.image_shape)

        for i in range(self.depth):
            for j in range(self.image_D):
                kernels_gradient[i, j] = signal.correlate2d(self.image[j], output_gradient[i], "valid")
                image_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")
        
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient

        return image_gradient