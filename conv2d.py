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
    def __init__(self, input_shape, kernel_size, depth):
        self.depth = depth

        self.image_shape = input_shape
        self.input_H, self.input_W, self.input_D = input_shape

        self.kernel_H = kernel_size[0]
        self.kernel_W = kernel_size[1]
        self.kernel_shape = (depth, self.input_D, self.kernel_H, self.kernel_W)
        self.kernels = np.random.randn(*self.kernel_shape)

        self.output_shape = (depth, 
                             self.input_H - self.kernel_H + 1, 
                             self.input_W - self.kernel_W + 1)
        
        self.biases = np.random.randn(depth)

    """
    Forward pass of the Conv2D layer
    @param image: input image
    @return: output of the Conv2D layer
    """
    def forward(self, input):
        self.input = input

        # Ensure input is in (channels, height, width) format
        if input.ndim == 3 and input.shape[-1] == 1:  
            self.input_shape = np.transpose(input, (2, 0, 1))  # Convert (H, W, C) -> (C, H, W)
        else:
            self.input_shape = input  # Keep as-is if already (C, H, W)

        # Initialize output correctly
        self.output = np.zeros(self.output_shape)

        for i in range(self.depth):
            for j in range(self.input_D):
                # Ensure correct indexing of 2D image slices
                if self.input_shape[j].ndim != 2:
                    raise ValueError(f"Expected 2D image slice, got {self.input_shape[j].shape}")

                if self.kernels[i, j].ndim != 2:
                    raise ValueError(f"Expected 2D kernel, got {self.kernels[i, j].shape}")

                self.output[i] += signal.correlate2d(self.input_shape[j], self.kernels[i, j], "valid")

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
