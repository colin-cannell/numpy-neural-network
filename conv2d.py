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
    def __init__(self, input_shape, kernel_size, depth):
        self.depth = depth

        self.input_H, self.input_W, self.input_D = input_shape
        self.input_shape = input_shape

        self.kerner_size = kernel_size
        self.kernel_H = kernel_size
        self.kernel_W = kernel_size

        self.kernel_shape = (depth, 
                            self.input_D,
                            self.kernel_H, 
                            self.kernel_W)
        # print(f"🔎 Kernel shape: {self.kernel_shape}")
        self.kernels = np.random.randn(*self.kernel_shape)

        self.output_shape = (self.input_H - self.kernel_H + 1, 
                            self.input_W - self.kernel_W + 1,
                            depth)
        
        self.biases = np.random.randn(*self.output_shape)

    """
    Forward pass of the Conv2D layer
    @param image: input image
    @return: output of the Conv2D layer
    """
    def forward(self, input):
        self.input = input
        # print(f"🔎 Input shape to Conv2D: {self.input.shape}")
        # print(f"🎯 Kernel shape: {self.kernels.shape}")
        # print(f"📤 Expected output shape: {self.output_shape}")

        # initialize the output
        self.output = np.copy(self.biases)

        # Perform convolution with numpy operations
        for i in range(self.depth):
            for j in range(self.input_D):
                #  self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
                # Slide the kernel over the input and perform element-wise multiplication
                for y in range(self.output_shape[1]):  # Height
                    for x in range(self.output_shape[2]):  # Width
                        region = self.input[y:y+self.kernel_H, x:x+self.kernel_W, j]
                        # print("y", y, "x", x, "i", i, "j", j)
                        if region.shape == self.kernels[i, j].shape:
                            self.output[y, x, i] += np.sum(region * self.kernels[i, j])

        return self.output

    """
    Backward pass of the Conv2D layer
    @param output_gradient: gradient of the output
    @param learning_rate: learning rate
    @return: gradient of the input
    """
    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels.shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_D):
                for y in range(self.output_shape[0]):  # Height
                    for x in range(self.output_shape[1]):  # Width
                        region = self.input[y:y+self.kernel_H, x:x+self.kernel_W, j]
                        if region.shape == self.kernels[i, j].shape:
                            kernels_gradient[i, j] += np.sum(region * output_gradient[y, x, i])
                            # Compute the gradient for the input
                            input_gradient[y:y+self.kernel_H, x:x+self.kernel_W, j] += self.kernels[i, j] * output_gradient[y, x, i]

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * np.sum(output_gradient, axis=(0, 1), keepdims=True)

        return input_gradient

