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

        self.input_H, self.input_W, self.input_D = input_shape
        self.input_shape = input_shape

        # The kernel shape is important because it defines the dimensions of the filters applied to the input image, ensuring that:
        self.kernel_H = kernel_size
        self.kernel_W = kernel_size

        self.kernel_shape = (filters, 
                            self.input_D,
                            self.kernel_H, 
                            self.kernel_W)
        
        # returns a sample (or samples) from the “standard normal” distribution.
        # kernels is a 4D array of shape (filters, input depth, kernerl hight, kernel width)
        # select random values from a normal distribution with mean 0 and standard deviation 1 to initialize the kernels
        self.kernels = np.random.randn(filters, self.input_D, self.kernel_H, self.kernel_W)


        # output shape is calculated by subtracting the kernel size from the input size and adding 1
        self.output_shape = (self.input_H - self.kernel_H + 2 * padding // stride + 1,
                             self.input_W - self.kernel_W + 2 * padding // stride + 1,
                            filters)
        
        # biases is a 1D array of shape (filters,)
        self.biases = np.random.randn(self.filters)

    """
    Forward pass of the Conv2D layer
    @param image: input image
    @return: output of the Conv2D layer
    """
    def forward(self, input):
        self.input = input
       
        self.output = self.convolve(input)

        return self.output
    
    def convolve(self, input, stride=1, padding=0):
        H, W, D = input.shape  
        f, _, k_H, k_W = self.kernels.shape  

        # Compute output dimensions
        output_H = (H - k_H) // stride + 1
        output_W = (W - k_W) // stride + 1

        # Initialize output tensor
        output = np.zeros((output_H, output_W, f))

        # Perform convolution operation
        for i in range(f):  # Iterate over filters
            for y in range(output_H):  
                for x in range(output_W):
                    region = input[y:y+k_H, x:x+k_W, :]  # Extract region from input
                    output[y, x, i] = np.sum(region * self.kernels[i, :, :, :])  # Correcting the shape here
                    output[y, x, i] += self.biases[i]  # Add the bias for this filter

        return output



    """
    Backward pass of the Conv2D layer
    @param output_gradient: gradient of the output
    @param learning_rate: learning rate
    @return: gradient of the input
    """
    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels.shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.filters):
            for j in range(self.input_D):
                # Use convolution to compute gradient w.r.t. the kernels
                kernels_gradient[i, j] = self.convolve(self.input[:, :, j], output_gradient[:, :, i])
                
                # Compute the gradient of the input
                input_gradient[:, :, j] += self.deconvolve(output_gradient[:, :, i], self.kernels[i, j])

        # Update the kernels and biases
        self.biases -= learning_rate * np.sum(output_gradient,  axis=(0, 1), keepdims=True)
        self.kernels -= learning_rate * kernels_gradient

        return input_gradient
    
    def deconvolve(self, output_gradient, kernel):
        """
        Deconvolve the output gradient with the kernel
        @param output_gradient: gradient of the output
        @param kernel: kernel
        @return: gradient of the input
        """
        H, W = output_gradient.shape
        kernel_H, kernel_W = kernel.shape

        input_gradient = np.zeros((H + kernel_H - 1, W + kernel_W - 1))

        for y in range(H):
            for x in range(W):
                input_gradient[y:y+kernel_H, x:x+kernel_W] += kernel * output_gradient[y, x]

        return input_gradient

