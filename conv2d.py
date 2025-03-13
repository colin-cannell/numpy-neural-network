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

        self.input_H, self.input_W, self.input_C = input_shape
        self.input_shape = input_shape

        # The kernel shape is important because it defines the dimensions of the filters applied to the input image, ensuring that:
        self.kernel_H = kernel_size
        self.kernel_W = kernel_size

        self.kernel_shape = (filters, 
                            self.input_C,
                            self.kernel_H, 
                            self.kernel_W)
        
        # returns a sample (or samples) from the “standard normal” distribution.
        # kernels is a 4D array of shape (filters, input depth, kernerl hight, kernel width)
        # select random values from a normal distribution with mean 0 and standard deviation 1 to initialize the kernels
        self.kernels = np.random.randn(filters, self.input_C, self.kernel_H, self.kernel_W)


        # output shape is calculated by subtracting the kernel size from the input size and adding 1
        self.output_shape = ((self.input_H - self.kernel_H + 2 * padding) // stride + 1,
                            (self.input_W - self.kernel_W + 2 * padding) // stride + 1,
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

        print(f"Input shape to Conv2D: {self.input.shape}")
       
        self.output = self.convolve(input)

        print(f"Output shape from Conv2D: {self.output.shape}")
        return self.output
    
    def convolve(self, input):
        H, W, D = input.shape  

        # Initialize output tensor
        output = np.zeros(self.output_shape)

        # Perform convolution operation
        for i in range(self.filters):  # Iterate over filters
            for y in range(self.output_shape[0]):  
                for x in range(self.output_shape[1]):                    
                    try:
                        region = input[y:y+self.kernel_H, x:x+self.kernel_W, :]
                        output[y, x, i] = np.sum(region.T * self.kernels[i, :, :, :]) 
                        output[y, x, i] += self.biases[i]  
                    except:
                        # print(f"Region shape: {region.T.shape}")
                        # print(f"Kernel shape: {self.kernels[i, :, :, :].shape}")
                        pass 

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

        print(f"self.input_shape: {self.input_shape}")

        for i in range(self.filters):
            for j in range(self.input_C):
                # Use convolution to compute gradient w.r.t. the kernels
                kernels_gradient[i, j] = self.convolve(self.input, output_gradient[:, :, i])
                
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

