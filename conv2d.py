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

        self.kernel_shape = (filters, 
                            self.input_C,
                            self.kernel_size, 
                            self.kernel_size)
        
        # returns a sample (or samples) from the “standard normal” distribution.
        # kernels is a 4D array of shape (filters, input depth, kernerl hight, kernel width)
        # select random values from a normal distribution with mean 0 and standard deviation 1 to initialize the kernels
        self.kernels = np.random.randn(self.filters, self.input_C, self.kernel_size, self.kernel_size) * np.sqrt(2.0 / self.input_C)
        
        # biases is a 1D array of shape (filters,)
        self.biases = np.zeros((1, self.filters))


        # output shape is calculated by subtracting the kernel size from the input size and adding 1
        self.output_shape = ((self.input_H - self.kernel_size + 2 * padding) // stride + 1,
                            (self.input_W - self.kernel_size + 2 * padding) // stride + 1,
                            filters)

    
    """
    Forward pass of the Conv2D layer
    @param image: input image
    @return: output of the Conv2D layer
    """
    def forward(self, input):
        self.input = input       
        self.output = self.convolve(input)
        return self.output
    
    def convolve(self, input, output_gradient=None):
        H, W, D = input.shape  

        # Initialize output tensor
        output = np.zeros(self.output_shape)

        # Perform convolution operation
        for f in range(self.filters):  # Iterate over filters
            for y in range(self.output_shape[0] * self.stride, self.stride):  
                for x in range(self.output_shape[1] * self.stride, self.stride):                    
                    try:
                        region = input[y:y+self.kernel_size, x:x+self.kernel_size, :]

                        if output_gradient is None:
                            output[y, x, f] = np.sum(region.T * self.kernels[f, :, :, :]) 
                        else:
                            output[y, x, f] = np.sum(region.T * output_gradient[y, x, f])  # Use output_gradient instead of the kernel here

                        output[y, x, f] += self.biases[f]  
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


        # Iterate over each filter to compute its gradient
        for f in range(self.kernels.shape[0]):  # Loop over each filter
            for c in range(self.kernels.shape[1]):  # Loop over input channels
                # Accumulate kernel gradients across all positions in the output
                for y in range(output_gradient.shape[0]):
                    for x in range(output_gradient.shape[1]):
                        # Extract the region of the input that corresponds to the current position
                        region = self.input[y:y+self.kernel_size, x:x+self.kernel_size, c]
                        kernels_gradient[f, c] += region * output_gradient[y, x, f]

        input_gradient = self.deconvolve(output_gradient)


        # Update the kernels and biases
        self.biases -= learning_rate * np.sum(output_gradient,  axis=(0, 1))
        self.kernels -= learning_rate * kernels_gradient

        return input_gradient
    
    def deconvolve(self, output_gradient):
        """
        Deconvolve the output gradient with the kernel
        @param output_gradient: gradient of the output
        @param kernel: kernel
        @return: gradient of the input
        """
        H, W, C =  output_gradient.shape

        input_gradient = np.zeros(self.input_shape)
        for f in range(self.filters):
            for y in range(H):
                for x in range(W):
                    input_gradient[y:y+self.kernel_size, x:x+self.kernel_size, :] += self.kernels[f].T * output_gradient[y, x, f]


        return input_gradient

