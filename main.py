import numpy as np

class Layer:
    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass


class Conv2D(Layer):
    """
    Conv2D layer searches through the image applying filters in order order to find patterns
    @param image: input image
    @param filters: number of filters
    @param kernel_size: size of the kernel
    @param strides: strides of the convolution
    @param padding: padding of the convolution
    @param activation: activation function  
    """
    def __init__(self, image, filters, kernel, strides=1, padding=0, activation=None):
        super().__init__()
        self.image = image
        self.filters = filters
        self.kernel = kernel
        self.strides = strides
        self.padding = padding
        self.activation = activation

    def forward(self):
        image_H, image_W = self.image.shape
        kernel_H, kernel_W = self.kernel.shape

        if self.padding > 0:
            self.image = np.pad(self.image, ((self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        output_H = (image_H - kernel_H) // self.strides + 1
        output_W = (image_W - kernel_W) // self.strides + 1

        output = np.zeros((output_H, output_W))

        for i in range(0, output_H):
            for j in range(0, output_W):
                region = self.image[i*self.strides:i*self.strides+kernel_H, j*self.strides:j*self.strides+kernel_W]
                output[i, j] = np.sum(region * self.kernel)
        
        if self.activation:
            output = self.activation(output)

        return output


class MaxPool(Layer):
    """
    MaxPool layer reduces the size of the image by taking the maximum value in each region
    @param image: input image
    @param pool_size: size of the pooling region
    @param strides: strides of the pooling
    """
    def __init__(self, image, pool_size=2, strides=2):
        self.image = image
        self.pool_size = pool_size
        self.strides = strides
    
    def forward(self):
        image_H, image_W = self.image.shape

        output_H = (image_H - self.pool_size) // self.strides + 1
        output_W = (image_W - self.pool_size) // self.strides + 1

        output = np.zeros((output_H, output_W))

        for i in range(0, output_H):
            for j in range(0, output_W):
                region = self.image[i*self.strides:i*self.strides+self.pool_size, j*self.strides:j*self.strides+self.pool_size]
                output[i, j] = np.max(region)

        return output
    


"""
"""
class Flatten(Layer):
    def __init__(self):
        pass

    def forward(self, input):
        return input.flatten()

"""
Dense layer is a fully connected layer that connects every neuron in the previous layer to every neuron in the next layer
@param input_size: number of input neurons
@param output_size: number of output neurons
"""
class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(output_size)

    def forward(self, input):
        self.input = input
        return np.dot(self.input, self.weights) + self.bias
    
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient      
"""
"""
class Dropout(Layer):
    def __init__(self):
        pass

"""
Relu activation function introduces non-linearity in the model
@ param x: input
"""
def relu(x):
    return np.maximum(0, x)

"""
Sigmoid activation function takes in any real number and returns the output between 0 and 1
@ param x: input
"""
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

"""
Tanh activation function takes in any real number and returns the output between -1 and 1
@ param x: input
"""
def tanh(x):
    return np.tanh(x)

