import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.layers = []
    
    def add(self, layer):
        """
        Adds a layer to the network
        @param layer: layer to add
        """
        self.layers.append(layer)
    
    def forward(self, input):
        """
        Forward pass through the network
        @param input: input data
        """
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, output_gradient, learning_rate):
        """
        Backward pass through the network
        @param output_gradient: gradient of the loss with respect to the output
        @param learning_rate: learning rate
        """
    
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient, learning_rate)
        return output_gradient

    def train(self, x, y, epochs, learning_rate):
        """
        Trains the model using the given data
        @param x: input data
        @param y: target data
        @param epochs: number of epochs
        @param learning_rate: learning rate
        """
        for epoch in range(epochs):
            loss = 0
            correct = 0
            for x, y in zip(x, y):
                output = self.forward(x)
                loss += self.loss(output, y)
                correct += self.accuracy(output, y)
                output_gradient = self.loss_derivative(output, y)
                self.backward(output_gradient, learning_rate)

                if np.argmax(output) == np.argmax(y):
                    correct += 1
            
            accuracy = correct / len(x)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")

    
    def loss(self, output, target):
        """
        Calculates the loss between the output and the target using mean squared error
        @param output: output of the model
        @param target: target data
        """
        return np.mean((output - target) ** 2)
    
    def loss_derivative(self, output, target):
        """
        Calculates the derivative of the loss function
        @param output: output of the model
        @param target: target data
        """
        return 2 * (output - target) / output.size


class Layer:
    """
    Base class for all layers
    """
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
    

class Flatten(Layer):
    """
    Flatten layer flattens the input image into a 1D array
    @param input: input image
    """
    def forward(self, input):
        return input.flatten()

class Dense(Layer):
    """
    Dense layer is a fully connected layer that connects every neuron in the previous layer to every neuron in the next layer
    @param input_size: number of input neurons
    @param output_size: number of output neurons
    """
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

class Dropout(Layer):
    """
    Dropout layer randomly sets a fraction of the input to 0 during training to provent over fitting
    @param rate: fraction of the input to set to 0 
    """
    def __init__(self, rate=0.5):
        if not 0 <= rate <= 1:
            raise ValueError("Dropout rate must be between 0 and 1")
        self.rate = rate
        self.training = True
    
    def forward(self, input):
        if not self.training:
            return input

        # Create dropout mask (1s where values are kept, 0s where they are dropped)
        self.mask = np.random.binomial(1, 1 - self.rate, size=input.shape)

        # Scale the remaining values to maintain expected sum
        return input * self.mask / (1 - self.rate)
    
    def training(self, mode=True):
        self.training = mode

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

