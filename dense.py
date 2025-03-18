import numpy as np
from layer import Layer
from visualize import dense

class Dense(Layer):
    """
    Dense layer is a fully connected layer that connects every neuron in the previous layer to every neuron in the next layer
    @param input_size: number of input neurons
    @param output_size: number of output neurons
    """
    def __init__(self, input_size, output_size, activation):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        stddev = np.sqrt(2.0 / self.input_size)
        self.weights = np.random.randn(input_size, output_size) * stddev
        self.bias = np.zeros(output_size)
        
        self.input = None
        self.output = None


    def forward(self, input):
        self.input = input

        z = np.dot(input, self.weights) + self.bias
        self.output = self.activation.forward(z)

        self.dense_neuron_activations(self.output, "Dense Layer Forward Pass")
        
        return self.output
        
    def backward(self, output_gradient, learning_rate):
        activation_gradient = self.activation.backward(self.output)
        dL_dz = output_gradient * activation_gradient

        dL_dw = np.dot(self.input.T, dL_dz)
        dL_db = np.sum(dL_dz, axis=0)

        self.weights -= learning_rate * dL_dw
        self.bias -= learning_rate * dL_db

        self.dense_weight_distribution(self.weights, "Dense Layer Weight Distribution")
        self.dense_gradient_flow(self, "Dense Layer Gradient Flow")

        input_gradient = np.dot(dL_dz, self.weights.T)
        return input_gradient

