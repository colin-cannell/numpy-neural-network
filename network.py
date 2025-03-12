import numpy as np
from activations import *

"""
Neural network class that defines the architecture of the network
"""
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
        output = input
        for layer in self.layers:
            print(f"Forward Input shape: {output.shape} for layer: {layer.__class__.__name__}")
            output = layer.forward(output)
            print(f"Forward Output shape: {output.shape} for layer: {layer.__class__.__name__}")
        return output

    def backward(self, output_gradient, learning_rate):
        """
        Backward pass through the network
        @param output_gradient: gradient of the loss with respect to the output
        @param learning_rate: learning rate
        """
        for layer in reversed(self.layers):
            print(f"Backward Output gradient shape: {output_gradient.shape} for layer: {layer.__class__.__name__}")
            output_gradient = layer.backward(output_gradient, learning_rate)
            print(f"Backward Input gradient shape: {output_gradient.shape} for layer: {layer.__class__.__name__}")
        return output_gradient

    def train(self, x, y, epochs, learning_rate, loss_function=CrossEntropyLoss().forward, loss_derivative=CrossEntropyLoss().backward):
        """
        Trains the model using the given data
        @param x: input data
        @param y: validation data
        @param epochs: number of epochs
        @param learning_rate: learning rate
        """
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            loss = 0
            correct = 0
            for xi, yi in zip(x, y):
                print(f"Training on sample: {xi.shape}")
                output = self.forward(xi)
                loss += loss_function(yi, output)
                
                # Calculate accuracy: increment correct if predicted class matches target class
                if np.argmax(output) == np.argmax(yi):
                    correct += 1
                
                # Calculate output gradient
                output_gradient = loss_derivative(output, yi)
                
                # Perform backward pass
                self.backward(output_gradient, learning_rate)

            accuracy = correct / len(x)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")
