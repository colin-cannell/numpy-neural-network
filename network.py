import numpy as np
from activation import Activation
from activation import cross_entropy_loss, cross_entropy_loss_derivative

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
            output = layer.forward(output)
        return output

    def backward(self, output_gradient, learning_rate):
        """
        Backward pass through the network
        @param output_gradient: gradient of the loss with respect to the output
        @param learning_rate: learning rate
        """
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient, learning_rate)
        return output_gradient

    def train(self, x, y, epochs, learning_rate, loss_function=cross_entropy_loss, loss_derivative=cross_entropy_loss_derivative):
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
            for xi, yi in zip(x, y):
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
