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

    def train(self, x, y, epochs, learning_rate, loss_function):
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
                loss += loss_function(y, output)
                correct += self.accuracy(output, y)
                output_gradient = self.loss_derivative(output, y)
                self.backward(output_gradient, learning_rate)

                if np.argmax(output) == np.argmax(y):
                    correct += 1
            
            accuracy = correct / len(x)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")

