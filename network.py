import numpy as np
from activations import *

"""
Neural Network class where layers can be added 
forward advancement between layers is handled here.
Backpropogation between layers is handled here.
"""
class NeuralNetwork:
    def __init__(self):
        self.layers = []
    
    def add(self, layer):
        """
        Adds a layer to the network
        @param layer: layer to be added
        """
        self.layers.append(layer)
    
    def forward(self, input):
        """
        Forward transition between layers through the network
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
            # print(f"Backward Output gradient shape: {output_gradient.shape} for layer: {layer.__class__.__name__}")
            output_gradient = layer.backward(output_gradient, learning_rate)
            # print(f"Backward Input gradient shape: {output_gradient.shape} for layer: {layer.__class__.__name__}")
        return output_gradient

    def train(self, x, y, epochs, learning_rate, loss_function=CrossEntropyLoss().forward, loss_derivative=CrossEntropyLoss().backward):
        """
        Trains the model using the given data
        @param x: input data
        @param y: validation data
        @param epochs: number of epochs
        @param learning_rate: learning rate
        """
        loss_history = []  # Store loss values for tracking trends

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Shuffle dataset
            indices = np.arange(len(x))
            np.random.shuffle(indices)
            x, y = x[indices], y[indices]

            loss = 0
            correct = 0
            images_num = 0
            total_images = len(x)

            for i, (xi, yi) in enumerate(zip(x, y), 1):
                images_num += 1
                output = self.forward(xi)
                sample_loss = loss_function(yi, output)
                loss += sample_loss

                # Calculate accuracy
                if np.argmax(output) == np.argmax(yi):
                    correct += 1

                # Backpropagation
                output_gradient = loss_derivative(output, yi)
                self.backward(output_gradient, learning_rate)

                # Print intermediate results every 10 images
                if images_num % 10 == 0 or images_num == total_images:
                    accuracy = correct / images_num
                    print(f"🖼️ Processed {images_num}/{total_images} images - Loss: {loss/images_num:.4f} - Accuracy: {accuracy:.4f}")

            # Store loss history
            epoch_loss = loss / len(x)
            loss_history.append(epoch_loss)
            
            # Print final epoch loss and accuracy
            final_accuracy = correct / len(x)
            print(f"Epoch {epoch+1}/{epochs} - Final Loss: {epoch_loss:.4f} - Final Accuracy: {final_accuracy:.4f}")

        return loss_history  # Return loss history for analysis
    
    def predict(self, x): 
        self.forward(x)

    def conv_output_shape(self, input_shape, kernel_size, filters, stride=1, padding=0):
        """
        Calculate the output shape of a convolutional layer
        @param input_shape: input shape
        @param kernel_size: kernel size
        @param filters: number of filters
        @return: output shape
        """
        H, W, D = input_shape
        k_H = kernel_size
        k_W = kernel_size
        H_out = (H - k_H + 2 * padding) // stride + 1
        W_out = (W - k_W + 2 * padding) // stride + 1
        return (H_out, W_out, filters)

    def maxpool_output_shape(self, input_shape, pool_size):
        """
        Calculate the output shape of a maxpooling layer
        @param input_shape: input shape
        @param pool_size: pool size
        @return: output shape
        """
        H, W, D = input_shape
        H_out = H // pool_size
        W_out = W // pool_size
        return (H_out, W_out, D)

    def flatten_output_shape(self, input_shape):
        """
        Calculate the output shape of a flatten layer
        @param input_shape: input shape
        @return: output shape
        """
        H, W, D = input_shape
        return (H * W * D, )

    def dense_output_shape(self, units):
        """
        Calculate the output shape of a dense layer
        @param input_shape: input shape
        @param units: number of units
        @return: output shape
        """
        return (units, )
    
    def dense_input_shape(self, input_shape):
        """
        Calculate the input shape of a dense layer
        @param input_shape: input shape
        @return: output shape
        """
        return (input_shape, )