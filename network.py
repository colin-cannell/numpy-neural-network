import numpy as np
import math
from activations import *

"""
Neural Network class where layers can be added 
forward advancement between layers is handled here.
Backpropogation between layers is handled here.
"""
class NeuralNetwork:
    def __init__(self, visualize=None):
        self.layers = []
        self.visualizer = visualize


    
    def add(self, layer):
        """
        Adds a layer to the network
        @param layer: layer to be added
        """
        self.layers.append(layer)
    
    def forward(self, input):
        self.activations = []
        self.inputs = [input]

        # print("input shape:", input.shape)  

        for layer in self.layers:
            input = layer.forward(input)  # Apply layer operation
            # print("output shape for layer:", input.shape, "layer:", layer.__class__.__name__)

            self.inputs.append(input)  # Store the output before activation

            # Apply activation function if the layer has one
            if hasattr(layer, "activation") and layer.activation is not None:
                input = layer.activation.forward(input)
                self.activations.append(layer.activation)

        return input

    def backward(self, output_gradient, learning_rate):
        # print("input_gradient shape:", output_gradient.shape)

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            output_gradient = layer.backward(output_gradient, learning_rate)
            # print("layer:", layer.__class__.__name__, "output_gradient shape:", output_gradient.shape)

        
        return output_gradient

    def gradient_clipping(self, gradients, threshold=1.0):
        norm = np.linalg.norm(gradients)
        if norm > threshold:
            gradients = gradients * (threshold / norm)
        return gradients

    def train(self, x, y, epochs, learning_rate, loss, optimizer):
        num_samples = x.shape[0]
        total = len(x)

        epoch_losses = []
        epoch_accuracies = []

        for epoch in range(epochs):
            epochs_loss = 0
            correct_pred = 0
            i = 0
            for xi, yi in zip(x, y):
                i += 1
                # xi = np.expand_dims(xi, axis=-1)

                output = self.forward(xi)

                # Calculate loss
                loss_value = loss.forward(output, yi)
                epochs_loss += loss_value

                output_gradient = loss.backward(output, yi)

                # Backward pass
                output_gradient = self.gradient_clipping(output_gradient)
                self.backward(output_gradient, learning_rate)

                # for layer in self.layers:
                #     if hasattr(layer, "kernels") and hasattr(layer, "bias"):
                #         optimizer.update([layer.kernels, layer.bias], [layer.kernel_gradient, layer.bias_gradient])

                prediction = np.argmax(output)
                true = np.argmax(yi)
                correct_pred += np.sum(prediction == true)

                print(f"\rProcessing {i}/{total}, Prediction : {prediction}, True : {true}, Loss : {loss_value}", end="", flush=True)

            
            epoch_accuracy = correct_pred / num_samples
            epoch_avg_loss = epochs_loss / num_samples

            epoch_losses.append(epoch_avg_loss)
            epoch_accuracies.append(epoch_accuracy)
            
            self.visualizer.update(epoch_losses, epoch_accuracies)
            print(f'\nEpoch {epoch+1}/{epochs}, Loss: {epochs_loss/num_samples}, Accuracy: {epoch_accuracy}')

    def predict(self, x):
        output = self.forward(x)
        return np.argmax(output)
    
    def conv_output_shape(self, input_shape, kernel_size, filters, stride=1, padding=1):
        H, W, C = input_shape
        k_H, k_W = kernel_size, kernel_size  # If square kernel, otherwise pass as tuple
        H_out = math.ceil((H - k_H + 2 * padding) / stride + 1)
        W_out = math.ceil((W - k_W + 2 * padding) / stride + 1)
        return (H_out, W_out, filters)  # Filters represent output channels

    def maxpool_output_shape(self, input_shape, pool_size):
        H, W, C = input_shape
        H_out = math.ceil(H / pool_size)  # Ensure floor division
        W_out = math.ceil(W / pool_size)
        return (H_out, W_out, C)  # Channels remain the same

    def flatten_output_shape(self, input):
        # flatteen of the input shape
        return input[0] * input[1] * input[2]
    
    def dense_output_shape(self, output_dim):
        return output_dim  # Return only the number of output neurons
    

