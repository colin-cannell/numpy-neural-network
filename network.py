import numpy as np
import math
from activations import *
import matplotlib.pyplot as plt

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
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            output_gradient = layer.backward(output_gradient, learning_rate)

    def gradient_clipping(self, gradients, threshold=1.0):
        norm = np.linalg.norm(gradients)
        if norm > threshold:
            gradients = gradients * (threshold / norm)
        return gradients


    def train(self, x, y, epochs, learning_rate, loss, optimizer):
        num_samples = x.shape[0]

        for epoch in range(epochs):
            epochs_loss = 0
            correct_pred = 0
            for xi, yi in zip(x, y):
                xi = np.expand_dims(xi, axis=-1)

                output = self.forward(xi)

                # Calculate loss
                loss_value = loss.forward(output, yi)
                epochs_loss += loss_value

                # Backward pass
                output_gradient = loss.backward(output, yi)
                output_gradient = self.gradient_clipping(output_gradient)
                self.backward(output_gradient, learning_rate)

                # for layer in self.layers:
                #     if hasattr(layer, "kernels") and hasattr(layer, "bias"):
                #         optimizer.update([layer.kernels, layer.bias], [layer.kernel_gradient, layer.bias_gradient])

                prediction = np.argmax(output)
                correct_pred += np.sum(prediction == np.argmax(yi))

            
            epoch_accuracy = correct_pred / num_samples
            self.visualizer.update(epoch + 1, epochs_loss / num_samples, epoch_accuracy)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epochs_loss/num_samples}, Accuracy: {epoch_accuracy}')
        self.visualizer.save()
        self.visualizer.show()

    def predict(self, x):
        output = self.forward(x)
        return np.argmax(output)
    
    def conv_output_shape(self, input_shape, kernel_size, filters, stride=1, padding=0):
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
    

class TrainingVisualizer:
    def __init__(self):
        self.epochs = []
        self.losses = []
        self.accuracies = []
        
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 5))
    
    def update(self, epoch, loss, accuracy):
        """Update the visualization with new data."""
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        
        self.ax[0].cla()
        self.ax[0].plot(self.epochs, self.losses, 'r-', label='Loss')
        self.ax[0].set_title('Training Loss')
        self.ax[0].set_xlabel('Epoch')
        self.ax[0].set_ylabel('Loss')
        self.ax[0].legend()
        
        self.ax[1].cla()
        self.ax[1].plot(self.epochs, self.accuracies, 'b-', label='Accuracy')
        self.ax[1].set_title('Training Accuracy')
        self.ax[1].set_xlabel('Epoch')
        self.ax[1].set_ylabel('Accuracy')
        self.ax[1].legend()
        
        plt.pause(0.1)  # Pause to update the figure
    
    def save(self, filename='training_progress.png'):
        """Save the plot to a file."""
        self.fig.savefig(filename)
        print(f"Plot saved as {filename}")
    
    def show(self):
        """Keep the plot open after training ends."""
        plt.ioff()
        plt.show()
