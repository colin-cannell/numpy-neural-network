import numpy as np
from activations import *
import matplotlib.pyplot as plt

GRADIENT_THRESHOLD = 1000  # You can adjust this value
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
        for layer in self.layers:
            input = layer.forward(input)
            self.inputs.append(input)
            self.activations.append(layer.activation)
                    
        return input

    def backward(self, output_gradient, learning_rate):
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            output_gradient = layer.backward(output_gradient, learning_rate)


    def train(self, x, y, epochs, learning_rate, batch_size, loss, optimizer):
        num_samples = x.shape[0]

        for epoch in range(epochs):
            epochs_loss = 0
            correct_pred = 0
            for i in range(0, num_samples, batch_size):
                x_batch = x[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                # Forward pass
                output = self.forward(x_batch)

                # Calculate loss
                loss_value = loss.forward(output, y_batch)
                epochs_loss += loss_value

                # Backward pass
                output_gradient = loss.backward(output, y_batch)
                self.backward(output_gradient, learning_rate)

                optimizer.update(self.layers)

                predictions = np.argmax(output, axis=1)
                correct_pred += np.sum(predictions == np.argmax(y_batch, axis=1))

            
            epoch_accuracy = correct_pred / num_samples
            self.visualizer.update(epoch + 1, epochs_loss / num_samples, epoch_accuracy)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epochs_loss/num_samples}, Accuracy: {epoch_accuracy}')
        self.visualizer.save()
        self.visualizer.show()

        def predict(self, x):
            output = self.forward(x)
            return np.argmax(output, axis=1)
    
    def conv_output_shape(self, input_shape, kernel_size, filters, stride=1, padding=0):
        """
        Calculate the output shape of a convolutional layer
        @param input_shape: input shape
        @param kernel_size: kernel size
        @param filters: number of filters
        @return: output shape
        """
        B, H, W, C = input_shape
        k_H = kernel_size
        k_W = kernel_size
        H_out = (H - k_H + 2 * padding) // stride + 1
        W_out = (W - k_W + 2 * padding) // stride + 1
        return (B, H_out, W_out, filters)

    def maxpool_output_shape(self, input_shape, pool_size):
        """
        Calculate the output shape of a maxpooling layer
        @param input_shape: input shape
        @param pool_size: pool size
        @return: output shape
        """
        B, H, W, C = input_shape
        H_out = H // pool_size
        W_out = W // pool_size
        return (B, H_out, W_out, C)

    def flatten_output_shape(self, input):
        # flatteen of the input shape
        B = input[0]
        flat = 1
        for i in input[1:]:
            flat *= i
        return (B, flat)
    
    def dense_input_shape(self, batch_size, input_dim):
        """
        Calculate the input shape of a dense layer
        @param input_shape: input shape
        @return: output shape
        """
        return (batch_size, input_dim)

    def dense_output_shape(self, batch_size, output_dim):

        """
        Calculate the output shape of a dense layer
        @param input_shape: input shape
        @return: output shape
        """
        return (batch_size, output_dim)
    

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
