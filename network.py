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
        """
        Forward transition between layers through the network
        @param input: input data
        """
        #with open('forward.txt', 'w') as f:
        output = input
        for layer in self.layers:
            #f.write(f"Forward input to layer  {layer.__class__.__name__} : {output}\n")
            # print(f"Forward input to layer  {layer.__class__.__name__} : {output.shape}")
            output = layer.forward(output)
            # print(f"Forward output from layer  {layer.__class__.__name__} : {output.shape}")
            #f.write(f"Forward output from layer  {layer.__class__.__name__} : {output}\n")
        #f.close()
                    
        return output

    def backward(self, output_gradient, learning_rate):
        """
        Backward pass through the network
        @param output_gradient: gradient of the loss with respect to the output
        @param learning_rate: learning rate
        """
        # with open('backward.txt', 'w') as f:
        for layer in reversed(self.layers):
            #f.write(f"Backward input to layer  {layer.__class__.__name__} : {output_gradient}\n")
            # print(f"Backward input to layer  {layer.__class__.__name__} : {output_gradient.shape}")
            output_gradient = layer.backward(output_gradient, learning_rate)
            # print(f"Backward output from layer  {layer.__class__.__name__} : {output_gradient.shape}")
            #f.write(f"Backward output from layer  {layer.__class__.__name__} : {output_gradient}\n")
        #f.close()
        
        return output_gradient

    def train(self, x, y, epochs, learning_rate, loss_function=CrossEntropyLoss().forward, loss_derivative=CrossEntropyLoss().backward, batch_size=10):
        """
        Trains the model using the given data
        @param x: input data
        @param y: validation data
        @param epochs: number of epochs
        @param learning_rate: learning rate
        """
        loss_history = []  # Store loss values for tracking trends
        total_images = len(x)
        total_correct = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            epoch_loss = 0
            images_num = 0
            correct = 0
            history = []


            for i, (xi, yi) in enumerate(zip(x, y), 1):

                output = self.forward(xi)

                if np.max(output) > GRADIENT_THRESHOLD:
                    print("Gradient explosion detected!")
                    break

                sample_loss = loss_function(yi, output)
                epoch_loss += sample_loss

                predict = f"P.{int(np.argmax(output))}"
                label = f"L{int(np.argmax(yi))}"

                history.append((predict, label))
                # Calculate accuracy
                if np.argmax(output) == np.argmax(yi):
                    correct += 1

                images_num += 1
                # Backpropagation
                output_gradient = loss_derivative(yi, output)
                self.backward(output_gradient, learning_rate)

                # Print intermediate results every 10 images
                if images_num % batch_size == 0 or images_num == total_images:
                    accuracy = correct / images_num
                    print(f"🖼️ Processed {images_num}/{total_images} images - Loss: {epoch_loss/images_num:.4f} - Batch Accuracy: {accuracy:.4f}")
                    print(f"Batch history: {history}")
                    total_correct += correct
                    correct = 0  # Reset correct count for the next batch
                    history = []


            # Store loss history
            epoch_loss /= total_images
            loss_history.append(epoch_loss)
            
            # Print final epoch loss and accuracy
            final_accuracy = total_correct / total_images
            print(f"Epoch {epoch+1}/{epochs} - Final Loss: {epoch_loss:.4f} - Final Accuracy: {final_accuracy:.4f}")

        return loss_history  # Return loss history for analysis
    
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