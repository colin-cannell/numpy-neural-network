import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import seaborn as sns
import time
import threading

class conv:
    # feature maps
    def conv_feature_maps(feature_maps, layer_name="Conv Layer"):
        num_filters = feature_maps.shape[-1]
        fig, axes = plt.subplots(1, num_filters, figsize=(20, 5))

        for i in range(num_filters):
            ax = axes[i] if num_filters > 1 else axes
            ax.imshow(feature_maps[:, :, i], cmap="viridis")
            ax.set_title(f"{i+1}", fontsize=10)
            ax.axis("off")

        plt.suptitle(f"Feature Maps from {layer_name}", fontsize=14, y=1.05)  # Move title up
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.subplots_adjust(top=0.85)  # Give space for the title
        plt.show()


    # kernels
    def conv_kernels(kernels, layer_name="Conv Layer"):
        num_filters = kernels.shape[-1]
        fig, axes = plt.subplots(1, num_filters, figsize=(20, 5))

        for i in range(num_filters):
            ax = axes[i] if num_filters > 1 else axes
            ax.imshow(kernels[:, :, 0, i], cmap="gray")  # Assuming 1 channel for visualization
            ax.set_title(f"{i+1}")
            ax.axis("off")

        plt.suptitle(f"Kernels from {layer_name}")
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()
    
    # gradient flow
    def conv_gradients(layer, layer_name="Conv Layer"):
        grad_values = layer.kernels.flatten()
        
        plt.figure(figsize=(8, 6))
        plt.hist(grad_values, bins=50, color='blue', alpha=0.7)
        plt.xlabel("Gradient Value")
        plt.ylabel("Frequency")
        plt.title(f"Gradient Distribution - {layer_name}")
        plt.grid(True)
        plt.show()

class maxpool:
    # pooled feature maps
    def maxpool_pooled_feature_maps(before_pooling, after_pooling, layer_name="MaxPool Layer"):
        num_filters = before_pooling.shape[-1]
        fig, axes = plt.subplots(2, num_filters, figsize=(20, 10))

        for i in range(num_filters):
            # Before Pooling
            ax1 = axes[0, i] if num_filters > 1 else axes[0]
            ax1.imshow(before_pooling[:, :, i], cmap="viridis")
            ax1.set_title(f"{i+1}")
            ax1.axis("off")

            # After Pooling
            ax2 = axes[1, i] if num_filters > 1 else axes[1]
            ax2.imshow(after_pooling[:, :, i], cmap="viridis")
            ax2.set_title(f"{i+1}")
            ax2.axis("off")

        plt.suptitle(f"Pooled Feature Maps - {layer_name}")
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()

    # activation distribution before and after pooling
    def maxpool_activation_distribution(before_pooling, after_pooling, layer_name="MaxPool Layer"):
        before_activations = np.array(before_pooling).flatten()
        after_activations = np.array(after_pooling).flatten()

        plt.figure(figsize=(10, 5))
        plt.hist(before_activations, bins=50, alpha=0.5, label="Before Pooling", color='blue')
        plt.hist(after_activations, bins=50, alpha=0.5, label="After Pooling", color='red')
        plt.xlabel("Activation Value")
        plt.ylabel("Frequency")
        plt.title(f"Activation Distribution Before & After Pooling - {layer_name}")
        plt.legend()
        plt.show()

class flatten:
    # flattened activation distribution
    def flattened_distribution(before_flatten, after_flatten, layer_name="Flatten Layer"):
        before_activations = np.array(before_flatten).flatten()
        after_activations = np.array(after_flatten).flatten()

        plt.figure(figsize=(10, 5))
        plt.hist(before_activations, bins=50, alpha=0.5, label="Before Flattening", color='blue')
        plt.hist(after_activations, bins=50, alpha=0.5, label="After Flattening", color='red')
        plt.xlabel("Activation Value")
        plt.ylabel("Frequency")
        plt.title(f"Activation Distribution Before & After Flattening - {layer_name}")
        plt.legend()
        plt.show()

class dense:
    # neuron activations
    def dense_neuron_activations(activations, layer_name="Dense Layer"):
        plt.figure(figsize=(10, 5))
        sns.heatmap(activations.reshape(1, -1), cmap="viridis", xticklabels=False, yticklabels=False)
        plt.xlabel("Neurons")
        plt.ylabel("Activation")
        plt.title(f"Neuron Activations - {layer_name}")
        plt.show()

    # weight distribution
    def dense_weight_distribution(weights, layer_name="Dense Layer"):
        plt.figure(figsize=(8, 5))
        plt.hist(weights.flatten(), bins=50, color='blue', alpha=0.7)
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")
        plt.title(f"Weight Distribution - {layer_name}")
        plt.grid(True)
        plt.show()

    # gradient flow
    def dense_gradient_flow(layer, layer_name="Dense Layer"):
        grad_values = layer.weights.flatten()
        
        plt.figure(figsize=(8, 6))
        plt.hist(grad_values, bins=50, color='red', alpha=0.7)
        plt.xlabel("Gradient Value")
        plt.ylabel("Frequency")
        plt.title(f"Gradient Distribution - {layer_name}")
        plt.grid(True)
        plt.show()

class dropout:
    # neuron activations before and after dropout
    def dropout_effect(activations_before, activations_after, layer_name="Dropout Layer"):
        plt.figure(figsize=(10, 5))
        plt.hist(activations_before.flatten(), bins=50, alpha=0.5, label="Before Dropout", color='blue')
        plt.hist(activations_after.flatten(), bins=50, alpha=0.5, label="After Dropout", color='red')
        plt.xlabel("Activation Value")
        plt.ylabel("Frequency")
        plt.title(f"Activation Distribution Before & After Dropout - {layer_name}")
        plt.legend()
        plt.show()

class NetworkVisualizer:
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

class ContinuousVisualizer:
    def __init__(self):
        self.epochs = []
        self.losses = []
        self.accuracies = []
        self.feature_maps = None
        self.kernels = None
        self.weights = None
        self.gradients = None
        
        plt.ion()
        self.fig, self.ax = plt.subplots(2, 3, figsize=(15, 10))
        self.running = True

        # Start visualization in a separate thread
        self.thread = threading.Thread(target=self.update_loop, daemon=True)
        self.thread.start()

        # Create axes for multiple buttons on the side
        button_ax1 = plt.axes([0.85, 0.8, 0.1, 0.05])  # Position 1
        self.button1 = Button(button_ax1, 'More Metrics')
    
    def update_metrics(self, epoch, loss, accuracy):
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.accuracies.append(accuracy)
    
    def update_feature_maps(self, feature_maps):
        self.feature_maps = feature_maps
    
    def update_kernels(self, kernels):
        self.kernels = kernels
    
    def update_weights(self, weights):
        self.weights = weights
    
    def update_gradients(self, gradients):
        self.gradients = gradients
    
    def update_loop(self):
        while self.running:
            self.refresh()
            time.sleep(0.5)  # Refresh rate
    
    def refresh(self):
        self.ax[0, 0].cla()
        self.ax[0, 1].cla()
        self.ax[0, 2].cla()
        self.ax[1, 0].cla()
        self.ax[1, 1].cla()
        self.ax[1, 2].cla()
        
        # Loss Plot
        if self.epochs:
            self.ax[0, 0].plot(self.epochs, self.losses, 'r-', label='Loss')
            self.ax[0, 0].set_title('Loss Over Time')
            self.ax[0, 0].legend()
        
        # Accuracy Plot
        if self.epochs:
            self.ax[0, 1].plot(self.epochs, self.accuracies, 'b-', label='Accuracy')
            self.ax[0, 1].set_title('Accuracy Over Time')
            self.ax[0, 1].legend()
        
        # Feature Maps
        if self.feature_maps is not None:
            self.ax[0, 2].imshow(self.feature_maps[:, :, 0], cmap='viridis')
            self.ax[0, 2].set_title('Feature Map')
        
        # Kernels
        if self.kernels is not None:
            self.ax[1, 0].imshow(self.kernels[:, :, 0, 0], cmap='gray')
            self.ax[1, 0].set_title('Kernel')
        
        # Weight Distribution
        if self.weights is not None:
            self.ax[1, 1].hist(self.weights.flatten(), bins=50, color='blue', alpha=0.7)
            self.ax[1, 1].set_title('Weight Distribution')
        
        # Gradient Distribution
        if self.gradients is not None:
            self.ax[1, 2].hist(self.gradients.flatten(), bins=50, color='red', alpha=0.7)
            self.ax[1, 2].set_title('Gradient Distribution')
        
        plt.draw()
        plt.pause(0.01)
    
    def stop(self):
        self.running = False
        self.thread.join()
        plt.ioff()
        plt.show()

