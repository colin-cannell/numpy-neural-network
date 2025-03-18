import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# network
# output gradient
# loss
# accuracy

class conv:
    # feature maps
    def conv_feature_maps(feature_maps, layer_name="Conv Layer"):
        num_filters = feature_maps.shape[-1]
        fig, axes = plt.subplots(1, num_filters, figsize=(20, 5))
        
        for i in range(num_filters):
            ax = axes[i] if num_filters > 1 else axes
            ax.imshow(feature_maps[0, :, :, i], cmap="viridis")  # Assuming batch size = 1
            ax.set_title(f"{layer_name} - Filter {i+1}")
            ax.axis("off")
        
        plt.suptitle(f"Feature Maps from {layer_name}")
        plt.show()

    # kernels
    def conv_kernels(kernels, layer_name="Conv Layer"):
        num_filters = kernels.shape[-1]
        fig, axes = plt.subplots(1, num_filters, figsize=(20, 5))

        for i in range(num_filters):
            ax = axes[i] if num_filters > 1 else axes
            ax.imshow(kernels[:, :, 0, i], cmap="gray")  # Assuming 1 channel for visualization
            ax.set_title(f"{layer_name} - Filter {i+1}")
            ax.axis("off")

        plt.suptitle(f"Kernels from {layer_name}")
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
            ax1.imshow(before_pooling[0, :, :, i], cmap="viridis")
            ax1.set_title(f"{layer_name} - Filter {i+1} (Before)")
            ax1.axis("off")

            # After Pooling
            ax2 = axes[1, i] if num_filters > 1 else axes[1]
            ax2.imshow(after_pooling[0, :, :, i], cmap="viridis")
            ax2.set_title(f"{layer_name} - Filter {i+1} (After)")
            ax2.axis("off")

        plt.suptitle(f"Pooled Feature Maps - {layer_name}")
        plt.show()

    # activation distribution before and after pooling
    def maxpool_activation_distribution(before_pooling, after_pooling, layer_name="MaxPool Layer"):
        before_activations = before_pooling.flatten()
        after_activations = after_pooling.flatten()

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
        before_activations = before_flatten.flatten()
        after_activations = after_flatten.flatten()

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
        sns.heatmap(activations, cmap="viridis", xticklabels=False, yticklabels=False)
        plt.xlabel("Neurons")
        plt.ylabel("Samples (Batch)")
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
        grad_values = layer.weights.flatten()  # Assuming gradients are stored in layer.weights

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



