import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import seaborn as sns
import time
import threading

# general view

# buttons of
# accuracy / loss
# weights
# gradients

# have id for each layer 
# conv layer 1 has different id than conv layer 2

# accuracy / loss
# epoch accuracy
# epoch loss

# gradients
# gradient of each individual layer
# stack gradients on top of each other

class Visualizer:
    def __init__(self):
        # plt.axes([left, bottom, width, height])
        # left X-position (left edge of the button) as a fraction of the figure width 0.7 (70% from the left edge of the figure)
        # bottom Y-position (bottom edge of the button) as a fraction of the figure height 0.05 (5% from the bottom of the figure)
        # width as a fraction of the figure width 0.1 (10% of the figure width)
        # height as a fraction of the figure height 0.075 (7.5% of the figure height)

        self.kernels = {}
        self.kernels_bias = {}
        self.weights = {}
        self.weights_bias = {}
        self.gradients = {}
        self.feature_maps = {}
        self.pooled_feature_maps = {}

        self.loss = []
        self.accuracy = []

        self.view_mode = "accuracy_loss"

        self.fig, self.axs = plt.subplots(2, 1, figsize=(10, 5))

        # Create buttons
        self.buttons = []
        button_labels = [
            ('Accuracy/Loss', self.show_accuracy_loss),
            ('Gradients', self.show_gradients),
            ('Weights/Bias', self.show_weights_bias),
            ('Kernels/Bias', self.show_kernels_bias),
            ('Feature Maps', self.show_feature_maps),
            ('Pooled Feature Maps', self.show_pooled_feature_maps)
        ]

        self.setup_buttons(button_labels)

        plt.ion()
        plt.show()

        # Start the update loop in a separate thread
        self.running = True
        update_thread = threading.Thread(target=self.update_loop, daemon=True)
        update_thread.start()
    
    def setup_buttons(self, button_labels):
        height = 0.075
        width = 0.2
        bottom = 0.9
        left = 0.05
        inc = 0.1

        for i, (label, callback) in enumerate(button_labels):
            ax_button = plt.axes([left, bottom - i*inc, width, height])
            button = Button(ax_button, label)
            button.on_clicked(callback)
            self.buttons.append(button)



    # update functions
    def update_accuracy_loss(self, loss, accuracy):
        self.loss = loss
        self.accuracy = accuracy
        
    def update_weights_bias(self, id, weights, bias):
        self.weights[id] = weights
        self.weights_bias[id] = bias
        
    def update_kernels_bias(self, id, kernels, bias):
        self.kernels[id] = kernels
        self.kernels_bias[id] = bias
    
    def update_gradients(self, name, gradients):
        self.gradients[name] = gradients

    def update_feature_maps(self, id, feature_maps):
        self.feature_maps[id] = feature_maps

    def update_pooled_feature_maps(self, id, pooled_feature_maps):
        self.pooled_feature_maps[id] = pooled_feature_maps

    # button functions
    def show_accuracy_loss(self):
        self.view_mode = "accuracy_loss"
        self.refresh()

    def show_gradients(self):
        self.view_mode = "gradients"
        self.refresh()

    def show_weights_bias(self):
        self.view_mode = "weights_bias"
        self.refresh()

    def show_kernels_bias(self):
        self.view_mode = "kernels_bias"
        self.refresh()

    def show_feature_maps(self):
        self.view_mode = "feature_maps"
        self.refresh()

    def show_pooled_feature_maps(self):
        self.view_mode = "pooled_feature_maps"
        self.refresh()

    # views
    def accuracy_loss_view(self):
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            self.fig, self.axs = plt.subplots(2, 1, figsize=(10, 5))
        else:
            self.fig.clf()
            self.axs = self.fig.subplots(2, 1)

        self.axs[0].plot(self.accuracy)
        self.axs[0].set_title("Accuracy per Epoch")
        self.axs[0].set_xlabel("Epoch")
        self.axs[0].set_ylabel("Accuracy")

        self.axs[1].plot(self.loss)
        self.axs[1].set_title("Loss per Epoch")
        self.axs[1].set_xlabel("Epoch")
        self.axs[1].set_ylabel("Loss")

        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

    def gradients_view(self):
        gradients = list(self.gradients.items())

        num_layers = len(gradients)
        if num_layers == 0:
            return
        
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            self.fig, self.axs = plt.subplots(num_layers, 1, figsize=(10, 5 * num_layers))
        else:
            self.fig.clf()
            self.axs = self.fig.subplots(num_layers, 1)

        for i, (name, gradient) in enumerate(gradients):
            self.axs[i].plot(gradient.flatten())
            self.axs[i].set_title(f"Gradients {name}")
            self.axs[i].legend([f"Gradients {name}"])

        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

    def weights_bias_view(self):
        weights = list(self.weights.items())
        biases = list(self.weights_bias.items())

        num_layers = len(weights)
        if num_layers == 0:
            return
        
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            self.fig, self.axs = plt.subplots(num_layers, 2, figsize=(10, 5 * num_layers))
        else:
            self.fig.clf()
            self.axs = self.fig.subplots(num_layers, 2)

        for i, ((w_id, weight), (b_id, bias)) in enumerate(zip(weights, biases)):
            self.axs[i][0].plot(weight.flatten())
            self.axs[i][0].set_title(f"Weights {w_id}")
            self.axs[i][0].legend([f"Weights {w_id}"])

            self.axs[i][1].plot(bias)
            self.axs[i][1].set_title(f"Bias {b_id}")
            self.axs[i][1].legend([f"Bias {b_id}"])

        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

    def kernels_bias_view(self):
        kernels = list(self.kernels.items())
        biases = list(self.kernels_bias.items())

        num_layers = len(kernels)
        if num_layers == 0:
            return

        # Reuse existing figure instead of creating a new one
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            self.fig, self.axs = plt.subplots(num_layers, 2, figsize=(10, 5 * num_layers))
        else:
            self.fig.clf()
            self.axs = self.fig.subplots(num_layers, 2)

        for i, ((k_id, kernel), (b_id, bias)) in enumerate(zip(kernels, biases)):
            self.axs[i][0].plot(kernel.flatten())
            self.axs[i][0].set_title(f"Kernels {k_id}")
            self.axs[i][0].legend([f"Kernels {k_id}"])

            self.axs[i][1].plot(bias)
            self.axs[i][1].set_title(f"Bias {b_id}")
            self.axs[i][1].legend([f"Bias {b_id}"])
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

    def feature_maps_view(self):
        feature_maps = list(self.feature_maps.items())

        num_layers = len(feature_maps)
        if num_layers == 0:
            return

        if self.fig is None or not plt.fignum_exists(self.fig.number):
            self.fig, self.axs = plt.subplots(num_layers, 1, figsize=(10, 5 * num_layers))
        else:
            self.fig.clf()
            self.axs = self.fig.subplots(num_layers, 1)

        for i, (id, feature_map) in enumerate(feature_maps):
            self.axs[i].imshow(feature_map)
            self.axs[i].set_title(f"Feature Maps {id}")
            self.axs[i].legend([f"Feature Maps {id}"])

        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

    def pooled_feature_maps_view(self):
        pooled_feature_maps = list(self.pooled_feature_maps.items())

        num_layers = len(pooled_feature_maps)
        if num_layers == 0:
            return

        if self.fig is None or not plt.fignum_exists(self.fig.number):
            self.fig, self.axs = plt.subplots(num_layers, 1, figsize=(10, 5 * num_layers))
        else:
            self.fig.clf()
            self.axs = self.fig.subplots(num_layers, 1)

        for i, (id, pooled_feature_map) in enumerate(pooled_feature_maps):
            self.axs[i].imshow(pooled_feature_map)
            self.axs[i].set_title(f"Pooled Feature Maps {id}")
            self.axs[i].legend([f"Pooled Feature Maps {id}"])

        plt.tight_layout()
        plt.draw()
        plt
    
    def update_loop(self):
        """Improved update loop with better thread handling"""
        try:
            while self.running:
                # Use plt.pause to process GUI events
                self.refresh()
                plt.pause(0.01)  # Slightly longer pause to ensure GUI responsiveness
        except Exception as e:
            print(f"Error in update loop: {e}")
        finally:
            plt.close(self.fig)

    def refresh(self):
        """Enhanced refresh method with additional error handling"""
        try:
            if not plt.get_fignums():
                self.running = False
                return
            
            # Clear the figure before redrawing
            plt.clf()
            
            # Choose view based on current mode
            if self.view_mode == "accuracy_loss":
                self.accuracy_loss_view()
            elif self.view_mode == "gradients":
                self.gradients_view()
            elif self.view_mode == "weights_bias":
                self.weights_bias_view()
            elif self.view_mode == "kernels_bias":
                self.kernels_bias_view()
            elif self.view_mode == "feature_maps":
                self.feature_maps_view()
            elif self.view_mode == "pooled_feature_maps":
                self.pooled_feature_maps_view()
        except Exception as e:
            print(f"Error in refresh: {e}")
        


