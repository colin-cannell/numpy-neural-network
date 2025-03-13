import numpy as np
import pickle
from conv2d import Conv2D
from dense import Dense
from flatten import Flatten
from maxpool import MaxPool
from dropout import Dropout
from network import NeuralNetwork
from activation import Activation
from activations import *

train_images_path = "MNIST_ORG/train-images.idx3-ubyte"
train_labels_path = "MNIST_ORG/train-labels.idx1-ubyte"
test_images_path = "MNIST_ORG/t10k-images.idx3-ubyte"
test_labels_path = "MNIST_ORG/t10k-labels.idx1-ubyte"

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        # Read the magic number and number of images
        magic, num_images, rows, cols = np.frombuffer(f.read(16), dtype='>i4')
        # Read the image data
        images = np.frombuffer(f.read(), dtype='>u1').reshape(num_images, rows, cols)
    return images

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        # Read the magic number and number of labels
        magic, num_labels = np.frombuffer(f.read(8), dtype='>i4')
        # Read the label data
        labels = np.frombuffer(f.read(), dtype='>u1')
    return labels

# print("Loading MNIST dataset...")
train_data = load_mnist_images(train_images_path)
train_labels = load_mnist_labels(train_labels_path)
# print("Loaded MNIST dataset")

size = 500
train_data = train_data[:size].T
train_labels = train_labels[:size].T

# Normalize the images, puts the numbers on a scale of 0,255 in order to be better read by the network
train_data = train_data / 255.0

input_shape = (28, 28, 1)

# Reshape the images to (num_samples, 1, 28, 28) for grayscale (1 channel, 28x28)
train_data = train_data.reshape(size, input_shape[0], input_shape[1], input_shape[2])

# create a 2d array with 1s on the diagonal and 0s elsewhere    
train_labels = np.eye(10)[train_labels]
train_labels = train_labels.reshape(-1, 10)

# Define the activation functions in use
relu = Relu()
softmax = Softmax()

# Define the model architecture
model = NeuralNetwork()

filters_1 = 32
filters_2 = 64

kernel_size = 3
pool_size = 2

dense1_out_neurons = 128
dense2_out_neurons = 10

conv1_out_shape = model.conv_output_shape(input_shape, kernel_size, filters_1)
print(f"Conv1 output shape: {conv1_out_shape}")
pool1_out_shape = model.maxpool_output_shape(conv1_out_shape, pool_size)
print(f"Pool1 output shape: {pool1_out_shape}")

conv2_out_shape = model.conv_output_shape(pool1_out_shape, kernel_size, filters_2)
print(f"Conv2 output shape: {conv2_out_shape}")
pool2_out_shape = model.maxpool_output_shape(conv2_out_shape, pool_size)
print(f"Pool2 output shape: {pool2_out_shape}")

flatten_out_shape = model.flatten_output_shape(pool2_out_shape)
print(f"Flatten output shape: {flatten_out_shape}")

dense1_out_shape = model.dense_output_shape(dense1_out_neurons)
print(f"Dense1 output shape: {dense1_out_shape}")
dense2_out_shape = model.dense_output_shape(dense2_out_neurons)
print(f"Dense2 output shape: {dense2_out_shape}")


# 1**Conv Layer 1**: 32 filters, (3x3) kernel, ReLU activation
model.add(Conv2D(input_shape=input_shape, kernel_size=kernel_size, filters=filters_1))
model.add(Activation(relu.relu, relu.relu_prime))

# **MaxPooling Layer**: Reduces spatial dimensions (downsampling)
model.add(MaxPool())

# **Conv Layer 2**: 64 filters, (3x3) kernel, ReLU activation
model.add(Conv2D(input_shape=pool1_out_shape, kernel_size=kernel_size, filters=filters_2)) 
model.add(Activation(relu.relu, relu.relu_prime))

# **MaxPooling Layer**: Downsampling again
model.add(MaxPool())

# **Flatten Layer**: Converts 2D feature maps into a 1D vector 
model.add(Flatten())

# **Fully Connected (Dense) Layer 1**: 128 neurons, ReLU
model.add(Dense(flatten_out_shape[0], dense1_out_neurons))

# **Fully Connected (Dense) Layer 2**: 10 neurons (digits 0-9), Softmax activation
model.add(Dense(dense1_out_neurons, dense2_out_neurons))
model.add(Activation(softmax.forward, softmax.backward))

# Train the model
model.train(train_data, train_labels, epochs=10, learning_rate=0.01, loss_function=CrossEntropyLoss().forward, loss_derivative=CrossEntropyLoss().backward)

def save_weights(model, filename_prefix):
    """
    Save the weights of each layer in the model.
    @param model: The neural network model.
    @param filename_prefix: Prefix for the saved weight files.
    """
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()  # Assuming the layer has a `get_weights` method
        for j, weight in enumerate(weights):
            np.save(f"{filename_prefix}_layer_{i}_weight_{j}.npy", weight)  # Save each weight matrix
    print("Weights saved successfully!")
