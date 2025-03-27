import numpy as np
import matplotlib.pyplot as plt
from conv2d import Conv2D
from dense import Dense
from flatten import Flatten
from maxpool import MaxPool
from dropout import Dropout
from network import NeuralNetwork
from dropout import Dropout
from activations import *
from losses import *
from optimizers import Adam
from visualizer import Visualizer

train_images_path = "MNIST_ORG/train-images.idx3-ubyte"
train_labels_path = "MNIST_ORG/train-labels.idx1-ubyte"
test_images_path = "MNIST_ORG/t10k-images.idx3-ubyte"
test_labels_path = "MNIST_ORG/t10k-labels.idx1-ubyte"

def visualize_batch(images, labels, batch_size):
    fig, axes = plt.subplots(1, batch_size, figsize=(batch_size * 2, 2))
    
    for i in range(batch_size):
        ax = axes[i]
        ax.imshow(images[i].squeeze(), cmap='gray')  # Squeeze to remove single dimension if needed
        ax.axis('off')  # Hide axes
        ax.set_title(f"Label: {np.argmax(labels[i])}")  # Convert one-hot to label

    plt.show()
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

# Load the MNIST dataset
raw_images = load_mnist_images(train_images_path)
raw_labels = load_mnist_labels(train_labels_path)

# print(f"train_data shape: {train_data.shape}")
# print(f"train_labels shape: {train_labels.shape}")

# Normalize the images
train_images = raw_images / 255.0

# Reshape the images to (num_samples, 1, 28, 28) for grayscale (1 channel, 28x28)
train_images = train_images.reshape(60000, 28, 28, 1)
T, H, W, C = train_images.shape
input_shape = (H, W, C)

size = 50

train_labels = np.eye(10)[raw_labels]

train_images = train_images[:size]
train_labels = train_labels[:size]

# Define the activation functions in use
conv_func = Relu()
dense1_func = Relu()
dense2_func = Softmax()
adam = Adam()


loss_funcion = CategoricalCrossEntropyLoss()

visualizer = Visualizer()

# Define the model architecture
model = NeuralNetwork(visualize=visualizer)

num_classes = 10

filters_1 = 32
filters_2 = 64

kernel_size = 3
pool_size = 2

dense1_out_neurons = 128

conv1_out_shape = model.conv_output_shape(input_shape, kernel_size, filters_1)
pool1_out_shape = model.maxpool_output_shape(conv1_out_shape, pool_size)
conv2_out_shape = model.conv_output_shape(pool1_out_shape, kernel_size, filters_2)
pool2_out_shape = model.maxpool_output_shape(conv2_out_shape, pool_size)
flatten_out_shape = model.flatten_output_shape(pool2_out_shape)
dense1_out_shape = model.dense_output_shape(dense1_out_neurons)
dense2_out_shape = model.dense_output_shape(num_classes)

# print(f"Conv1 output shape: {conv1_out_shape}")
# print(f"Pool1 output shape: {pool1_out_shape}")
# print(f"Conv2 output shape: {conv2_out_shape}")
# print(f"Pool2 output shape: {pool2_out_shape}")
# print(f"Flatten output shape: {flatten_out_shape}")
# print(f"Dense1 output shape: {dense1_out_shape}")
# print(f"Dense2 output shape: {dense2_out_shape}")

dropout_rate = 0.5

batch_size = 10

# test_data = train_data[:batch_size]
# test_data = test_data.reshape(batch_size, H, W, C)

# test_labels = train_labels[:batch_size]
# test_labels = test_labels.reshape(batch_size, num_classes)

# 1**Conv Layer 1**: 32 filters, (3x3) kernel, ReLU activation
conv1 = Conv2D(input_shape=input_shape, kernel_size=kernel_size, filters=filters_1, activation=conv_func, visualize=visualizer, id=1)
model.add(conv1)

# **MaxPooling Layer**: Reduces spatial dimensions (downsampling)
maxpool1 = MaxPool(visualize=visualizer, id=1)
model.add(maxpool1)

# **Conv Layer 2**: 64 filters, (3x3) kernel, ReLU activation
conv2 = Conv2D(input_shape=pool1_out_shape, kernel_size=kernel_size, filters=filters_2, activation=conv_func, visualize=visualizer, id=2)
model.add(conv2)

# **MaxPooling Layer**: Downsampling again
maxpool2 = MaxPool(visualize=visualizer, id=2)
model.add(maxpool2)

# **Flatten Layer**: Converts 2D feature maps into a 1D vector 
flatten1 = Flatten()
model.add(flatten1)

# **Dropout Layer**: Regularization to prevent overfitting
dropout1 = Dropout(dropout_rate)
model.add(dropout1)

# **Fully Connected (Dense) Layer 1**: 128 neurons, ReLU
dense1 = Dense(flatten_out_shape, dense1_out_neurons, activation=dense1_func, visualize=visualizer, id=1)
model.add(dense1)

# **Dropout Layer**: Regularization to prevent overfitting
dropout2 = Dropout(dropout_rate)
model.add(dropout2)

# **Fully Connected (Dense) Layer 2**: 10 neurons (digits 0-9), Softmax activation
dense2 = Dense(dense1_out_shape, num_classes, activation=dense2_func, visualize=visualizer, id=2)
model.add(dense2)

# Train the model
model.train(train_images, train_labels, epochs=10, learning_rate=0.01, loss=loss_funcion, optimizer=adam)
