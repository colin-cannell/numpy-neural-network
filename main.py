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

def load_mnist_images(filename, batch_size=1):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = np.frombuffer(f.read(16), dtype='>i4', count=4)
        images = np.frombuffer(f.read(), dtype='>u1').reshape(num_images, rows, cols, 1)

    # If batch size is greater than 1, return batches
    for i in range(0, num_images, batch_size):
        yield images[i:i+batch_size]

def load_mnist_labels(filename, batch_size=1):
    with open(filename, 'rb') as f:
        magic, num_labels = np.frombuffer(f.read(8), dtype='>i4')
        labels = np.frombuffer(f.read(), dtype='>u1')

    # If batch size is greater than 1, return batches
    for i in range(0, num_labels, batch_size):
        yield labels[i:i+batch_size]

batch_size = 10
train_data = load_mnist_images(train_images_path, batch_size)
train_labels = load_mnist_labels(train_labels_path, batch_size)

# Convert the generator to a numpy array
train_data = np.array(list(train_data))
train_labels = np.array(list(train_labels))

# print(f"train_data shape: {train_data.shape}")
# print(f"train_labels shape: {train_labels.shape}")

size = 10

train_data = train_data[:size]
train_labels = train_labels[:size]
# print(f"train_data shape: {train_data.shape}")
# print(f"train_labels shape: {train_labels.shape}")

# Normalize the images, puts the numbers on a scale of 0,255 in order to be better read by the network
train_data = train_data / 255.0
T, B, H, W, C  = train_data.shape
input_shape = (B, H, W, C)
# create a 2d array with 1s on the diagonal and 0s elsewhere    
train_labels = np.eye(10)[train_labels]

# Define the activation functions in use
conv_func = LeakyRelu()
dense1_func = LeakyRelu()
dense2_func = Softmax()

loss_funcion = CategoricalCrossEntropyLoss()

# Define the model architecture
model = NeuralNetwork()

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
dense1_out_shape = model.dense_output_shape(flatten_out_shape[0], dense1_out_neurons)
dense2_out_shape = model.dense_output_shape(batch_size, num_classes)

print(f"Conv1 output shape: {conv1_out_shape}")
print(f"Pool1 output shape: {pool1_out_shape}")
print(f"Conv2 output shape: {conv2_out_shape}")
print(f"Pool2 output shape: {pool2_out_shape}")
print(f"Flatten output shape: {flatten_out_shape}")
print(f"Dense1 output shape: {dense1_out_shape}")
print(f"Dense2 output shape: {dense2_out_shape}")

dropout_rate = 0.5

batch_size = 10

test_data = train_data[:batch_size]
test_labels = train_labels[:batch_size]

# Call the function to visualize
# visualize_batch(test_data, test_labels, batch_size)


# 1**Conv Layer 1**: 32 filters, (3x3) kernel, ReLU activation
model.add(Conv2D(input_shape=input_shape, kernel_size=kernel_size, filters=filters_1, activation=conv_func))

# **MaxPooling Layer**: Reduces spatial dimensions (downsampling)
model.add(MaxPool())

# **Conv Layer 2**: 64 filters, (3x3) kernel, ReLU activation
model.add(Conv2D(input_shape=pool1_out_shape, kernel_size=kernel_size, filters=filters_2, activation=conv_func)) 

# **MaxPooling Layer**: Downsampling again
model.add(MaxPool())

# **Flatten Layer**: Converts 2D feature maps into a 1D vector 
model.add(Flatten())

# **Dropout Layer**: Regularization to prevent overfitting
model.add(Dropout(dropout_rate))

# **Fully Connected (Dense) Layer 1**: 128 neurons, ReLU
model.add(Dense(flatten_out_shape[0], dense1_out_neurons, activation=dense1_func))

# **Dropout Layer**: Regularization to prevent overfitting
model.add(Dropout(dropout_rate))

# **Fully Connected (Dense) Layer 2**: 10 neurons (digits 0-9), Softmax activation
model.add(Dense(dense1_out_neurons, num_classes, activation=dense2_func))

# Train the model
model.train(train_data, train_labels, epochs=10, learning_rate=0.01, batch_size=batch_size, loss=loss_funcion, optimizer=Adam)
