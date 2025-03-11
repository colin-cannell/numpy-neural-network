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
train_images = load_mnist_images(train_images_path)
train_labels = load_mnist_labels(train_labels_path)
# print("Loaded MNIST dataset")

# Normalize the images
train_images = train_images / 255.0

# Reshape the images to (num_samples, 1, 28, 28) for grayscale (1 channel, 28x28)
train_images = train_images.reshape(60000, 28, 28, 1)

# One hot encode the labels
train_labels = np.eye(10)[train_labels]
train_labels = train_labels.reshape(-1, 10)

# Define the model architecture
model = NeuralNetwork()

# **Conv Layer 1**: 32 filters, (3x3) kernel, ReLU activation
model.add(Conv2D(input_shape=(28, 28, 1), kernel_size=3, depth=32))  
model.add(Activation(Relu().forward, Relu().backward))

# **MaxPooling Layer**: Reduces spatial dimensions (downsampling)
model.add(MaxPool(pool_size=2, stride=2))

# **Conv Layer 2**: 64 filters, (3x3) kernel, ReLU activation
model.add(Conv2D(input_shape=(26, 26, 1), kernel_size=3, depth=64))  # Updated input shape
model.add(Activation(Relu().forward, Relu().backward))

# **MaxPooling Layer**: Downsampling again
model.add(MaxPool(pool_size=2, stride=2))

# **Flatten Layer**: Converts 2D feature maps into a 1D vector 
model.add(Flatten())

# **Fully Connected (Dense) Layer 1**: 128 neurons, ReLU
model.add(Dense(768, 128))  # Update input size to 2304 (output of Flatten layer)model.add(Dense(1600, 128))
model.add(Activation(Relu().forward, Relu().backward))

# **Fully Connected (Dense) Layer 2**: 10 neurons (digits 0-9), Softmax activation
model.add(Dense(128, 10))
model.add(Activation(Softmax().forward, Softmax().backward))

# Train the model
model.train(train_images, train_labels, epochs=10, learning_rate=0.01)


# Save the model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
