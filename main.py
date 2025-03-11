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

print("Loading MNIST dataset...")
train_images = load_mnist_images(train_images_path)
train_labels = load_mnist_labels(train_labels_path)
print("Loaded MNIST dataset")

# Normalize the images
train_images = train_images / 255.0
# Reshape the images to 28x28 with the channel dimension (grayscale)
train_images = train_images.reshape(-1, 28, 28, 1)
# One hot encode the labels
train_labels = np.eye(10)[train_labels]
train_labels = train_labels.reshape(-1, 10)

# Define the model architecture
model = NeuralNetwork()

# **Conv Layer 1**: 32 filters, (3x3) kernel, ReLU activation
model.add(Conv2D(input_shape=(28, 28, 1), kernel_size=(3, 3), depth=32))  
model.add(Activation(relu, None))

# **MaxPooling Layer**: Reduces spatial dimensions (downsampling)
model.add(MaxPool(pool_size=2, stride=2))

# **Conv Layer 2**: 64 filters, (3x3) kernel, ReLU activation
model.add(Conv2D(input_shape=(13, 13, 32), kernel_size=(3, 3), depth=64))
model.add(Activation(relu, None))

# **MaxPooling Layer**: Downsampling again
model.add(MaxPool(pool_size=2, stride=2))

# **Flatten Layer**: Converts 2D feature maps into a 1D vector 
model.add(Flatten())

# **Fully Connected (Dense) Layer 1**: 128 neurons, ReLU
model.add(Dense(1600, 128))
model.add(Activation(relu, None))

# **Fully Connected (Dense) Layer 2**: 10 neurons (digits 0-9), Softmax activation
model.add(Dense(128, 10))
model.add(Activation(softmax, None))

# Train the model
model.train(train_images, train_labels, epochs=10, learning_rate=0.01)

# Save the model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
