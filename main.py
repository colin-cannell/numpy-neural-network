import numpy as np
import pickle
from conv2d import Conv2D
from dense import Dense
from flatten import Flatten
from maxpool import MaxPool
from dropout import Dropout
from network import NeuralNetwork

train_images = "MNIST_ORG/train-images.idx3-ubyte"
train_labels = "MNIST_ORG/train-labels.idx1-ubyte"
test_images = "MNIST_ORG/t10k-images.idx3-ubyte"
test_labels = "MNIST_ORG/t10k-labels.idx1-ubyte"

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
train_images = load_mnist_images(train_images)
train_labels = load_mnist_labels(train_labels)
print("Loaded MNIST dataset")

# Normalize the images
train_images = train_images / 255.0
# Reshape the images to 28x28
train_images = train_images.reshape(-1, 28, 28, 1)  # Added channel dimension for Conv2D
# One hot encode the labels
train_labels = np.eye(10)[train_labels]
# Reshape the labels to 10
train_labels = train_labels.reshape(-1, 10)

# Create the model
model = NeuralNetwork()

# Add the layers
# Ensure the input to Conv2D is 4D
kernel_shape = (3, 3)  # (kernel_height, kernel_width)
conv_layer = Conv2D(input_shape=(28, 28, 1), kernel_size=kernel_shape, depth=32)  # Initialize Conv2D
model.add(conv_layer)  # Add Conv2D to the model

maxpool_layer = MaxPool(pool_size=2, strides=2)  # Initialize MaxPool
model.add(maxpool_layer)  # Add MaxPool to the model

model.add(Flatten())  # Add Flatten after MaxPool output
model.add(Dense(32 * 14 * 14, 128))  # Adjust the input size based on MaxPool output
model.add(Dropout(rate=0.5))
model.add(Dense(128, 10))

# Train the model
print(f"Train images shape: {train_images.shape}")  # Check the shape
model.train(train_images, train_labels, epochs=10, learning_rate=0.01)

# Load the test dataset
test_images = load_mnist_images(test_images)
test_labels = load_mnist_labels(test_labels)
# Normalize the test images
test_images = test_images / 255.0
# Reshape the test images to 28x28
test_images = test_images.reshape(-1, 28, 28, 1)  # Added channel dimension for Conv2D
# One hot encode the test labels
test_labels = np.eye(10)[test_labels]
# Reshape the test labels to 10
test_labels = test_labels.reshape(-1, 10)
# Evaluate the model
output = model.forward(test_images)
# Calculate the loss
loss = model.loss_function(test_labels, output)
print(f"Test loss: {loss:.4f}")
# Calculate the accuracy
correct = 0
for i in range(len(test_images)):
    if np.argmax(output[i]) == np.argmax(test_labels[i]):
        correct += 1

accuracy = correct / len(test_images)
print(f"Test accuracy: {accuracy:.4f}")

# Save the model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
