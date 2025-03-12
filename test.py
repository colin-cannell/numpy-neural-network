import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load the model from the file
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Open image and label files for MNIST test set
with open("MNIST_ORG/t10k-images.idx3-ubyte", "rb") as file:
    image_data = file.read()

with open("MNIST_ORG/t10k-labels.idx1-ubyte", "rb") as file:
    label_data = file.read()

# Skip the header (first 16 bytes for images, 8 bytes for labels)
image_data = list(image_data[16:])
label_data = list(label_data[8:])

# Define image dimensions for MNIST
height = 28
width = 28
size = width * height

# Split image data into a list of 28x28 images
image_list = []
for i in range(0, len(image_data), size):
    if i + size <= len(image_data):
        image_chunk = image_data[i:i + size]
        image_list.append(np.array(image_chunk).reshape(height, width))

# Add a channel dimension (since it's grayscale, we use 1 for the channel)
image_list = [np.expand_dims(image, axis=-1) for image in image_list]

# Convert image_list into an array for easier processing
image_array = image_list[:10]  # Display only the first 10 images for now

# Preprocess the image data as needed (flattening for model input)
X_test = np.array(image_array).reshape(-1, 28*28)  # Flatten images to 1D

# Predict using the loaded model
predictions = model.predict(X_test)

# Set up plot grid
num_images = len(image_array)
cols = 5  # Number of columns in the grid
rows = (num_images + cols - 1) // cols  # Calculate the number of rows needed

# Create figure with subplots
fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
axes = axes.flatten()

# Loop through images and plot them
for i in range(rows * cols):
    if i < num_images:
        axes[i].imshow(image_array[i], cmap='gray')  # Display the image in grayscale
        axes[i].axis('off')  # Turn off axis
        axes[i].set_title(f"True: {label_data[i]} Pred: {predictions[i]}", fontsize=12)  # Show label and prediction
    else:
        axes[i].axis('off')  # Hide unused subplots

# Adjust layout and display
plt.tight_layout()
plt.show()
