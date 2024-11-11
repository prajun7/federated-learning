# A method that prints both the image and its corresponding label together for each image. 
# This code will load both the images and their labels, and display the image along with its label in the title.

import numpy as np
import matplotlib.pyplot as plt

# number of images it will display
NUM_IMAGES = 15

# Function to read the MNIST image file
# This function loads the image data from the .ubyte file. The images are reshaped into 28x28 pixel arrays.
def read_mnist_images(filename):
    with open(filename, 'rb') as f:
        # Read the magic number and number of images
        magic_number, num_images, num_rows, num_cols = np.frombuffer(f.read(16), dtype='>i4')
        # Read the image data
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows, num_cols)
    return images

# Function to read the MNIST labels file
# This function reads the labels from the .ubyte file. Each label corresponds to a digit (0-9)
def read_mnist_labels(filename):
    with open(filename, 'rb') as f:
        # Read the magic number and number of labels
        magic_number, num_labels = np.frombuffer(f.read(8), dtype='>i4')
        # Read the label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Function to display the image with its corresponding label
# This function loops through the first few images (as specified by num_images) and displays each image using matplotlib, with its corresponding label as the title.
def display_images_and_labels(image_file, label_file, num_images):
    # Read the images and labels
    images = read_mnist_images(image_file)
    labels = read_mnist_labels(label_file)
    
    # Display images and corresponding labels
    for i in range(num_images):
        plt.imshow(images[i], cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.show()

# Path to your MNIST files
image_file = 'data/mnist/MNIST/raw/t10k-images-idx3-ubyte'
label_file = 'data/mnist/MNIST/raw/t10k-labels-idx1-ubyte'

# Call the function to display images with their labels
display_images_and_labels(image_file, label_file, num_images=NUM_IMAGES)