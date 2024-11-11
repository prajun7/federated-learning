import numpy as np
import matplotlib.pyplot as plt

def read_mnist_labels(filename):
    with open(filename, 'rb') as f:
        # Read the magic number and number of labels
        magic_number, num_labels = np.frombuffer(f.read(8), dtype='>i4')
        # Read the label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Load the test labels
labels = read_mnist_labels('data/mnist/MNIST/raw/t10k-labels-idx1-ubyte')

# Print the label for the first image
print(f"Label for first image: {labels[0]}")