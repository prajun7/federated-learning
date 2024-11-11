import numpy as np
import matplotlib.pyplot as plt

# Function to read the MNIST image file
def read_mnist_images(filename):
    with open(filename, 'rb') as f:
        # Read the magic number and number of images
        magic_number, num_images, num_rows, num_cols = np.frombuffer(f.read(16), dtype='>i4')
        # Read the image data
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows, num_cols)
    return images

# Load the test images
images = read_mnist_images('data/mnist/MNIST/raw/t10k-images-idx3-ubyte')

# Display the first 5 images
for i in range(5):
    plt.imshow(images[i], cmap='gray')
    plt.title(f"Image {i+1}")
    plt.show()