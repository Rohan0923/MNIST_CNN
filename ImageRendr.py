from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def test_my_handwriting(image_path):
    # Open image, convert to grayscale, resize to 28x28
    img = Image.open(image_path).convert('L').resize((28, 28))
    
    img_array = np.array(img, dtype="float32")

    # Invert + normalize
    processed_array = (255.0 - img_array) / 255.0

    # Show the processed image
    plt.imshow(processed_array, cmap='gray')
    plt.title("Processed 28x28 Image")
    plt.axis('off')
    plt.show()

test_my_handwriting("IMG_0512.jpg")

