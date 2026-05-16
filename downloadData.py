import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
import time
from dense import Dense
from ConvolutionalLayer import convolutionL
from reshape import Reshape
from activations import Tanh, Sigmoid, Softmax 
from losses import categorical_cross_entropy, categorical_cross_entropy_prime

from PIL import Image

def preprocess_data(x, y, limit):

    indices = []
    for i in range(10):
        indices.append(np.where(y == i)[0][:limit])
    

    all_indices = np.hstack(indices)
    all_indices = np.random.permutation(all_indices)
    
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    
    y = to_categorical(y, num_classes=10)
    y = y.reshape(len(y), 10, 1)
    return x, y


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 1000)
x_test, y_test = preprocess_data(x_test, y_test, 1000)


network = [
    convolutionL((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 10), 
    Softmax()       
]

epochs = 100
learning_rate = 0.002


for e in range(epochs):
    error = 0
    for x, y in zip(x_train, y_train):

        output = x
        for layer in network:
            output = layer.forward(output)
        

        error += categorical_cross_entropy(y, output)
        

        grad = categorical_cross_entropy_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)
    
    error /= len(x_train)
    print(f"{e + 1}/{epochs}, error={error}")
    time.sleep(1)


for x, y in zip(x_test, y_test):
    output = x
    for layer in network:
        output = layer.forward(output)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")




# This bit is what operates on images and spits the ascii value



def test_my_handwriting(image_path, trained_network):
    #Open image convert to grayscale ('L') resize to 28x28
    img = Image.open(image_path).convert('L').resize((28, 28))
    
    img_array = np.array(img, dtype="float32")
    
    # temp to store inverted
    processed_array = np.zeros((28, 28), dtype="float32")
    
    for row in range(28):
        for col in range(28):
            original_pixel = img_array[row][col]
            inverted_pixel = 255.0 - original_pixel
            
            #normalize pixel values between 0 and 1
            normalized_pixel = inverted_pixel / 255.0
            
            processed_array[row][col] = normalized_pixel
            

    print("\n--- Processed 28x28 Image Preview ---")
    for row in range(28):
        row_str = ""
        for col in range(28):
            # This converts it to either black or white
            row_str += "XX" if processed_array[row][col] > 0.3 else "  "
        print(row_str)
        
    network_input = processed_array.reshape(1, 28, 28)
    
    output = network_input
    for layer in trained_network:
        output = layer.forward(output)
        
    prediction = np.argmax(output)
    
    print("\n--- Final Prediction ---")
    print(f"Number is: {prediction}\n")

# TODO: update .jpg to actual picture file
test_my_handwriting("IMG_0512.jpeg", network)

"""
# --- pick a single random test sample and run it through the network ---
idx = np.random.randint(0, len(x_test))
sample_x = x_test[idx]   # shape: (1, 28, 28)
sample_y = y_test[idx]   # shape: (10, 1)

# forward pass
output = sample_x
for layer in network:
    output = layer.forward(output)

# convert output to 1D array and compute probabilities (if Softmax already applied, this is fine)
out_arr = np.array(output).reshape(-1)
# if last layer is Softmax, out_arr are probabilities; otherwise compute softmax:
exps = np.exp(out_arr - np.max(out_arr))
probs = exps / np.sum(exps)

pred = int(np.argmax(probs))
true = int(np.argmax(sample_y))

print(f"Random test index: {idx}")
print(f"Predicted: {pred}, True: {true}")
print("Top probabilities:")
top3 = probs.argsort()[-3:][::-1]
for i in top3:
    print(f"  {i}: {probs[i]:.4f}")
"""