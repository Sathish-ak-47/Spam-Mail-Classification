import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import cv2
import numpy as np

# 1. Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# 2. Build the model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=3)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)

# User input image (should contain 2 digits horizontally)
img_path = input("Enter image path containing TWO digits: ")

# Load in grayscale
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Convert to numpy array
img = np.array(img)

# Split the image into two halves
h, w = img.shape

mid = w // 2
left_digit  = img[:, :mid]
right_digit = img[:, mid:]

# Function to preprocess each digit
def preprocess(d):
    d = cv2.resize(d, (28, 28))       # Resize
    d = d / 255.0                     # Normalize
    d = d.reshape(1, 28, 28)          # Reshape for model
    return d

left_prep  = preprocess(left_digit)
right_prep = preprocess(right_digit)

# Predict
pred_left  = model.predict(left_prep)
pred_right = model.predict(right_prep)

digit1 = np.argmax(pred_left)
digit2 = np.argmax(pred_right)

print("\nPredicted digits:", digit1, digit2)

# Show both digit crops
plt.figure(figsize=(6,3))

plt.subplot(1,2,1)
plt.imshow(left_digit, cmap='gray')
plt.title("Digit 1: " + str(digit1))

plt.subplot(1,2,2)
plt.imshow(right_digit, cmap='gray')
plt.title("Digit 2: " + str(digit2))

plt.show()
