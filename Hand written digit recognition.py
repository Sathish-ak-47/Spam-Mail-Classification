# mnist_with_dataset.py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

# Reproducible
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

#  Load dataset 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

# Add channel dimension
x_train = x_train[..., np.newaxis]  # shape (N,28,28,1)
x_test  = x_test[..., np.newaxis]

#  CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28,28,1)),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

EPOCHS = 6
history = model.fit(x_train, y_train, validation_split=0.1,
                    epochs=EPOCHS, batch_size=128, verbose=2)

# Evaluate on test set and print clear result
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest loss = {loss:.4f}    Test accuracy = {acc:.4f}\n")

# Predict and show some examples with clear printed output and images
def show_predictions(n=8):
    idx = np.random.choice(len(x_test), n, replace=False)
    preds = model.predict(x_test[idx], verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    for i, tid in enumerate(idx):
        prob = preds[i, pred_labels[i]]
        true = y_test[tid]
        pred = pred_labels[i]
        print(f"Example {i+1}: True={true}  Pred={pred}  Confidence={prob:.3f}")
        plt.imshow(x_test[tid].squeeze(), cmap="gray")
        plt.title(f"True: {true}  Pred: {pred}  Conf: {prob:.3f}")
        plt.axis("off")
        plt.show()

# run demonstration
if __name__ == "__main__":
    show_predictions(8)
