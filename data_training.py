import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report  # Import classification report
import matplotlib.pyplot as plt  # Importing matplotlib for plotting

# Initialize flags and variables
label = []
dictionary = {}
X = []
y = []

# Load the data from .npy files
for i in os.listdir():
    if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "labels"):
        try:
            temp_data = np.load(i)
            temp_label = i.split('.')[0]

            # Append data and label
            X.append(temp_data)
            y.append([temp_label] * len(temp_data))  # Repeat label for each sample

            # Add label to dictionary
            if temp_label not in dictionary:
                dictionary[temp_label] = len(label)
                label.append(temp_label)
        except Exception as e:
            print(f"Error loading {i}: {e}")

# Concatenate data
X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0).reshape(-1, 1)

# Convert labels to integers using dictionary
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")

# Convert labels to categorical (one-hot encoding)
y = to_categorical(y, num_classes=len(label))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the input shape for the model
ip = Input(shape=(X.shape[1],))

# Build the model
m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)
op = Dense(len(label), activation="softmax")(m) # Ensure the final layer has correct number of outputs

# Create the model
model = Model(inputs=ip, outputs=op)

# Compile the model
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

# Train the model and store the training history
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Get predictions on the test data
y_pred = model.predict(X_test)

# Convert predictions back to class labels
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Generate classification report
report = classification_report(y_true, y_pred, target_names=['Happy', 'Sad', 'Angry'])
print(report)

# Plotting the accuracy and loss curves
# Accuracy curve
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss curve
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Save the model and labels
model.save("model.h5")
np.save("labels.npy", np.array(label))
