# train_model.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import glob

from model import build_model

# Initial parameters
epochs = 100
lr = 1e-3
batch_size = 64
img_dims = (96, 96, 3)

data = []
labels = []

# Load image data from dataset
image_files = [f for f in glob.glob(r'dataset' + "/**/*", recursive=True) if
               not os.path.isdir(f) and f != 'dataset/labels.txt']
random.shuffle(image_files)

# Read labels from file
with open('dataset/labels.txt', 'r') as file:
    _labels = file.read().split('\n')

for img in image_files:
    image = cv2.imread(img)

    # Check if image was not loaded successfully
    if image is None:
        print(f"Error loading image: {img}")
        continue

    # If image is loaded successfully, resize it
    image = cv2.resize(image, (img_dims[0], img_dims[1]))
    image = img_to_array(image)
    data.append(image)

    label = img.split(os.path.sep)[-2]
    label = _labels.index(label)
    labels.append([label])

# Convert data and labels to numpy arrays and normalize
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

# Split data into training and testing sets
trainX, testX, trainY, testY = train_test_split(
    data, labels, test_size=0.2, random_state=42
)
trainY = to_categorical(trainY, num_classes=len(_labels))
testY = to_categorical(testY, num_classes=len(_labels))

# Data augmentation
aug = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Build and compile the model
model = build_model(
    width=img_dims[0],
    height=img_dims[1],
    depth=img_dims[2],
    classes=len(_labels)
)
opt = Adam(learning_rate=lr, weight_decay=lr / epochs)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the model
H = model.fit(
    aug.flow(trainX, trainY, batch_size=batch_size),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // batch_size,
    epochs=epochs,
    verbose=1
)
model.save('face_recognition_model.h5')

# Draw training loss and accuracy plot
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0, N), H.history["loss"], label="Train Loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="Validation Loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="Train Accuracy")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="Validation Accuracy")
plt.title("Loss and Accuracy During Training")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.savefig('plot.png')