# model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def build_model(width, height, depth, classes):
    model = Sequential([
        Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(width, height, depth)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), padding="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(classes, activation="softmax")
    ])
    return model