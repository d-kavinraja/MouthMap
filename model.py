
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, BatchNormalization, Reshape

def build_model(frame_length, image_height, image_width, vocabulary_size):
    model = Sequential([
        tf.keras.Input(shape=(frame_length, image_height, image_width, 1)),
        Conv3D(64, kernel_size=(3, 3, 3), activation='relu'),
        MaxPool3D(pool_size=(1, 2, 2)),
        BatchNormalization(),

        Conv3D(128, kernel_size=(3, 3, 3), activation='relu'),
        MaxPool3D(pool_size=(1, 2, 2)),
        BatchNormalization(),

        Conv3D(256, kernel_size=(3, 3, 3), activation='relu'),
        MaxPool3D(pool_size=(1, 2, 2)),
        BatchNormalization(),

        Reshape((-1, 256)),

        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.5),

        Bidirectional(LSTM(64)),
        Dropout(0.5),

        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(vocabulary_size, activation='softmax')
    ])
    return model