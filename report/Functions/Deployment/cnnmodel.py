"""
@file cnnmodel.py
@brief Defines and compiles the Convolutional Neural Network (CNN) model.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Input
from tensorflow.keras.optimizers import Adam
import logging

def create_cnn_model():
    """
    @brief Creates and compiles the CNN model.
    @return Compiled CNN model.
    """
    logging.info("Creating CNN model.")
    try:
        model = Sequential([
            Input(shape=(3, 1)),
            Conv1D(64, kernel_size=2, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(50, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        logging.info("CNN model created successfully.")
        return model
    except Exception as e:
        logging.error(f"Error in CNN model creation: {e}")
        raise