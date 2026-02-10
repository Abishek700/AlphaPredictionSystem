## @file cnnmodel.py
#  @brief Contains functions to define, compile, and configure a Convolutional Neural Network (CNN) model.
#
#  This module provides utilities to create a CNN model using TensorFlow's Keras API.
#  The CNN architecture is designed for sequence data with features such as:
#  - A 1D convolutional layer to extract features from the input sequence.
#  - MaxPooling to reduce dimensionality and computational complexity.
#  - Dense layers for classification tasks.
#  - A sigmoid activation in the output layer for binary classification.
#
#  Example Use Case:
#  - Recognizing patterns in gait data (e.g., normal vs. Parkinson's gait).
#  - Processing sequential data with fixed input shapes.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Input
from tensorflow.keras.optimizers import Adam
import logging


## @brief Defines and compiles a Convolutional Neural Network (CNN) model.
#
#  This function creates a CNN with the following architecture:
#  - **Input Layer**: Accepts input data with the specified shape from `params["cnnInputShape"]`.
#  - **Convolutional Layer**: Extracts features using 64 filters with a kernel size of 2.
#  - **MaxPooling Layer**: Down-samples the feature map by taking the maximum value in each pool.
#  - **Flatten Layer**: Converts the 2D feature map into a 1D feature vector.
#  - **Dense Layers**:
#    - A hidden dense layer with 50 neurons and ReLU activation for non-linearity.
#    - An output dense layer with a sigmoid activation for binary classification.
#
#  The model is compiled with:
#  - **Loss Function**: Binary cross-entropy, suitable for binary classification.
#  - **Optimizer**: Adam, with a configurable learning rate from `params["cnnLearningRate"]`.
#  - **Metrics**: Accuracy, to evaluate classification performance.
#
#  @param params A dictionary containing the CNN model configuration:
#     - **"cnnInputShape"**: (tuple) Shape of the input data (e.g., `(3, 1)` for a 3-feature sequence).
#     - **"cnnLearningRate"**: (float) Learning rate for the Adam optimizer (e.g., `0.001`).
#  @return Returns a compiled Keras Sequential model ready for training.
#  @exception Raises an exception if the model cannot be created, compiled, or configured properly.
#
#  Example:
#  @code{.py}
#  params = {
#      "cnnInputShape": (3, 1),
#      "cnnLearningRate": 0.001
#  }
#  model = createCnnModel(params)
#  model.summary()
#  @endcode
def createCnnModel(params):
    """Define and compile a CNN model."""
    try:
        # Log the start of the CNN creation process
        logging.info("Creating CNN model.")

        # Define the Sequential CNN model architecture
        model = Sequential([
            # Input layer with the specified shape
            Input(shape=params["cnnInputShape"]),
            # Convolutional layer to extract features
            Conv1D(64, kernel_size=2, activation='relu'),
            # MaxPooling layer to reduce feature map size
            MaxPooling1D(pool_size=2),
            # Flatten layer to convert feature map into a vector
            Flatten(),
            # Fully connected (dense) hidden layer
            Dense(50, activation='relu'),
            # Output layer for binary classification
            Dense(1, activation='sigmoid')
        ])

        # Configure the optimizer with the specified learning rate
        optimizer = Adam(learning_rate=params["cnnLearningRate"])

        # Compile the model with the specified loss function and metrics
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # Log successful model creation
        logging.info("CNN model created successfully.")

        return model
    except Exception as e:
        # Log any errors encountered during model creation
        logging.error(f"Error creating CNN model: {e}")
        raise