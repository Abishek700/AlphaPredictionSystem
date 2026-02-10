# Required Libraries
from tensorflow.keras.models import Sequential  # To define the CNN model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Input
from tensorflow.keras.optimizers import Adam  # For model optimization

## @brief Train a Convolutional Neural Network (CNN) model.
#  @param features Input features for training.
#  @param labels Ground truth labels for the input features.
#  @return A tuple containing the trained model and training history.
#  @exception Exception Raised if any error occurs during model training.
def train_model(features, labels):
    """
    Train a Convolutional Neural Network (CNN) model using the provided features and labels.

    This function creates a CNN model, compiles it, and trains it using the specified features and labels.
    The model is trained for a fixed number of epochs with a specified batch size and a validation split.

    :param features: Input features for training the CNN model.
    :type features: numpy.ndarray
    :param labels: Ground truth labels corresponding to the input features.
    :type labels: numpy.ndarray
    :return: A tuple containing the trained model and the training history.
    :rtype: (tensorflow.keras.Model, tensorflow.keras.callbacks.History)
    :raises Exception: If an error occurs during model training.
    """
    try:
        # Create the CNN model
        model = create_cnn_model()

        # Train the model
        history = model.fit(
            features,
            labels,
            epochs=10,
            batch_size=10,
            validation_split=0.2
        )

        # Return the trained model and training history
        return model, history

    except Exception as e:
        # Log the error and re-raise it
        print(f"Error during model training: {e}")
        raise