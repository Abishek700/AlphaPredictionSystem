## @file config.py
#  @brief Contains default configuration parameters for the Kalman Filter and CNN model.

## @var DEFAULT_PARAMETERS
#  Dictionary containing default parameters for the Kalman Filter and CNN model.
DEFAULT_PARAMETERS = {
    ## Kalman Filter parameters
    "kalmanTransitionMatrices": [1],  # State transition matrix
    "kalmanObservationMatrices": [1],  # Observation matrix
    "kalmanInitialStateMean": 0,  # Initial state mean
    "kalmanInitialStateCovariance": 1,  # Initial state covariance
    "kalmanObservationCovariance": 1,  # Observation covariance
    "kalmanTransitionCovariance": 0.01,  # Transition covariance

    ## CNN model parameters
    "cnnInputShape": (3, 1),  # Shape of the input data for the CNN model
    "cnnEpochs": 10,  # Number of epochs for training
    "cnnBatchSize": 10,  # Batch size for training
    "cnnValidationSplit": 0.2,  # Fraction of data for validation
    "cnnLearningRate": 0.001  # Learning rate for the Adam optimizer
}