############################
# Creator of the File: Fedor Kotiaev and Abishek Senthilkumar
# Date created: 10.01.2025
# Path: Code/MicroPython/Project code
# Version: 1.5
# Reviewed by: 
# Review Date: 
############################
#
# @mainpage Parkinson Gait Pattern Recognition
# @section intro_sec Introduction
# This script serves as the main entry point for the Parkinson Gait Pattern Recognition project. 
# It coordinates the pipeline, including data loading, preprocessing, model training, exporting, and visualization.
#
# @section overview_sec Overview
# The main pipeline includes:
# - Loading datasets (`data/normalpatterns.csv` and `data/parkinsonpatterns.csv`).
# - Preprocessing data using the Kalman Filter to smooth noisy IMU data.
# - Training a Convolutional Neural Network (CNN) for binary classification.
# - Exporting the trained model to TensorFlow Lite and generating a `.h` file for Arduino integration.
# - Visualizing the model’s predictions to evaluate performance.
#
# @section structure_sec Project Directory Structure
# ```
# .
# ├── models/
# │   └── gait_model.h            # Exported header file for Arduino integration
# ├── src/
# │   ├── dataloader.py           # Module to load CSV files
# │   ├── kalmanfilter.py         # Kalman Filter for data smoothing
# │   ├── cnnmodel.py             # CNN model creation and training
# │   ├── visualization.py        # Visualization of predictions
# │   ├── tfliteexporter.py       # Export model to TensorFlow Lite and .h file
# │   ├── config.py               # Configuration file for pipeline parameters
# │   ├── logger.py               # Logging setup
# ├── data/
# │   ├── normalpatterns.csv      # Gait data for normal patterns
# │   ├── parkinsonpatterns.csv   # Gait data for Parkinson's patterns
# ├── main.py                     # Entry point for the ML pipeline
# ├── README.txt                  # Project documentation
# ```
#
# @section usage_sec Usage
# To execute the pipeline:
# 1. Install dependencies: `pip install -r requirements.txt`
# 2. Record new gait pattern data or use existing CSV datasets located in the `data/` directory.
#    - Example files: `normalpatterns.csv`, `parkinsonpatterns.csv`.
# 3. Run the script: `python main.py`
#
# Outputs:
# - `gait_model.tflite`: TensorFlow Lite model for deployment.
# - `models/gait_model.h`: Header file for Arduino integration.
#
# @note The script logs all major steps and raises exceptions in case of failure.
## @file main.py
#  @brief The main entry point for executing the machine learning pipeline.
#  @details This script coordinates the entire pipeline, including data loading, preprocessing,
#  model training, exporting, and visualization.

import logging
from src.logger import configureLogging
from src.dataloader import loadData
from src.kalmanfilter import applyKalmanFilter
from src.cnnmodel import createCnnModel
from src.visualization import visualizePredictions
from src.tfliteexporter import saveAsHFile
import tensorflow as tf
import numpy as np


## @brief Executes the machine learning pipeline.
#  @details The pipeline includes:
#  - Loading datasets
#  - Preprocessing data using the Kalman Filter
#  - Training a Convolutional Neural Network (CNN)
#  - Exporting the trained model to TensorFlow Lite and .h file formats
#  - Visualizing model predictions
#  @param params Dictionary containing configuration parameters for the pipeline.
#  @exception Raises an exception if any step in the pipeline fails.
import os  # Import for directory path handling

def main(params):
    """Main function for executing the ML pipeline."""
    try:
        # Configure logging
        configureLogging()

        logging.info("Starting the pipeline.")

        # Load data
        logging.info("Loading data from 'data/normalpatterns.csv'")
        dataNormal = loadData('data/normalpatterns.csv')
        logging.info(f"Data loaded successfully with shape: {dataNormal.shape}")

        logging.info("Loading data from 'data/parkinsonpatterns.csv'")
        dataParkinson = loadData('data/parkinsonpatterns.csv')
        logging.info(f"Data loaded successfully with shape: {dataParkinson.shape}")

        # Apply Kalman Filter
        logging.info("Applying Kalman Filter to dataset.")
        columns = ['Step_length', 'Stride_length', 'Step_time']
        dataNormal = applyKalmanFilter(dataNormal, columns, params)
        dataParkinson = applyKalmanFilter(dataParkinson, columns, params)
        logging.info("Kalman Filter applied successfully to all columns.")

        # Prepare features and labels
        logging.info("Preparing features and labels for training.")
        featuresNormal = dataNormal[columns].values.reshape(-1, 3, 1)
        featuresParkinson = dataParkinson[columns].values.reshape(-1, 3, 1)
        features = np.concatenate((featuresNormal, featuresParkinson), axis=0)
        labels = np.array([0] * len(featuresNormal) + [1] * len(featuresParkinson))

        # Train the CNN model
        logging.info("Creating CNN model.")
        model = createCnnModel(params)
        logging.info("CNN model created successfully.")

        logging.info("Training the CNN model.")
        history = model.fit(
            features,
            labels,
            epochs=params["cnnEpochs"],
            batch_size=params["cnnBatchSize"],
            validation_split=params["cnnValidationSplit"]
        )
        logging.info("Model training completed successfully.")

        # Export the model
        logging.info("Converting the model to TensorFlow Lite format.")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tfliteModel = converter.convert()
        logging.info("Model converted to TensorFlow Lite format successfully.")

        # Ensure the models directory exists
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)

        # Save the TFLite model in the models directory
        tflite_path = os.path.join(models_dir, "gait_model.tflite")
        logging.info(f"Saving TFLite model to file '{tflite_path}'.")
        with open(tflite_path, 'wb') as f:
            f.write(tfliteModel)
        logging.info(f"TFLite model saved successfully as '{tflite_path}'.")

        # Save the TFLite model as a .h file in the models directory
        h_file_path = os.path.join(models_dir, "gait_model.h")
        logging.info(f"Saving TFLite model as a .h file: '{h_file_path}'.")
        saveAsHFile(tfliteModel, filename=h_file_path)
        logging.info(f"TFLite model saved successfully as '{h_file_path}'.")

        # Visualize predictions
        logging.info("Visualizing predictions.")
        visualizePredictions(model, features, labels)

        # Log final accuracy
        finalAccuracy = history.history['accuracy'][-1] * 100
        logging.info(f"Final Training Accuracy: {finalAccuracy:.2f}%")

        logging.info("Pipeline completed successfully.")

    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    ## @brief Default configuration parameters for the pipeline.
    from src.config import DEFAULT_PARAMETERS
    main(DEFAULT_PARAMETERS)