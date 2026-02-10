## @file README.tex
#
# @brief Project Documentation for Parkinson Gait Pattern Recognition with an Arduino Nano 33 BLE
#
############################
# Authors:
#   Fedor Kotiaev
#   Abishek Senthilkumar
#
# Date Created: 10.01.2025
# Path: Project/Parkinson_Gait_Recognition/README.py
# Version: 1.5
# Reviewed by: 
# Review Date: 
############################

## @mainpage Parkinson Gait Pattern Recognition
#
#@section intro_sec Introduction
# This project implements a machine learning pipeline to classify Parkinson gait patterns, using the Arduino Nano 33 BLE Sense’s embedded IMU sensor.
#
# The pipeline includes:
#   - **Data Acquisition**: Collects and stores gait pattern data in CSV format (normalpatterns.csv, parkinsonpatterns.csv).
#   - **Data Preprocessing**: Applies Kalman filtering to smooth noisy IMU sensor data.
#   - **Machine Learning**: Uses a Convolutional Neural Network (CNN) model to process and classify gait patterns.
#   - **Model Deployment**: Converts the trained model into TensorFlow Lite and exports it as a `.h` file for microcontroller integration.

## @section features Key Features
# - **Data Preprocessing**:
#   - Smoothing using a Kalman Filter.
#   - Feature extraction from IMU sensor data (e.g., Step_length, Stride_length, Step_time).
# - **Machine Learning**:
#   - CNN model for binary classification (normal vs. Parkinson’s gait patterns).
#   - Metrics: Training accuracy and validation loss.
# - **Deployment**:
#   - TensorFlow Lite model export for integration with Arduino-based devices.
#   - `.h` file generation for microcontroller programming.

## @section structure Project Directory Structure
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

## @section usage Usage
# ### Setup
# Install the required Python dependencies:
# @code{.sh}
# pip install -r requirements.txt
# @endcode
#
# ### Run the Pipeline
# Execute the main pipeline script:
# @code{.sh}
# python main.py
# @endcode
#
# ### Output
# - **Model Training**:
#   - Training accuracy and validation loss displayed during each epoch.
# - **Exported Files**:
#   - `gait_model.tflite`: TensorFlow Lite model for deployment.
#   - `models/gait_model.h`: C header file for Arduino integration.

## @section modules Modules Description
# - `src/dataloader.py`: Handles CSV file loading.
# - `src/kalmanfilter.py`: Applies Kalman Filter for data smoothing.
# - `src/cnnmodel.py`: Defines and compiles the CNN model.
# - `src/visualization.py`: Visualizes predictions vs true labels.
# - `src/tfliteexporter.py`: Converts and exports the model for embedded systems.
# - `src/logger.py`: Configures project-wide logging.
# - `src/config.py`: Stores default pipeline parameters.

## @section dependencies Dependencies
# - **Python 3.7+**
# - **Required libraries**:
#   - tensorflow
#   - numpy
#   - pandas
#   - matplotlib
#   - pykalman


