# Project Directory Structure

	data/
  normalpatterns.csv        # Normal gait pattern dataset
  parkinsonpatterns.csv   # Parkinsonian gait pattern dataset
	models/
  gait_model.h             # TensorFlow Lite model in .h 
	src/
  dataloader.py                 # Loads and validates CSV data
  kalmanfilter.py               # Applies Kalman Filter for data smoothing
  cnnmodel.py                  # Defines and trains a CNN for classification
  tfliteexporter.py              # Converts and exports TensorFlow Lite models
  visualization.py              # Visualizes model performance
  logger.py                        # Configures logging utilities
  config.py                         # Parameter configuration for the pipeline
	main.py                     # Entry point for pipeline execution
