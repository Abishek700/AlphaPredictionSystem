"""
@file tflite_converter.py
@brief Converts a TensorFlow Keras model to TensorFlow Lite (TFLite) format and saves it to a file.
@details This script uses TensorFlow's TFLiteConverter to convert a trained Keras model into a lightweight format suitable for deployment on edge devices like microcontrollers.
@libraries
    - tensorflow: Used for deep learning model development and TFLite conversion.
"""

import tensorflow as tf

def convert_to_tflite(model, output_file='gait_model.tflite'):
    """
    @brief Converts a TensorFlow Keras model to TFLite format and saves it.
    @param model Trained TensorFlow Keras model to convert.
    @param output_file Name of the file to save the converted TFLite model.
    """
    try:
        # Convert the model to TensorFlow Lite format
        print("Converting the model to TensorFlow Lite format...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        # Save the TFLite model to a file
        with open(output_file, 'wb') as f:
            f.write(tflite_model)

        print(f"TFLite model saved as '{output_file}'")
    except Exception as e:
        print(f"Error during TFLite conversion: {e}")
        raise