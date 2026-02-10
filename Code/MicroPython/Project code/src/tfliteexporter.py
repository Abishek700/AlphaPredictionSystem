## @file tfliteexporter.py
#  @brief Contains functions to export a TensorFlow Lite model as a .h file.

import logging

## @brief Saves a TensorFlow Lite model as a C header file for microcontroller integration.
#  @param tfliteModel Byte representation of the TensorFlow Lite model.
#  @param filename Name of the output .h file (default is "model.h").
#  @exception Raises an exception if the file cannot be saved.
def saveAsHFile(tfliteModel, filename="model.h"):
    """Save a TFLite model as a .h file."""
    try:
        logging.info(f"Saving TFLite model as a .h file: {filename}")
        with open(filename, "w") as f:
            f.write('#include <stdint.h>\n\n')
            f.write(f'const unsigned char modelTflite[] = {{\n')
            f.write(",\n".join(f"0x{b:02x}" for b in tfliteModel))
            f.write('\n};\n')
            f.write(f'const int modelTfliteLen = {len(tfliteModel)};\n')
        logging.info(f"TFLite model saved as {filename}")
    except Exception as e:
        logging.error(f"Error saving model as .h file: {e}")
        raise