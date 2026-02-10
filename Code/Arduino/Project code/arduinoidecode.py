############################
# Creator of the File: Fedor Kotiaev and Abishek Senthilkumar
# Date created: 10.01.2025
# Path: Code/Arduino/ProjectCode
# Version: 1.5
# Reviewed by:
# Review Date:
############################
#
# @mainpage Parkinson Gait Pattern Classification on Arduino Nano 33 BLE Sense
# @section intro_sec Introduction
# This script is designed for classifying gait patterns using the Arduino Nano 33 BLE Sense. It collects real-time acceleration data from an onboard IMU sensor and classifies gait patterns using a TensorFlow Lite Micro model.
#
# @section overview_sec Overview
# The program integrates the following steps:
# - **IMU Initialization**: Captures acceleration data from the X, Y, and Z axes.
# - **Step Detection**: Applies a threshold-based algorithm to filter noise and detect valid steps.
# - **Machine Learning Inference**: Executes a TensorFlow Lite Micro model to classify gait patterns.
# - **Visual Feedback**: Provides feedback using the onboard RGB LED.
#
# @section usage_sec Usage
# 1. Install necessary libraries:
#    - TensorFlow Lite Micro
#    - Arduino_LSM9DS1
# 2. Upload the program to the Arduino Nano 33 BLE Sense.
# 3. Observe classification results via the serial monitor and the RGB LED.
#
# @section output_sec Outputs
# - **Step Count**: Total steps detected during runtime.
# - **Gait Classification**: Logs whether the detected gait pattern is normal or Parkinsonian.
# - **Visual Feedback**: RGB LED feedback—red for normal gait, blue for Parkinsonian gait.
#
# @note Ensure the TensorFlow Lite Micro model (`gait_model.h`) is included in the project directory.

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include <Arduino_LSM9DS1.h>
#include "gait_model.h"

/// @namespace arduinoidecode
/// @brief Namespace for gait classification using the Arduino Nano 33 BLE Sense.
namespace arduinoidecode {

/// @brief Pre-allocated memory buffer for TensorFlow Lite Micro.
/// @details
/// - **Purpose**: Stores intermediate tensors, weights, and input/output data during inference.
/// - **Size**: 2048 bytes.
constexpr int kTensorArenaSize = 2 * 1024;

/// @brief TensorFlow Lite Micro tensor arena.
/// @details Holds the memory buffer allocated for TensorFlow Lite computations.
uint8_t tensor_arena[kTensorArenaSize];

/// @brief Pointer to the TensorFlow Lite model.
/// @details Loaded using `tflite::GetModel()` and validated for schema compatibility.
const tflite::Model* model = nullptr;

/// @brief Resolves TensorFlow Lite operators.
/// @details Registers all operators required for model execution, such as convolution and activation functions.
tflite::AllOpsResolver resolver;

/// @brief TensorFlow Lite Micro interpreter instance.
/// @details Manages memory allocation and executes inference.
tflite::MicroInterpreter* interpreter = nullptr;

/// @brief Input tensor for the TensorFlow Lite model.
/// @details Stores acceleration data (X, Y, Z) before inference.
TfLiteTensor* input = nullptr;

/// @brief Output tensor for the TensorFlow Lite model.
/// @details Holds the classification result (probability of Parkinsonian gait).
TfLiteTensor* output = nullptr;

/// @brief Threshold for step detection.
/// @details Filters out noise by ignoring accelerations below this value.
const float STEP_THRESHOLD = 1.2;

/// @brief Minimum time between consecutive step detections.
/// @details Prevents false positives caused by rapid motion.
const int STEP_DELAY = 300;

/// @brief Timestamp of the last detected step.
/// @details Used with `STEP_DELAY` to enforce a minimum delay between steps.
unsigned long lastStepTime = 0;

/// @brief Total number of detected steps.
/// @details Incremented each time a valid step is detected.
int stepCount = 0;

/// @brief Calculated total acceleration magnitude.
/// @details Represents the intensity of motion, computed as the Euclidean norm of X, Y, Z acceleration.
float acceleration = 0;

/// @brief Classification result from the TensorFlow Lite model.
/// @details Probability that the detected gait is Parkinsonian.
float gait_prediction = 0;

/// @brief Red LED pin for normal gait indication.
#define LEDR 22

/// @brief Blue LED pin for Parkinsonian gait indication.
#define LEDB 24

} // namespace arduinoidecode

using namespace arduinoidecode;

/**
 * @brief Sets up the Arduino Nano 33 BLE Sense for gait classification.
 * @details
 * - Initializes the IMU sensor.
 * - Configures RGB LED pins.
 * - Loads the TensorFlow Lite model and initializes the interpreter.
 * - Allocates memory for input and output tensors.
 */
void setup() {
  Serial.begin(9600);
  while (!Serial);

  // Configure RGB LED pins.
  pinMode(LEDR, OUTPUT);
  pinMode(LEDB, OUTPUT);

  // Initialize the IMU sensor.
  if (!IMU.begin()) {
    Serial.println("IMU initialization failed!");
    while (1);
  }
  Serial.println("IMU initialized successfully.");

  // Load the TensorFlow Lite model.
  model = tflite::GetModel(gait_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    while (1);
  }

  // Create and initialize the TensorFlow Lite interpreter.
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Tensor allocation failed!");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Setup complete. Starting classification...");
}

/**
 * @brief Main loop for real-time gait classification.
 * @details
 * - Captures acceleration data from the IMU sensor.
 * - Detects steps based on acceleration magnitude and delay logic.
 * - Performs inference using the TensorFlow Lite model.
 * - Logs results and provides RGB LED feedback.
 */
void loop() {
  float ax, ay, az; ///< Acceleration data along X, Y, Z axes.

  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(ax, ay, az);

    acceleration = sqrt(ax * ax + ay * ay + az * az); ///< Compute total acceleration.

    if (acceleration > STEP_THRESHOLD) {
      unsigned long currentTime = millis();

      if (currentTime - lastStepTime > STEP_DELAY) {
        lastStepTime = currentTime;
        stepCount++;

        input->data.f[0] = ax;
        input->data.f[1] = ay;
        input->data.f[2] = az;

        if (interpreter->Invoke() != kTfLiteOk) {
          Serial.println("Inference failed!");
          return;
        }

        gait_prediction = output->data.f[0];

        if (gait_prediction > 0.5) {
          Serial.println("Parkinsonian gait detected!");
          digitalWrite(LEDR, LOW);
          digitalWrite(LEDB, HIGH);
        } else {
          Serial.println("Normal gait detected!");
          digitalWrite(LEDR, HIGH);
          digitalWrite(LEDB, LOW);
        }

        Serial.print("Step Count: ");
        Serial.println(stepCount);
      }
    }
  }

  delay(20);
}