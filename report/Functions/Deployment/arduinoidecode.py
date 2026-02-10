/**
 * @file gait_classification.ino
 * @brief Deploy TensorFlow Lite Micro model on Arduino Nano 33 BLE Sense for real-time gait pattern classification.
 * @details This code reads IMU data, processes it, and uses a TensorFlow Lite Micro model to classify gait patterns.
 */

// Include TensorFlow Lite Micro library and IMU library
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include <Arduino_LSM9DS1.h>
#include "gait_model.h" // The TensorFlow Lite model in .h format

// TensorFlow Lite Micro configuration
constexpr int kTensorArenaSize = 2 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// TensorFlow Lite model variables
const tflite::Model* model = nullptr;
tflite::AllOpsResolver resolver;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Gait classification constants
const float STEP_THRESHOLD = 1.2;  // Adjust based on calibration
const int STEP_DELAY = 300;       // Minimum delay between steps in milliseconds
unsigned long lastStepTime = 0;
int stepCount = 0;

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  while (!Serial);

  // Initialize IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }
  Serial.println("IMU initialized successfully.");

  // Load TensorFlow Lite model
  model = tflite::GetModel(gait_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    while (1);
  }

  // Create TensorFlow Lite interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory for tensors
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Tensor allocation failed!");
    while (1);
  }

  // Get pointers to input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Setup complete. Starting classification...");
}

void loop() {
  float ax, ay, az;

  // Read acceleration data
  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(ax, ay, az);

    // Calculate total acceleration magnitude
    float acceleration = sqrt(ax * ax + ay * ay + az * az);

    // Step detection logic
    if (acceleration > STEP_THRESHOLD) {
      unsigned long currentTime = millis();
      if (currentTime - lastStepTime > STEP_DELAY) {
        lastStepTime = currentTime;
        stepCount++;

        // Prepare input tensor with acceleration data
        input->data.f[0] = ax;
        input->data.f[1] = ay;
        input->data.f[2] = az;

        // Run inference
        if (interpreter->Invoke() != kTfLiteOk) {
          Serial.println("Inference failed!");
          return;
        }

        // Get the output prediction
        float gait_prediction = output->data.f[0];
        if (gait_prediction > 0.5) {
          Serial.println("Parkinsonian gait detected!");
        } else {
          Serial.println("Normal gait detected!");
        }

        // Log step count
        Serial.print("Step Count: ");
        Serial.println(stepCount);
      }
    }
  }

  delay(20); // Adjust based on sampling rate
}