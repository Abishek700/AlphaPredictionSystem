// Include TensorFlow Lite Micro header files
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "gait_model.h" // The TFLite model converted to .h format

// Allocate memory for TensorFlow Lite Micro
constexpr int kTensorArenaSize = 2 * 1024; // Adjust based on available memory
uint8_t tensor_arena[kTensorArenaSize];

// Initialize the TensorFlow Lite Micro components
const tflite::Model* model = tflite::GetModel(gait_model);
tflite::AllOpsResolver resolver;
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);

// Function to perform inference
void setup() {
    Serial.begin(9600);
    while (!Serial);

    // Allocate memory for tensors
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        Serial.println("Tensor allocation failed!");
        while (1);
    }

    Serial.println("TensorFlow Lite Micro setup complete.");
}