#include <Arduino_LSM9DS1.h> // Library for IMU sensor on Arduino Nano 33 BLE Sense
#include <math.h> // Math library for sqrt function

/// @brief Threshold for detecting steps.
/// @details Adjust this value based on testing and calibration to detect steps accurately.
const float STEP_THRESHOLD = 1.2;

/**
 * @brief Setup function initializes the IMU and serial communication.
 *
 * @details This function initializes the IMU sensor and the serial communication. If the IMU
 *          initialization fails, the program halts and logs an error to the serial monitor.
 */
void setup() {
    Serial.begin(9600); // 
    while (!Serial);    // Wait for the serial port to initialize

    // Initialize IMU
    if (!IMU.begin()) {
        Serial.println("Error: Failed to initialize IMU!"); // Log error
        while (1); // Halt program if IMU initialization fails
    }
    Serial.println("IMU initialized successfully."); // Log success
}

/**
 * @brief Loop function continuously reads IMU data and performs step detection.
 *
 * @details The function reads acceleration values from the IMU sensor, validates the data,
 *          and calculates acceleration magnitude. If the magnitude exceeds the step threshold,
 *          a step is detected and logged to the serial monitor.
 */
void loop() {
    float ax, ay, az; // Variables to store acceleration in X, Y, Z axes

    // Check if acceleration data is available
    if (IMU.accelerationAvailable()) {
        IMU.readAcceleration(ax, ay, az); // Read acceleration values

        // Validate IMU data
        if (isnan(ax) || isnan(ay) || isnan(az)) {
            Serial.println("Error: Invalid IMU data detected."); // Log error
            return; // Skip further processing
        }

        // Calculate acceleration magnitude
        float acceleration = sqrt(ax * ax + ay * ay + az * az);

        // Step detection logic
        if (acceleration > STEP_THRESHOLD) {
            // Log step detection
            Serial.println("Step detected.");
        }
    } else {
        Serial.println("Error: IMU data not available."); // Log error if no data
    }
    delay(20); 
}