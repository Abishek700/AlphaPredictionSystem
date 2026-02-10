
"""
@section Libraries Used
- Arduino_LSM9DS1: Library for interfacing with the IMU on Arduino Nano 33 BLE Sense.
- math.h: Provides mathematical functions like `isnan` for validation.
"""

#include <Arduino_LSM9DS1.h> // Library for IMU sensor on Arduino Nano 33 BLE Sense
#include <math.h> // Math library for validation functions like isnan

"""
@brief Function to log messages to the serial monitor.

@param message A constant character pointer to the message to log.
"""
void logMessage(const char* message) {
    Serial.println(message);
}

"""
@brief Setup function initializes the IMU and serial communication.

@details
This function sets up the IMU sensor and starts serial communication.
If IMU initialization fails, it halts the program and logs an error message.
"""
void setup() {
    Serial.begin(9600); // Start serial communication at 9600 baud rate
    while (!Serial);    // Wait for the serial port to initialize

    logMessage("Initializing IMU...");

    // Initialize IMU
    if (!IMU.begin()) {
        logMessage("Error: IMU initialization failed!"); // Log error
        while (1); // Halt program if IMU initialization fails
    }
    logMessage("IMU initialized successfully."); // Log success
}

"""
@brief Loop function continuously reads IMU data and logs the processing status.

@details
The function reads acceleration values from the IMU sensor, validates the data,
and logs messages indicating success or errors. It also logs the availability of IMU data.
"""
void loop() {
    float ax, ay, az; // Variables to store acceleration in X, Y, Z axes

    // Check if acceleration data is available
    if (IMU.accelerationAvailable()) {
        IMU.readAcceleration(ax, ay, az); // Read acceleration values

        // Validate IMU data
        if (isnan(ax) || isnan(ay) || isnan(az)) {
            logMessage("Error: Invalid IMU data detected."); // Log error
            return; // Skip further processing
        }

        logMessage("IMU data processed successfully."); // Log success
    } else {
        logMessage("Error: IMU data not available."); // Log error if no data
    }
    delay(20); // Adjust delay for desired sampling rate (e.g., 50 Hz)
}