# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflitemodel = converter.convert()

# Save the TFLite model to a file
with open('gaitmodel.tflite', 'wb') as f:
    f.write(tflitemodel)

print("TFLite model saved as 'gaitmodel.tflite'")