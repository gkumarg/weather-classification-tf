# convert keras model to tensforflow lite model
import tensorflow as tf


# Load the TensorFlow model
model = tf.keras.models.load_model('MobileNetv2.keras')
print(model.summary())

#convert keras model to tensforflow lite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('MobileNetv2.tflite', 'wb') as f:
  f.write(tflite_model)
