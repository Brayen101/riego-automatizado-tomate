import tensorflow as tf
import numpy as np

# Cargar modelo entrenado
modelo = tf.keras.models.load_model('../modelo/modelo_mlp.h5')

# Convertir a TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(modelo)
modelo_tflite = converter.convert()

# Guardar archivo .tflite
with open('../modelo/modelo_mlp.tflite', 'wb') as f:
    f.write(modelo_tflite)

print("=== MODELO CONVERTIDO A TFLITE ===")
print(f"Tamaño del archivo: {len(modelo_tflite)} bytes")