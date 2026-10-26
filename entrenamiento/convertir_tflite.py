import tensorflow as tf
import numpy as np

# Cargar modelo entrenado
modelo = tf.keras.models.load_model('../modelo/modelo_mlp.h5')

# Dataset representativo para la cuantizacion INT8 (debe tener la misma distribucion
# que las entradas que recibira el modelo: los datos ya normalizados)
X_scaled = np.load('../datos/X_scaled.npy').astype(np.float32)

def representative_dataset():
    muestras = min(200, len(X_scaled))
    for i in range(muestras):
        yield [X_scaled[i:i + 1]]

# Convertir a TensorFlow Lite con cuantizacion INT8 completa
# (Tabla "Fases de desarrollo", Fase 4, y Anexo E Paso 7: el modelo debe quedar
# cuantizado para reducir su huella de memoria en el ESP32)
converter = tf.lite.TFLiteConverter.from_keras_model(modelo)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

modelo_tflite = converter.convert()

# Guardar archivo .tflite
with open('../modelo/modelo_mlp.tflite', 'wb') as f:
    f.write(modelo_tflite)

print("=== MODELO CONVERTIDO A TFLITE (INT8) ===")
print(f"Tamaño del archivo: {len(modelo_tflite)} bytes")
