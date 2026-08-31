import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Cargar datos preprocesados
X = np.load('../datos/X_scaled.npy')
y = np.load('../datos/y_scaled.npy')

# Dividir en entrenamiento y validación (80% - 20%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Construcción del modelo MLP
modelo = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])

# Compilar
modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Entrenar
historial = modelo.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_val, y_val),
    verbose=1
)

# Guardar modelo
modelo.save('../modelo/modelo_mlp.h5')
print("\n=== MODELO GUARDADO ===")

# Graficar pérdida
plt.plot(historial.history['loss'], label='entrenamiento')
plt.plot(historial.history['val_loss'], label='validación')
plt.title('Pérdida del modelo')
plt.xlabel('Épocas')
plt.ylabel('MSE')
plt.legend()
plt.savefig('../modelo/grafica_perdida.png')
plt.show()