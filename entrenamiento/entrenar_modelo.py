import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

# Cargar datos preprocesados
X = np.load('../datos/X_scaled.npy')
y = np.load('../datos/y_scaled.npy')

# Dividir en entrenamiento (70%), validación (15%) y prueba (15%)
# (el articulo original exige evaluar sobre un conjunto de prueba aparte del de validacion)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Construcción del modelo MLP (4-16-8-1, topologia recomendada como punto de partida
# en la Discusion del articulo original)
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


def evaluar_en_litros(X_eval, y_eval, nombre, scaler_y):
    """Calcula MAE, RMSE y R2 en litros reales (desnormalizados), tal como
    exige el protocolo de evaluacion del articulo original (ecuaciones 3-5)."""
    y_pred_escalado = modelo.predict(X_eval)
    y_pred_litros = scaler_y.inverse_transform(y_pred_escalado)
    y_real_litros = scaler_y.inverse_transform(y_eval)

    mae = np.mean(np.abs(y_real_litros - y_pred_litros))
    rmse = np.sqrt(np.mean((y_real_litros - y_pred_litros) ** 2))
    ss_res = np.sum((y_real_litros - y_pred_litros) ** 2)
    ss_tot = np.sum((y_real_litros - np.mean(y_real_litros)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    print(f"\n=== METRICAS EN {nombre.upper()} (litros) ===")
    print(f"MAE:  {mae:.3f} L")
    print(f"RMSE: {rmse:.3f} L")
    print(f"R2:   {r2:.4f}")
    return mae, rmse, r2


# Evaluación sobre validación y prueba, en litros reales
scaler_y = joblib.load('../modelo/scaler_y.save')
metricas_val = evaluar_en_litros(X_val, y_val, 'validacion', scaler_y)
metricas_test = evaluar_en_litros(X_test, y_test, 'prueba', scaler_y)

with open('../modelo/metricas_evaluacion.txt', 'w') as f:
    f.write("Metricas de evaluacion del modelo MLP (en litros)\n\n")
    f.write(f"Validacion -> MAE: {metricas_val[0]:.3f} L | RMSE: {metricas_val[1]:.3f} L | R2: {metricas_val[2]:.4f}\n")
    f.write(f"Prueba     -> MAE: {metricas_test[0]:.3f} L | RMSE: {metricas_test[1]:.3f} L | R2: {metricas_test[2]:.4f}\n")

print("\nMetricas guardadas en ../modelo/metricas_evaluacion.txt")

# Graficar pérdida
plt.plot(historial.history['loss'], label='entrenamiento')
plt.plot(historial.history['val_loss'], label='validación')
plt.title('Pérdida del modelo')
plt.xlabel('Épocas')
plt.ylabel('MSE')
plt.legend()
plt.savefig('../modelo/grafica_perdida.png')
plt.show()
