import os
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATOS_DIR_DEFECTO = os.path.join(BASE_DIR, '..', 'datos')
MODELO_DIR_DEFECTO = os.path.join(BASE_DIR, '..', 'modelo')


def cargar_datos_entrenamiento(datos_dir):
    X = np.load(os.path.join(datos_dir, 'X_scaled.npy'))
    y = np.load(os.path.join(datos_dir, 'y_scaled.npy'))
    return X, y


def dividir_datos(X, y, semilla=42):
    """Divide en entrenamiento (70%), validacion (15%) y prueba (15%)
    (el articulo original exige evaluar sobre un conjunto de prueba aparte del de validacion)."""
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=semilla)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=semilla)
    return X_train, X_val, X_test, y_train, y_val, y_test


def construir_modelo():
    """Modelo MLP 4-16-8-1, topologia recomendada como punto de partida
    en la discusion del articulo original."""
    modelo = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(4,)),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1, activation='linear')
    ])
    modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return modelo


def entrenar(modelo, X_train, y_train, X_val, y_val, epocas=100):
    return modelo.fit(
        X_train, y_train,
        epochs=epocas,
        validation_data=(X_val, y_val),
        verbose=1
    )


def evaluar_en_litros(modelo, X_eval, y_eval, nombre, scaler_y):
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


def guardar_metricas(modelo_dir, metricas_val, metricas_test):
    ruta = os.path.join(modelo_dir, 'metricas_evaluacion.txt')
    with open(ruta, 'w') as f:
        f.write("Metricas de evaluacion del modelo MLP (en litros)\n\n")
        f.write(f"Validacion -> MAE: {metricas_val[0]:.3f} L | RMSE: {metricas_val[1]:.3f} L | R2: {metricas_val[2]:.4f}\n")
        f.write(f"Prueba     -> MAE: {metricas_test[0]:.3f} L | RMSE: {metricas_test[1]:.3f} L | R2: {metricas_test[2]:.4f}\n")
    return ruta


def graficar_perdida(historial, ruta_salida):
    plt.figure()
    plt.plot(historial.history['loss'], label='entrenamiento')
    plt.plot(historial.history['val_loss'], label='validación')
    plt.title('Pérdida del modelo')
    plt.xlabel('Épocas')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig(ruta_salida)
    plt.show()
    return ruta_salida


def main(datos_dir=DATOS_DIR_DEFECTO, modelo_dir=MODELO_DIR_DEFECTO, epocas=100):
    X, y = cargar_datos_entrenamiento(datos_dir)
    X_train, X_val, X_test, y_train, y_val, y_test = dividir_datos(X, y)

    modelo = construir_modelo()
    historial = entrenar(modelo, X_train, y_train, X_val, y_val, epocas=epocas)

    modelo.save(os.path.join(modelo_dir, 'modelo_mlp.h5'))
    print("\n=== MODELO GUARDADO ===")

    scaler_y = joblib.load(os.path.join(modelo_dir, 'scaler_y.save'))
    metricas_val = evaluar_en_litros(modelo, X_val, y_val, 'validacion', scaler_y)
    metricas_test = evaluar_en_litros(modelo, X_test, y_test, 'prueba', scaler_y)

    ruta_metricas = guardar_metricas(modelo_dir, metricas_val, metricas_test)
    print(f"\nMetricas guardadas en {ruta_metricas}")

    graficar_perdida(historial, os.path.join(modelo_dir, 'grafica_perdida.png'))

    return modelo, metricas_val, metricas_test


if __name__ == "__main__":
    main()
