import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATOS_DIR_DEFECTO = os.path.join(BASE_DIR, '..', 'datos')
MODELO_DIR_DEFECTO = os.path.join(BASE_DIR, '..', 'modelo')


def cargar_datos(ruta):
    df = pd.read_csv(ruta)

    print("=== DATOS CARGADOS ===")
    print(f"Total de registros: {len(df)}")
    print(df.head())
    print(df.describe())

    print("\n=== VALORES NULOS ===")
    print(df.isnull().sum())

    return df


def limpiar(df):
    return df.dropna()


def escalar(df):
    X = df[['humedad_suelo', 'temperatura_ambiental', 'humedad_ambiental', 'dias_desde_trasplante']]
    y = df['litros_aplicados']

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    return X_scaled, y_scaled, scaler_X, scaler_y


def guardar_parametros(scaler_X, scaler_y, modelo_dir):
    """Guarda los escaladores y los min/max en texto plano, para que el
    firmware de la Fase 2 normalice y desnormalice con los MISMOS valores
    (el ESP32 no puede leer objetos pickle/joblib de scikit-learn)."""
    joblib.dump(scaler_X, os.path.join(modelo_dir, 'scaler_X.save'))
    joblib.dump(scaler_y, os.path.join(modelo_dir, 'scaler_y.save'))

    parametros_normalizacion = {
        'humedad_suelo':          {'min': float(scaler_X.data_min_[0]), 'max': float(scaler_X.data_max_[0])},
        'temperatura_ambiental':  {'min': float(scaler_X.data_min_[1]), 'max': float(scaler_X.data_max_[1])},
        'humedad_ambiental':      {'min': float(scaler_X.data_min_[2]), 'max': float(scaler_X.data_max_[2])},
        'dias_desde_trasplante':  {'min': float(scaler_X.data_min_[3]), 'max': float(scaler_X.data_max_[3])},
        'litros_aplicados':       {'min': float(scaler_y.data_min_[0]), 'max': float(scaler_y.data_max_[0])},
    }

    ruta_json = os.path.join(modelo_dir, 'parametros_normalizacion.json')
    with open(ruta_json, 'w') as f:
        json.dump(parametros_normalizacion, f, indent=2)

    return parametros_normalizacion


def main(datos_dir=DATOS_DIR_DEFECTO, modelo_dir=MODELO_DIR_DEFECTO):
    ruta_csv = os.path.join(datos_dir, 'datos_riego.csv')

    df = cargar_datos(ruta_csv)
    df = limpiar(df)
    X_scaled, y_scaled, scaler_X, scaler_y = escalar(df)

    np.save(os.path.join(datos_dir, 'X_scaled.npy'), X_scaled)
    np.save(os.path.join(datos_dir, 'y_scaled.npy'), y_scaled)

    guardar_parametros(scaler_X, scaler_y, modelo_dir)

    print("\n=== PREPROCESAMIENTO COMPLETO ===")
    print(f"Registros listos para entrenamiento: {len(X_scaled)}")
    print(f"Escaladores guardados en {os.path.join(modelo_dir, 'scaler_X.save')} y {os.path.join(modelo_dir, 'scaler_y.save')}")
    print(f"Parametros de normalizacion guardados en {os.path.join(modelo_dir, 'parametros_normalizacion.json')}")

    return X_scaled, y_scaled


if __name__ == "__main__":
    main()
