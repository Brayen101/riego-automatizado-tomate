import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import json
import os

# Cargar datos
df = pd.read_csv('../datos/datos_riego.csv')

print("=== DATOS CARGADOS ===")
print(f"Total de registros: {len(df)}")
print(df.head())
print(df.describe())

# Verificar valores nulos
print("\n=== VALORES NULOS ===")
print(df.isnull().sum())

# Eliminar filas con valores nulos
df = df.dropna()

# Variables de entrada (X) y variable objetivo (y)
X = df[['humedad_suelo', 'temperatura_ambiental', 'humedad_ambiental', 'dias_desde_trasplante']]
y = df['litros_aplicados']

# Normalizar entre 0 y 1
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Guardar datos preprocesados
np.save('../datos/X_scaled.npy', X_scaled)
np.save('../datos/y_scaled.npy', y_scaled)

# Guardar los escaladores (Anexo E, Paso 3: el firmware de la Fase 2 debe normalizar
# y desnormalizar con los MISMOS parametros usados aqui)
joblib.dump(scaler_X, '../modelo/scaler_X.save')
joblib.dump(scaler_y, '../modelo/scaler_y.save')

# Exportar los min/max en texto plano para copiarlos como constantes en el firmware del ESP32
# (el ESP32 no puede leer objetos pickle/joblib de scikit-learn)
parametros_normalizacion = {
    'humedad_suelo':          {'min': float(scaler_X.data_min_[0]), 'max': float(scaler_X.data_max_[0])},
    'temperatura_ambiental':  {'min': float(scaler_X.data_min_[1]), 'max': float(scaler_X.data_max_[1])},
    'humedad_ambiental':      {'min': float(scaler_X.data_min_[2]), 'max': float(scaler_X.data_max_[2])},
    'dias_desde_trasplante':  {'min': float(scaler_X.data_min_[3]), 'max': float(scaler_X.data_max_[3])},
    'litros_aplicados':       {'min': float(scaler_y.data_min_[0]), 'max': float(scaler_y.data_max_[0])},
}
with open('../modelo/parametros_normalizacion.json', 'w') as f:
    json.dump(parametros_normalizacion, f, indent=2)

print("\n=== PREPROCESAMIENTO COMPLETO ===")
print(f"Registros listos para entrenamiento: {len(X_scaled)}")
print("Escaladores guardados en ../modelo/scaler_X.save y ../modelo/scaler_y.save")
print("Parametros de normalizacion guardados en ../modelo/parametros_normalizacion.json")