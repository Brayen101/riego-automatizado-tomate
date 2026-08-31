import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
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

print("\n=== PREPROCESAMIENTO COMPLETO ===")
print(f"Registros listos para entrenamiento: {len(X_scaled)}")