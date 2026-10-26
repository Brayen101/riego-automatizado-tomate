import os
import sys

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest

# Permite importar los modulos de entrenamiento/ directamente (preprocesamiento,
# entrenar_modelo) sin tener que convertir esa carpeta en un paquete instalable.
RAIZ_PROYECTO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_ENTRENAMIENTO = os.path.join(RAIZ_PROYECTO, 'entrenamiento')
if RUTA_ENTRENAMIENTO not in sys.path:
    sys.path.insert(0, RUTA_ENTRENAMIENTO)


def generar_datos_simulados(n_filas=200, semilla=42):
    """Genera un DataFrame con la misma forma que datos_riego.csv, incluyendo
    filas con litros_aplicados = 0 (horas sin riego), tal como las registra
    la Fase 1 en campo."""
    rng = np.random.default_rng(semilla)

    timestamps = pd.date_range('2026-01-01', periods=n_filas, freq='h')
    humedad_suelo = rng.uniform(10, 90, n_filas)
    temperatura_ambiental = rng.uniform(5, 40, n_filas)
    humedad_ambiental = rng.uniform(20, 95, n_filas)
    dias_desde_trasplante = rng.integers(1, 100, n_filas)

    litros_aplicados = np.where(
        rng.random(n_filas) < 0.3,
        rng.uniform(0.5, 5.0, n_filas),
        0.0,
    )

    return pd.DataFrame({
        'timestamp': timestamps.astype(str),
        'humedad_suelo': humedad_suelo,
        'temperatura_ambiental': temperatura_ambiental,
        'humedad_ambiental': humedad_ambiental,
        'dias_desde_trasplante': dias_desde_trasplante,
        'litros_aplicados': litros_aplicados,
    })


@pytest.fixture
def datos_simulados():
    return generar_datos_simulados()


@pytest.fixture
def csv_simulado(tmp_path, datos_simulados):
    ruta = tmp_path / 'datos_riego_simulado.csv'
    datos_simulados.to_csv(ruta, index=False)
    return ruta


@pytest.fixture
def carpetas_temporales(tmp_path):
    datos_dir = tmp_path / 'datos'
    modelo_dir = tmp_path / 'modelo'
    datos_dir.mkdir()
    modelo_dir.mkdir()
    return str(datos_dir), str(modelo_dir)
