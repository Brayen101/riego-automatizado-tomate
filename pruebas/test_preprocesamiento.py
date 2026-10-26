import json
import os

import numpy as np
import pandas as pd
import pytest

from preprocesamiento import cargar_datos, limpiar, escalar, guardar_parametros


def test_limpiar_elimina_filas_con_nulos(datos_simulados):
    df = datos_simulados.copy()
    df.loc[0, 'humedad_suelo'] = np.nan
    df.loc[5, 'litros_aplicados'] = np.nan

    total_antes = len(df)
    df_limpio = limpiar(df)

    assert len(df_limpio) == total_antes - 2
    assert df_limpio.isnull().sum().sum() == 0


def test_escalar_deja_valores_entre_0_y_1(datos_simulados):
    X_scaled, y_scaled, scaler_X, scaler_y = escalar(datos_simulados)

    assert X_scaled.min() >= 0.0
    assert X_scaled.max() <= 1.0
    assert y_scaled.min() >= 0.0
    assert y_scaled.max() <= 1.0


def test_guardar_parametros_contiene_las_5_variables_con_min_max_correctos(datos_simulados, carpetas_temporales):
    _, modelo_dir = carpetas_temporales
    _, _, scaler_X, scaler_y = escalar(datos_simulados)

    parametros = guardar_parametros(scaler_X, scaler_y, modelo_dir)

    variables_esperadas = {
        'humedad_suelo', 'temperatura_ambiental', 'humedad_ambiental',
        'dias_desde_trasplante', 'litros_aplicados',
    }
    assert set(parametros.keys()) == variables_esperadas
    for variable in variables_esperadas:
        assert 'min' in parametros[variable]
        assert 'max' in parametros[variable]

    ruta_json = os.path.join(modelo_dir, 'parametros_normalizacion.json')
    assert os.path.exists(ruta_json)

    with open(ruta_json) as f:
        guardado = json.load(f)

    assert guardado['humedad_suelo']['min'] == pytest.approx(datos_simulados['humedad_suelo'].min())
    assert guardado['humedad_suelo']['max'] == pytest.approx(datos_simulados['humedad_suelo'].max())
    assert guardado['litros_aplicados']['min'] == pytest.approx(datos_simulados['litros_aplicados'].min())
    assert guardado['litros_aplicados']['max'] == pytest.approx(datos_simulados['litros_aplicados'].max())

    assert os.path.exists(os.path.join(modelo_dir, 'scaler_X.save'))
    assert os.path.exists(os.path.join(modelo_dir, 'scaler_y.save'))


def test_cargar_datos_falla_con_csv_vacio(tmp_path):
    ruta = tmp_path / 'vacio.csv'
    ruta.write_text('')

    with pytest.raises(pd.errors.EmptyDataError):
        cargar_datos(str(ruta))


def test_escalar_falla_si_falta_una_columna(datos_simulados):
    df_incompleto = datos_simulados.drop(columns=['litros_aplicados'])

    with pytest.raises(KeyError):
        escalar(df_incompleto)
