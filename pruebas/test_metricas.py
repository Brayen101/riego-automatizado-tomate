import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

from entrenar_modelo import evaluar_en_litros


class _ModeloFalso:
    """Reemplaza al modelo de Keras en estas pruebas: predict() devuelve
    siempre las predicciones que se le indiquen de antemano, para poder
    verificar el calculo de las metricas de forma aislada."""

    def __init__(self, predicciones):
        self._predicciones = predicciones

    def predict(self, X):
        return self._predicciones


def _scaler_litros(minimo, maximo):
    scaler = MinMaxScaler()
    scaler.fit(np.array([[minimo], [maximo]]))
    return scaler


def test_evaluar_en_litros_con_prediccion_perfecta_da_metricas_ideales():
    scaler_y = _scaler_litros(0.0, 10.0)
    y_real_litros = np.array([[0.0], [2.5], [5.0], [7.5], [10.0]])
    y_real_escalado = scaler_y.transform(y_real_litros)

    modelo = _ModeloFalso(predicciones=y_real_escalado)

    mae, rmse, r2 = evaluar_en_litros(modelo, y_real_escalado, y_real_escalado, 'prueba', scaler_y)

    assert mae == pytest.approx(0.0, abs=1e-6)
    assert rmse == pytest.approx(0.0, abs=1e-6)
    assert r2 == pytest.approx(1.0, abs=1e-6)


def test_evaluar_en_litros_con_caso_calculado_a_mano():
    # Rango 0-10 L: con este rango el escalado queda 1:1 con los litros reales
    scaler_y = _scaler_litros(0.0, 10.0)

    y_real_litros = np.array([[0.0], [10.0]])
    y_pred_litros = np.array([[2.0], [8.0]])

    y_real_escalado = scaler_y.transform(y_real_litros)
    y_pred_escalado = scaler_y.transform(y_pred_litros)

    modelo = _ModeloFalso(predicciones=y_pred_escalado)

    mae, rmse, r2 = evaluar_en_litros(modelo, y_real_escalado, y_real_escalado, 'prueba', scaler_y)

    # Error absoluto en ambos puntos: 2 L -> MAE = 2, RMSE = 2
    # Media real = 5; SS_tot = (0-5)^2 + (10-5)^2 = 50; SS_res = 2^2 + 2^2 = 8
    # R2 = 1 - 8/50 = 0.84
    assert mae == pytest.approx(2.0, abs=1e-6)
    assert rmse == pytest.approx(2.0, abs=1e-6)
    assert r2 == pytest.approx(0.84, abs=1e-6)
