import numpy as np
import pytest

from entrenar_modelo import dividir_datos, construir_modelo


def test_dividir_datos_da_aproximadamente_70_15_15():
    n = 1000
    X = np.random.default_rng(0).random((n, 4))
    y = np.random.default_rng(1).random((n, 1))

    X_train, X_val, X_test, y_train, y_val, y_test = dividir_datos(X, y)

    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_val) + len(X_test) == n

    assert len(X_train) / n == pytest.approx(0.70, abs=0.02)
    assert len(X_val) / n == pytest.approx(0.15, abs=0.02)
    assert len(X_test) / n == pytest.approx(0.15, abs=0.02)


def test_construir_modelo_tiene_entrada_4_y_salida_1():
    modelo = construir_modelo()

    assert modelo.input_shape == (None, 4)
    assert modelo.output_shape == (None, 1)
