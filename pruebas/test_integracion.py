import os

import preprocesamiento
import entrenar_modelo


def test_flujo_completo_genera_todos_los_archivos(datos_simulados, carpetas_temporales):
    datos_dir, modelo_dir = carpetas_temporales

    ruta_csv = os.path.join(datos_dir, 'datos_riego.csv')
    datos_simulados.to_csv(ruta_csv, index=False)

    preprocesamiento.main(datos_dir=datos_dir, modelo_dir=modelo_dir)
    entrenar_modelo.main(datos_dir=datos_dir, modelo_dir=modelo_dir, epocas=3)

    archivos_esperados = [
        'modelo_mlp.h5',
        'scaler_X.save',
        'scaler_y.save',
        'parametros_normalizacion.json',
        'metricas_evaluacion.txt',
    ]

    for nombre_archivo in archivos_esperados:
        ruta = os.path.join(modelo_dir, nombre_archivo)
        assert os.path.exists(ruta), f"Falta el archivo {nombre_archivo}"
        assert os.path.getsize(ruta) > 0
