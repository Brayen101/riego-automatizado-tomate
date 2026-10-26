"""
Convierte modelo_mlp.tflite en un arreglo de C (modelo_datos.h) para incluirlo en el
firmware de Fase 2 del ESP32. Alternativa portable a `xxd -i`, que no viene instalado
por defecto en Windows.

Ejecutar despues de convertir_tflite.py:
    python convertir_a_c_array.py
"""

RUTA_TFLITE = '../modelo/modelo_mlp.tflite'
RUTA_SALIDA = '../firmware/riego_esp32_fase2/modelo_datos.h'

with open(RUTA_TFLITE, 'rb') as f:
    datos = f.read()

if len(datos) == 0:
    raise SystemExit(
        "ERROR: modelo_mlp.tflite esta vacio. "
        "Ejecuta primero entrenar_modelo.py y convertir_tflite.py con datos reales."
    )

with open(RUTA_SALIDA, 'w') as f:
    f.write('#ifndef MODELO_DATOS_H\n#define MODELO_DATOS_H\n\n')
    f.write(f'// Generado automaticamente desde {RUTA_TFLITE} ({len(datos)} bytes)\n')
    f.write('alignas(8) const unsigned char modelo_mlp_tflite[] = {\n')
    for i in range(0, len(datos), 12):
        fila = ', '.join(f'0x{b:02x}' for b in datos[i:i + 12])
        f.write(f'  {fila},\n')
    f.write('};\n')
    f.write(f'const unsigned int modelo_mlp_tflite_len = {len(datos)};\n\n')
    f.write('#endif\n')

print(f"=== modelo_datos.h generado ({len(datos)} bytes) ===")
print(f"Guardado en: {RUTA_SALIDA}")
