#ifndef PARAMETROS_NORMALIZACION_H
#define PARAMETROS_NORMALIZACION_H

// Copiar aqui los valores de ../../modelo/parametros_normalizacion.json
// (ese archivo lo genera entrenamiento/preprocesamiento.py a partir de datos_riego.csv).
// Los valores de abajo son PLACEHOLDERS: no vienen de datos reales todavia.

#define NORM_HUMEDAD_SUELO_MIN        0.0f
#define NORM_HUMEDAD_SUELO_MAX      100.0f

#define NORM_TEMPERATURA_MIN           0.0f
#define NORM_TEMPERATURA_MAX          45.0f

#define NORM_HUMEDAD_AMBIENTAL_MIN     0.0f
#define NORM_HUMEDAD_AMBIENTAL_MAX   100.0f

#define NORM_DIAS_MIN                  1.0f
#define NORM_DIAS_MAX                100.0f

#define NORM_LITROS_MIN                0.0f
#define NORM_LITROS_MAX               10.0f

#endif
