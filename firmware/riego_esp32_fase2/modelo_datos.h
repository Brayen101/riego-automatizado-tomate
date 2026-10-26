#ifndef MODELO_DATOS_H
#define MODELO_DATOS_H

// Generar este archivo con: python entrenamiento/convertir_a_c_array.py
// (ejecutar despues de entrenamiento/convertir_tflite.py, una vez que modelo_mlp.tflite
// contenga el modelo real entrenado con datos de la carpa solar).
//
// El arreglo de abajo es un PLACEHOLDER VACIO: compila, pero NO es un modelo valido.
// tflite::GetModel() fallara hasta que este archivo se regenere con el .tflite real.

alignas(8) const unsigned char modelo_mlp_tflite[] = {
  0x00
};
const unsigned int modelo_mlp_tflite_len = 1;

#endif
