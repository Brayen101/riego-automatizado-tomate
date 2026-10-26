// Firmware de Fase 2: riego automatico mediante inferencia local con TensorFlow Lite Micro.
// Requiere la libreria Arduino "TensorFlowLite_ESP32" (o equivalente del puerto tflite-micro).
//
// ORDEN DE TRABAJO antes de cargar este archivo (ver Anexo E, punto 10, y README.md):
//   1. entrenamiento/preprocesamiento.py   -> genera ../modelo/parametros_normalizacion.json
//   2. entrenamiento/entrenar_modelo.py    -> genera ../modelo/modelo_mlp.h5
//   3. entrenamiento/convertir_tflite.py   -> genera ../modelo/modelo_mlp.tflite (cuantizado INT8)
//   4. entrenamiento/convertir_a_c_array.py -> genera modelo_datos.h en esta misma carpeta
//   5. Copiar a mano los valores de parametros_normalizacion.json dentro de
//      parametros_normalizacion.h (en esta carpeta)
//   6. Calcular RIEGO_MINIMO_LITROS = minimo valor de litros_aplicados > 0 en datos_riego.csv
//      (RF6: "igual o mayor que el riego mas pequeno aplicado por el agricultor")
//
// Mientras los pasos 1-6 no se hayan hecho con datos reales, este sketch compila pero
// modelo_datos.h y parametros_normalizacion.h solo traen valores de relleno.

#include <Arduino.h>
#include <DHT.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "modelo_datos.h"
#include "parametros_normalizacion.h"

// ===== PINES (Fase 2 - Tabla 3.2 "FASE 2" del Anexo E / Manual Tecnico) =====
#define PIN_HUMEDAD_SUELO 32
#define PIN_DHT22         4
#define PIN_CAUDALIMETRO  23
#define PIN_RELE          27
#define PIN_LED_VERDE     12
#define PIN_LED_AMARILLO  13
#define PIN_LED_ROJO      14
// La MicroSD no se usa en Fase 2 (Diagrama 2 / Anexo E punto 4)

#define TIPO_DHT DHT22
#define PULSOS_POR_LITRO 450  // recalibrar segun Anexo E punto 9

// Litros minimos para regar (RF6). Calcular con el dataset real:
// minimo de "litros_aplicados" entre las filas donde litros_aplicados > 0
#define RIEGO_MINIMO_LITROS 0.5f  // PLACEHOLDER: actualizar tras analizar datos_riego.csv

#define ESPERA_TRAS_RIEGO_MS      7200000UL  // 2 horas tras regar, para que el suelo absorba el agua (Alcance 3.4 / RF6)
#define INTERVALO_SIN_RIEGO_MS     900000UL  // 15 min si no se riega: NO esta fijado por el documento, ajustar segun pruebas de campo
#define INTERVALO_LECTURA_MS         2000UL  // minimo exigido por el DHT22
#define DURACION_MAXIMA_RIEGO_MS   600000UL  // resguardo de seguridad (no exigido por el documento, RNF6 no prevé redundancia):
                                              // cierra la valvula si en 10 min no se alcanzo el volumen estimado
                                              // (posible falla de caudalimetro o de la valvula)

DHT dht(PIN_DHT22, TIPO_DHT);

// ===== TensorFlow Lite Micro =====
constexpr int kTensorArenaSize = 8 * 1024;  // ajustar si AllocateTensors() falla
uint8_t tensorArena[kTensorArenaSize];

const tflite::Model* modelo = nullptr;
tflite::MicroInterpreter* interprete = nullptr;
TfLiteTensor* entradaModelo = nullptr;
TfLiteTensor* salidaModelo = nullptr;

// ===== Variables globales =====
volatile uint32_t contadorPulsos = 0;
int diasDesdeTrasplante = 1;
unsigned long ultimoDiaMs = 0;
unsigned long ultimaLecturaMs = 0;
unsigned long proximaEvaluacionMs = 0;

float humedadSuelo = 0, temperatura = 0, humedadAmbiental = 0;

void IRAM_ATTR contarPulso() {
  contadorPulsos++;
}

// Fallo irrecuperable: detiene el programa (requiere intervencion manual)
void indicarFalloYDetener(const char* mensaje) {
  digitalWrite(PIN_LED_VERDE, LOW);
  digitalWrite(PIN_LED_ROJO, HIGH);
  Serial.println(mensaje);
  while (true) delay(1000);
}

void leerSensores() {
  humedadSuelo = analogRead(PIN_HUMEDAD_SUELO) * (100.0 / 4095.0);
  float t = dht.readTemperature();
  float h = dht.readHumidity();
  if (!isnan(t) && !isnan(h)) {
    temperatura = t;
    humedadAmbiental = h;
    digitalWrite(PIN_LED_ROJO, LOW);
  } else {
    digitalWrite(PIN_LED_ROJO, HIGH);
    Serial.println("AVISO: lectura de DHT22 invalida, se reutiliza la ultima lectura valida");
  }
}

// Escala un valor real [minimo, maximo] al nivel INT8 cuantizado del tensor de entrada,
// usando el (scale, zero_point) propio de ese tensor en vez de valores fijos.
int8_t cuantizar(float valorReal, float minimo, float maximo, const TfLiteTensor* tensor) {
  float normalizado = (valorReal - minimo) / (maximo - minimo);
  normalizado = constrain(normalizado, 0.0f, 1.0f);
  int32_t q = (int32_t)roundf(normalizado / tensor->params.scale) + tensor->params.zero_point;
  return (int8_t)constrain(q, -128, 127);
}

float descuantizar(int8_t valor, float minimo, float maximo, const TfLiteTensor* tensor) {
  float normalizado = (valor - tensor->params.zero_point) * tensor->params.scale;
  normalizado = constrain(normalizado, 0.0f, 1.0f);
  return minimo + normalizado * (maximo - minimo);
}

float estimarLitros() {
  entradaModelo->data.int8[0] = cuantizar(humedadSuelo, NORM_HUMEDAD_SUELO_MIN, NORM_HUMEDAD_SUELO_MAX, entradaModelo);
  entradaModelo->data.int8[1] = cuantizar(temperatura, NORM_TEMPERATURA_MIN, NORM_TEMPERATURA_MAX, entradaModelo);
  entradaModelo->data.int8[2] = cuantizar(humedadAmbiental, NORM_HUMEDAD_AMBIENTAL_MIN, NORM_HUMEDAD_AMBIENTAL_MAX, entradaModelo);
  entradaModelo->data.int8[3] = cuantizar((float)diasDesdeTrasplante, NORM_DIAS_MIN, NORM_DIAS_MAX, entradaModelo);

  if (interprete->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: fallo la inferencia del modelo");
    return 0.0f;
  }

  return descuantizar(salidaModelo->data.int8[0], NORM_LITROS_MIN, NORM_LITROS_MAX, salidaModelo);
}

// Abre la electrovalvula, monitorea el caudalimetro y la cierra al alcanzar litrosEstimados
// (Fig. E.6, pasos 3-5). Es bloqueante a proposito: solo corre mientras el riego esta activo,
// con un tope de seguridad (DURACION_MAXIMA_RIEGO_MS) por si el caudalimetro o la valvula fallan.
void ejecutarRiego(float litrosEstimados) {
  Serial.print("Abriendo electrovalvula. Litros estimados: "); Serial.println(litrosEstimados);

  noInterrupts();
  contadorPulsos = 0;
  interrupts();

  digitalWrite(PIN_RELE, HIGH);  // electrovalvula normalmente cerrada -> se abre con el rele activo
  digitalWrite(PIN_LED_AMARILLO, HIGH);

  unsigned long inicioRiego = millis();
  float litrosAcumulados = 0.0f;

  while (true) {
    noInterrupts();
    uint32_t pulsos = contadorPulsos;
    interrupts();
    litrosAcumulados = pulsos / (float)PULSOS_POR_LITRO;

    if (litrosAcumulados >= litrosEstimados) {
      Serial.println("Volumen estimado alcanzado, cerrando electrovalvula.");
      break;
    }
    if (millis() - inicioRiego >= DURACION_MAXIMA_RIEGO_MS) {
      digitalWrite(PIN_LED_ROJO, HIGH);
      Serial.println("AVISO: tiempo maximo de riego alcanzado sin completar el volumen estimado.");
      break;
    }
    delay(100);
  }

  digitalWrite(PIN_RELE, LOW);
  digitalWrite(PIN_LED_AMARILLO, LOW);
  Serial.print("Riego finalizado. Litros aplicados: "); Serial.println(litrosAcumulados);
}

void setup() {
  Serial.begin(115200);

  pinMode(PIN_RELE, OUTPUT);
  pinMode(PIN_LED_VERDE, OUTPUT);
  pinMode(PIN_LED_AMARILLO, OUTPUT);
  pinMode(PIN_LED_ROJO, OUTPUT);
  pinMode(PIN_CAUDALIMETRO, INPUT_PULLUP);
  digitalWrite(PIN_RELE, LOW);

  dht.begin();
  attachInterrupt(digitalPinToInterrupt(PIN_CAUDALIMETRO), contarPulso, RISING);

  modelo = tflite::GetModel(modelo_mlp_tflite);
  if (modelo->version() != TFLITE_SCHEMA_VERSION) {
    indicarFalloYDetener("ERROR: version de esquema del modelo .tflite incompatible con esta libreria");
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter interpreteEstatico(modelo, resolver, tensorArena, kTensorArenaSize);
  interprete = &interpreteEstatico;

  if (interprete->AllocateTensors() != kTfLiteOk) {
    indicarFalloYDetener("ERROR: memoria insuficiente para el modelo (aumentar kTensorArenaSize)");
  }

  entradaModelo = interprete->input(0);
  salidaModelo = interprete->output(0);

  ultimoDiaMs = millis();
  leerSensores();
  proximaEvaluacionMs = millis() + INTERVALO_LECTURA_MS;

  digitalWrite(PIN_LED_VERDE, HIGH);
  Serial.println("Sistema iniciado correctamente - Fase 2 (riego automatico)");
}

void loop() {
  unsigned long ahora = millis();

  if (ahora - ultimoDiaMs >= 86400000UL) {
    diasDesdeTrasplante++;
    ultimoDiaMs = ahora;
  }

  if (ahora - ultimaLecturaMs >= INTERVALO_LECTURA_MS) {
    ultimaLecturaMs = ahora;
    leerSensores();
  }

  if (ahora >= proximaEvaluacionMs) {
    float litrosEstimados = estimarLitros();
    Serial.print("Litros estimados por el modelo: "); Serial.println(litrosEstimados);

    if (litrosEstimados >= RIEGO_MINIMO_LITROS) {
      ejecutarRiego(litrosEstimados);
      proximaEvaluacionMs = millis() + ESPERA_TRAS_RIEGO_MS;
    } else {
      Serial.println("Estimacion por debajo del riego minimo, no se riega.");
      proximaEvaluacionMs = millis() + INTERVALO_SIN_RIEGO_MS;
    }
  }
}
