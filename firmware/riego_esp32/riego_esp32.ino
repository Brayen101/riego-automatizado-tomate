#include <Arduino.h>
#include <DHT.h>
#include <SD.h>
#include <SPI.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ===== PINES =====
#define PIN_HUMEDAD_SUELO 32
#define PIN_DHT22         4
#define PIN_CAUDALIMETRO  23
#define PIN_RELE          27
#define PIN_LED_VERDE     12
#define PIN_LED_AMARILLO  13
#define PIN_LED_ROJO      14
#define PIN_SD_CS         5

// ===== CONFIGURACION =====
#define TIPO_DHT DHT22
#define PULSOS_POR_LITRO 450

DHT dht(PIN_DHT22, TIPO_DHT);

// ===== VARIABLES GLOBALES =====
volatile int contadorPulsos = 0;
float litrosAplicados = 0.0;
int diasDesdeTrasplante = 1;
unsigned long ultimoDia = 0;

// ===== INTERRUPCION CAUDALIMETRO =====
void IRAM_ATTR contarPulso() {
  contadorPulsos++;
}

void setup() {
  Serial.begin(115200);

  // Pines
  pinMode(PIN_RELE, OUTPUT);
  pinMode(PIN_LED_VERDE, OUTPUT);
  pinMode(PIN_LED_AMARILLO, OUTPUT);
  pinMode(PIN_LED_ROJO, OUTPUT);
  pinMode(PIN_CAUDALIMETRO, INPUT_PULLUP);

  digitalWrite(PIN_RELE, LOW);

  // Caudalimetro
  attachInterrupt(digitalPinToInterrupt(PIN_CAUDALIMETRO), contarPulso, RISING);

  // DHT22
  dht.begin();

  // MicroSD
  if (!SD.begin(PIN_SD_CS)) {
    digitalWrite(PIN_LED_ROJO, HIGH);
    Serial.println("ERROR: MicroSD no detectada");
    return;
  }

  digitalWrite(PIN_LED_VERDE, HIGH);
  ultimoDia = millis();
  Serial.println("Sistema iniciado correctamente");
}

void loop() {
  // Contar dias desde trasplante
  if (millis() - ultimoDia >= 86400000) {
    diasDesdeTrasplante++;
    ultimoDia = millis();
  }

  // Leer sensores
  float humedadSuelo     = analogRead(PIN_HUMEDAD_SUELO) * (100.0 / 4095.0);
  float temperatura      = dht.readTemperature();
  float humedadAmbiental = dht.readHumidity();

  // Verificar sensores
  if (isnan(temperatura) || isnan(humedadAmbiental)) {
    digitalWrite(PIN_LED_ROJO, HIGH);
    Serial.println("ERROR: Sensor DHT22 desconectado");
    return;
  }

  // Guardar en microSD (Fase 1)
  File archivo = SD.open("/datos.csv", FILE_APPEND);
  if (archivo) {
    archivo.print(humedadSuelo);    archivo.print(",");
    archivo.print(temperatura);     archivo.print(",");
    archivo.print(humedadAmbiental);archivo.print(",");
    archivo.print(diasDesdeTrasplante); archivo.print(",");
    archivo.println(litrosAplicados);
    archivo.close();
  }

  Serial.print("Humedad suelo: ");    Serial.println(humedadSuelo);
  Serial.print("Temperatura: ");      Serial.println(temperatura);
  Serial.print("Humedad ambiental: ");Serial.println(humedadAmbiental);
  Serial.print("Dias trasplante: ");  Serial.println(diasDesdeTrasplante);

  delay(60000); // Leer cada minuto
}