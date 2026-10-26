#include <Arduino.h>
#include <DHT.h>
#include <SD.h>
#include <SPI.h>
#include <time.h>

// ===== PINES (Fase 1 - Tabla 3.1 "FASE 1" del Anexo E / Manual Tecnico) =====
#define PIN_HUMEDAD_SUELO 32
#define PIN_DHT22         4
#define PIN_CAUDALIMETRO  23
#define PIN_LED_VERDE     12
#define PIN_LED_AMARILLO  13
#define PIN_LED_ROJO      14
#define PIN_SD_CS         5
#define PIN_SD_CLK        18
#define PIN_SD_MISO       19
#define PIN_SD_MOSI       26  // reasignado: el MOSI por defecto del ESP32 es GPIO23, que ya usa el caudalimetro

// ===== CONFIGURACION =====
#define TIPO_DHT DHT22
#define PULSOS_POR_LITRO 450          // valor nominal YF-S201; recalibrar segun Anexo E punto 9
#define INTERVALO_SIN_RIEGO_MS  3600000UL  // 1 hora sin riego -> registrar fila con 0 litros (RF3 / Fig. E.5)
#define TIMEOUT_FIN_RIEGO_MS    5000UL     // sin pulsos nuevos durante este tiempo = el riego termino
#define INTERVALO_LECTURA_MS    2000UL     // el DHT22 exige minimo 2 s entre lecturas

// Fecha y hora del trasplante (dia en que se enciende el sistema por primera vez).
// Actualizar este valor antes de cargar el firmware y despues de cualquier corte de energia
// (el ESP32 no tiene RTC con bateria; el reloj interno se reconfigura por software, Anexo E punto 6).
#define FECHA_TRASPLANTE "2026 10 12 08 00 00"  // "AAAA MM DD HH MM SS"

DHT dht(PIN_DHT22, TIPO_DHT);

// ===== VARIABLES GLOBALES =====
volatile uint32_t contadorPulsos = 0;
volatile unsigned long ultimoPulsoMs = 0;

int diasDesdeTrasplante = 1;
unsigned long ultimoDiaMs = 0;
unsigned long ultimaLecturaMs = 0;
unsigned long ultimoRegistroMs = 0;

float humedadSuelo = 0, temperatura = 0, humedadAmbiental = 0;
float humedadSueloPreRiego = 0, temperaturaPreRiego = 0, humedadAmbientalPreRiego = 0;
bool regando = false;

// ===== INTERRUPCION CAUDALIMETRO =====
void IRAM_ATTR contarPulso() {
  contadorPulsos++;
  ultimoPulsoMs = millis();
}

// Configura el reloj interno del ESP32 a partir de FECHA_TRASPLANTE (sin modulo RTC externo)
void configurarRelojInterno() {
  struct tm tmTrasplante = {0};
  int anio, mes, dia, hora, minuto, segundo;
  sscanf(FECHA_TRASPLANTE, "%d %d %d %d %d %d", &anio, &mes, &dia, &hora, &minuto, &segundo);
  tmTrasplante.tm_year = anio - 1900;
  tmTrasplante.tm_mon  = mes - 1;
  tmTrasplante.tm_mday = dia;
  tmTrasplante.tm_hour = hora;
  tmTrasplante.tm_min  = minuto;
  tmTrasplante.tm_sec  = segundo;

  time_t epoch = mktime(&tmTrasplante);
  struct timeval tv = { epoch, 0 };
  settimeofday(&tv, NULL);
}

String obtenerTimestamp() {
  time_t ahora;
  struct tm infoTiempo;
  time(&ahora);
  localtime_r(&ahora, &infoTiempo);
  char buffer[20];
  strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &infoTiempo);
  return String(buffer);
}

// Fallo irrecuperable (p. ej. MicroSD no detectada): detiene el programa, como exige la Fig. E.5
void indicarFalloYDetener(const char* mensaje) {
  digitalWrite(PIN_LED_VERDE, LOW);
  digitalWrite(PIN_LED_ROJO, HIGH);
  Serial.println(mensaje);
  while (true) {
    delay(1000); // requiere intervencion manual
  }
}

// Escribe una fila en datos.csv: timestamp, 3 lecturas, dias, litros (esquema de la Fig. E.4)
void registrarFila(float hSuelo, float temp, float hAmb, float litros) {
  File archivo = SD.open("/datos.csv", FILE_APPEND);
  if (!archivo) {
    digitalWrite(PIN_LED_ROJO, HIGH);
    Serial.println("ERROR: no se pudo abrir datos.csv");
    return;
  }
  archivo.print(obtenerTimestamp());  archivo.print(",");
  archivo.print(hSuelo);              archivo.print(",");
  archivo.print(temp);                archivo.print(",");
  archivo.print(hAmb);                archivo.print(",");
  archivo.print(diasDesdeTrasplante); archivo.print(",");
  archivo.println(litros, 3);
  archivo.close();
}

// Lee humedad de suelo y DHT22. Si el DHT22 falla, conserva la ultima lectura valida
// en vez de interrumpir el ciclo (evita el bucle sin pausa de la version anterior).
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

void setup() {
  Serial.begin(115200);

  pinMode(PIN_LED_VERDE, OUTPUT);
  pinMode(PIN_LED_AMARILLO, OUTPUT);
  pinMode(PIN_LED_ROJO, OUTPUT);
  pinMode(PIN_CAUDALIMETRO, INPUT_PULLUP);

  configurarRelojInterno();
  dht.begin();
  attachInterrupt(digitalPinToInterrupt(PIN_CAUDALIMETRO), contarPulso, RISING);

  // Pines SPI remapeados: el MOSI por defecto (GPIO23) choca con el caudalimetro
  SPI.begin(PIN_SD_CLK, PIN_SD_MISO, PIN_SD_MOSI, PIN_SD_CS);
  if (!SD.begin(PIN_SD_CS)) {
    indicarFalloYDetener("ERROR: MicroSD no detectada");
  }

  if (!SD.exists("/datos.csv")) {
    File archivo = SD.open("/datos.csv", FILE_WRITE);
    if (archivo) {
      archivo.println("timestamp,humedad_suelo,temperatura_ambiental,humedad_ambiental,dias_desde_trasplante,litros_aplicados");
      archivo.close();
    }
  }

  ultimoDiaMs = millis();
  ultimoRegistroMs = millis();
  leerSensores();
  humedadSueloPreRiego = humedadSuelo;
  temperaturaPreRiego = temperatura;
  humedadAmbientalPreRiego = humedadAmbiental;

  digitalWrite(PIN_LED_VERDE, HIGH);
  Serial.println("Sistema iniciado correctamente - Fase 1 (recoleccion de datos)");
}

void loop() {
  unsigned long ahora = millis();

  // Contar dias desde el trasplante (resta con unsigned long: segura ante el desborde de millis() a ~49 dias)
  if (ahora - ultimoDiaMs >= 86400000UL) {
    diasDesdeTrasplante++;
    ultimoDiaMs = ahora;
  }

  // Lectura periodica de sensores (no bloqueante)
  if (ahora - ultimaLecturaMs >= INTERVALO_LECTURA_MS) {
    ultimaLecturaMs = ahora;
    leerSensores();
  }

  // Inicio de un riego: el caudalimetro empieza a generar pulsos
  if (!regando && contadorPulsos > 0) {
    regando = true;
    humedadSueloPreRiego = humedadSuelo;
    temperaturaPreRiego = temperatura;
    humedadAmbientalPreRiego = humedadAmbiental;
    digitalWrite(PIN_LED_AMARILLO, HIGH);
    Serial.println("Riego manual detectado, acumulando litros...");
  }

  // Fin de un riego: no llegan pulsos nuevos desde hace TIMEOUT_FIN_RIEGO_MS
  if (regando && (ahora - ultimoPulsoMs >= TIMEOUT_FIN_RIEGO_MS)) {
    noInterrupts();
    uint32_t pulsos = contadorPulsos;
    contadorPulsos = 0;
    interrupts();

    float litros = pulsos / (float)PULSOS_POR_LITRO;
    registrarFila(humedadSueloPreRiego, temperaturaPreRiego, humedadAmbientalPreRiego, litros);
    Serial.print("Riego registrado: "); Serial.print(litros); Serial.println(" L");

    digitalWrite(PIN_LED_AMARILLO, LOW);
    regando = false;
    ultimoRegistroMs = ahora;
  }

  // Sin riego durante una hora: registrar fila con litros = 0 (para que el modelo aprenda ambos casos)
  if (!regando && (ahora - ultimoRegistroMs >= INTERVALO_SIN_RIEGO_MS)) {
    registrarFila(humedadSuelo, temperatura, humedadAmbiental, 0.0);
    ultimoRegistroMs = ahora;
  }
}
