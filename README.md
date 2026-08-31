🌱 Sistema Inteligente de Riego Automatizado — Tomate Tipo Pera

**Universidad Adventista de Bolivia — Carrera de Ingeniería de Sistemas**
**Autor:** Josue Brayen Rojas Vasquez
**Tutor:** Ximena Wendy Calizaya Perez
**Año:** 2026

---

¿De qué trata?

Sistema que automatiza el riego del cultivo de tomate tipo pera en la
carpa solar del municipio de Incahuasi, Bolivia (2.970 m.s.n.m.),
usando una Red Neuronal Artificial ejecutada localmente en un ESP32
sin necesidad de internet.

---

¿Cómo funciona?

El ESP32 recolecta datos del cultivo durante 80–100 días, se entrena
un modelo MLP en Python con esos datos, y el modelo vuelve al ESP32
para que riegue de forma autónoma abriendo y cerrando la electroválvula
según la cantidad de agua estimada.

---

Hardware

- Microcontrolador ESP32
- Sensor capacitivo de humedad del suelo
- Sensor DHT22 (temperatura y humedad ambiental)
- Caudalímetro YF-S201
- Electroválvula de solenoide
- Módulo relé

---

Software

- Firmware ESP32: C/C++ con Arduino IDE
- Entrenamiento: Python, TensorFlow, Keras
- Modelo embebido: TensorFlow Lite (INT8)

---

#Archivos principales

- `riego_esp32.ino` — corre en el ESP32, lee sensores y guarda datos
- `preprocesamiento.py` — limpia y normaliza los datos del CSV
- `entrenar_modelo.py` — entrena la red neuronal MLP
- `convertir_tflite.py` — convierte el modelo para el ESP32

---

Costo estimado del prototipo

Entre **Bs. 465 y Bs. 660** usando componentes disponibles en Bolivia.
¿Cómo se ejecuta cada archivo?

Paso 1 — Cargar el firmware en el ESP32
Abrir `riego_esp32.ino` en Arduino IDE y subirlo al ESP32.
El sistema empieza a leer sensores y guardar datos en la microSD.

Paso 2 — Preprocesar los datos
Cuando terminen los 80–100 días, extraer la microSD y copiar
`datos_riego.csv` a la carpeta `datos/`. Luego ejecutar:

```bash
python entrenamiento/preprocesamiento.py
```

 Paso 3 — Entrenar el modelo
```bash
python entrenamiento/entrenar_modelo.py
```

Paso 4 — Convertir el modelo para el ESP32
```bash
python entrenamiento/convertir_tflite.py
```

Paso 5 — Cargar el modelo en el ESP32
Copiar `modelo_mlp.tflite` a la carpeta del firmware y
volver a subir `riego_esp32.ino` al ESP32.
El sistema ya riega de forma autónoma.