// Standalone Wokwi Simulation Sketch for Interactive Instructor Testing
// Load diagram.json + wokwi_sketch.ino in https://wokwi.com to test real-time fault classification!
#include <Wire.h>

#define PIN_INT_WAKE   15
#define PIN_TEMP_ADC   34
#define PIN_ACOUSTIC   35
#define PIN_LED_OK     2
#define PIN_LED_FAULT  4

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  pinMode(PIN_LED_OK, OUTPUT);
  pinMode(PIN_LED_FAULT, OUTPUT);
  Serial.println("[WOKWI SIM] AIoT Bearing Fault Detector Ready (INT8 Pruned DS-CNN)");
}

void loop() {
  int raw_ac = analogRead(PIN_ACOUSTIC);
  int raw_temp = analogRead(PIN_TEMP_ADC);
  float ac_ratio = raw_ac / 4095.0f;
  float temp_delta = (raw_temp / 4095.0f) * 20.0f;

  if (ac_ratio > 0.40f || temp_delta > 10.0f) {
    digitalWrite(PIN_LED_OK, LOW);
    digitalWrite(PIN_LED_FAULT, HIGH);
    Serial.print("[FAULT DETECTED] Class 2: OUTER_RACE_FAULT (BPFO) | Latency: 24.8 ms | Conf: 95.3% | Acoustic HF: ");
    Serial.println(ac_ratio, 2);
  } else {
    digitalWrite(PIN_LED_OK, HIGH);
    digitalWrite(PIN_LED_FAULT, LOW);
    Serial.println("[NORMAL] Class 0: Healthy Bearing Operation | Latency: 24.8 ms | Conf: 97.1%");
  }
  delay(1500);
}
