# AIoT Prototype Repository: On-Device Multi-Modal Bearing Fault Detector

This repository contains the complete source code, deterministic dataset generator, training and optimization pipeline, quantized TensorFlow Lite (`.tflite`) model artifacts, embedded C++ microcontroller firmware, Wokwi hardware simulation files, and the written project report (`BEARING-FAULT-DETECTOR-REPORT.pdf`).

---

## 1. Repository Structure

```text
BEARING-FAULT-DETECTOR-REPO/
├── BEARING-FAULT-DETECTOR-REPORT.pdf        # Final written project report (PDF)
├── README.md                                # Setup, wiring, and reproducibility guide
├── data/
│   ├── bearing_sensor_dataset.csv           # 1,000 labeled multi-modal sensor windows (seed=42)
│   └── pareto_benchmark_results.csv         # Exact 5-stage Pareto frontier evaluation metrics
├── models/
│   ├── bearing_fault_fp32.tflite            # Baseline Float32 DS-CNN model artifact (148.6 KB)
│   └── bearing_fault_pruned50_int8.tflite   # Deployed 50% Pruned + Full INT8 PTQ artifact (26.4 KB)
├── training/
│   ├── generate_synthetic_dataset.py        # Reproduces data/bearing_sensor_dataset.csv (seed=42)
│   ├── train_and_optimize.py                # Model training, PolynomialDecay pruning & INT8 PTQ
│   ├── evaluate_pareto.py                   # Standalone reproducibility script for instructor grading
│   └── requirements.txt                     # Python package dependencies
├── firmware/
│   ├── platformio.ini                       # PlatformIO build configuration for nRF52840
│   └── main_firmware/
│       ├── main_firmware.ino                # Main embedded TinyML inference & BLE alerting sketch
│       ├── sensing_pipeline.h               # Multi-modal DMA/FIFO sensing pipeline header
│       ├── sensing_pipeline.cpp             # Windowed FFT DSP & INT8 affine input quantization
│       ├── model_data.h                     # C byte-array declaration for TFLite Micro
│       └── model_data.cpp                   # 16-byte aligned INT8 quantized model buffer (26.4 KB)
└── simulation/
    ├── diagram.json                         # Wokwi circuit diagram (MCU + IMU + Temp + Acoustic + LEDs)
    └── wokwi_sketch.ino                     # Interactive Wokwi simulation firmware
```

---

## 2. Hardware Specification & Exact Wiring Diagram

### 2.1 Primary Physical Hardware
* **Microcontroller Board:** **Arduino Nano 33 BLE Sense Rev2**
  * **SoC:** Nordic Semiconductor **nRF52840** (64 MHz 32-bit ARM Cortex-M4F with FPU and CMSIS-NN SIMD instructions)
  * **Memory:** 256 KB SRAM, 1 MB Internal Flash
  * **On-Board Sensors Used:**
    * **BMI270 / LSM9DS1** 3-Axis Accelerometer (configured to $\pm 16	ext{g}$, $1,600	ext{ Hz}$ ODR, 32-sample hardware FIFO watermark + wake-on-motion interrupt at $>0.35	ext{g}$)
    * **MP34DT05-A** Digital MEMS PDM Microphone ($16,000	ext{ Hz}$ via hardware EasyDMA double-buffering)
  * **External High-Precision Contact Temperature Probe:** **Texas Instruments TMP117** ($\pm 0.1^\circ	ext{C}$ accuracy, I2C address `0x48`) mounted to the bearing outer housing.

### 2.2 Physical Pin-to-Pin Wiring Table

| Sensor / Component | Sensor Pin | Arduino Nano 33 BLE Sense Rev2 Pin | Bus / Signal Description |
| :--- | :--- | :--- | :--- |
| **On-Board IMU (BMI270)** | `SDA1` / `SCL1` | Internal I2C1 (`P0.14` / `P0.15`) | 400 kHz Fast-Mode I2C FIFO Burst Read |
| **On-Board IMU Interrupt** | `INT1` | `D2` (`P1.11` / Internal IRQ) | Hardware Wake-on-Motion Interrupt ($>0.35	ext{g}$) |
| **On-Board PDM Mic (MP34DT05)** | `CLK` / `DOUT` | Internal PDM (`P0.26` / `P0.25`) | 16 kHz EasyDMA Double-Buffered Audio Stream |
| **External TMP117 Probe** | `VIN` | `3.3V` | Regulated 3.3V Supply |
| **External TMP117 Probe** | `GND` | `GND` | Common Ground |
| **External TMP117 Probe** | `SDA` | `A4` (`P0.31` / External I2C SDA) | I2C Data Line (with $4.7	ext{ k}\Omega$ pull-up) |
| **External TMP117 Probe** | `SCL` | `A5` (`P0.02` / External I2C SCL) | I2C Clock Line (with $4.7	ext{ k}\Omega$ pull-up) |
| **External TMP117 Probe** | `ADD0` | `GND` | Sets I2C slave address to `0x48` |

---

## 3. Wokwi Simulation Setup (Alternative Virtual Verification)

For instant browser-based testing without physical hardware:
1. Open [https://wokwi.com](https://wokwi.com) and start a new ESP32/Arduino project.
2. Upload `simulation/diagram.json` into the `diagram.json` tab.
3. Copy the contents of `simulation/wokwi_sketch.ino` into the main code editor.
4. Click **Start Simulation**. Adjust the **Acoustic HF Envelope** potentiometer and **NTC Temperature Sensor** slider to observe real-time transition between `Class 0: Healthy Bearing Operation` (Green LED) and `Class 2: OUTER_RACE_FAULT (BPFO)` (Red LED + Serial telemetry).

---

## 4. Reproducing the Results and Numbers in the Report

Follow these steps to reproduce the dataset, optimization pipeline, accuracy metrics, model compression ratios, and Pareto frontier table reported in `BEARING-FAULT-DETECTOR-REPORT.pdf`:

### Step 1: Install Python Dependencies
```bash
cd training
pip install -r requirements.txt
```

### Step 2: Regenerate the Deterministic Multi-Modal Dataset (`seed=42`)
```bash
python3 generate_synthetic_dataset.py
```
*Output:* Generates `../data/bearing_sensor_dataset.csv` containing 1,000 samples (250 per class across `Normal`, `Inner_Race_Fault`, `Outer_Race_Fault`, and `Ball_Defect`).

### Step 3: Run the Pareto Frontier & INT8 Quantization Verification Script
```bash
python3 evaluate_pareto.py
```
*Expected Output:*
* Evaluates the 80/20 stratified train/test split (`random_state=42`).
* Applies the exact INT8 affine quantization parameters ($S = 0.03125$, $Z = -12$).
* Prints per-class Precision, Recall, F1-Score, Confusion Matrix, verified binary `.tflite` artifact sizes (`148.6 KB` FP32 vs. `26.4 KB` Pruned+INT8 = **5.63x reduction**), and the complete 5-stage Pareto Frontier comparison table:

| Stage | Precision | Sparsity | Test Acc (%) | Macro F1 | Flash Size (KB) | Peak RAM Arena (KB) | Latency (ms) | Energy / Inf (mJ) | Pareto Optimal |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline Float32 DS-CNN** | FP32 | 0% | 96.4% | 0.963 | 148.6 KB | 94.2 KB | 112.4 ms | 1.65 mJ | No |
| **2. Dynamic Range Quantization** | INT8 W / FP32 A | 0% | 96.1% | 0.960 | 44.8 KB | 78.5 KB | 68.2 ms | 1.00 mJ | Yes |
| **3. Pruned (30%) + Full INT8 PTQ** | Full INT8 | 30% | 95.8% | 0.957 | 34.2 KB | 41.0 KB | 34.6 ms | 0.51 mJ | Yes |
| **4. Pruned (50%) + Full INT8 PTQ [DEPLOYED]** | Full INT8 | 50% | **95.3%** | **0.952** | **26.4 KB** | **36.8 KB** | **24.8 ms** | **0.36 mJ** | **Yes (Optimum)** |
| **5. Pruned (75%) + Full INT8 PTQ [Over-Pruned]** | Full INT8 | 75% | 86.2% | 0.854 | 18.1 KB | 32.4 KB | 19.5 ms | 0.29 mJ | No |

### Step 4: Compile and Flash the Embedded Firmware
* **Using Arduino IDE:**
  1. Open `firmware/main_firmware/main_firmware.ino`.
  2. Install the **Arduino Mbed OS Nano Boards** core via Board Manager.
  3. Select **Arduino Nano 33 BLE** and click **Upload**.
  4. Open the Serial Monitor at **115200 baud** to inspect real-time inference latency (`~24.8 ms`), class probabilities, and conditional 12-byte BLE 5.0 GATT anomaly alerts.
* **Using PlatformIO CLI:**
  ```bash
  cd firmware
  pio run -t upload
  pio device monitor -b 115200
  ```

---

## 5. Mandatory Declaration of AI Tool Usage

In accordance with the course policy on academic integrity, the use of AI tools is declared below:
* **AI Tool Used:** Google Gemini
* **Model / Version Used:** Gemini (2026 release)
* **Purpose of Use:** Assisting with debugging TensorFlow Lite for Microcontrollers (TFLM) C++ tensor arena alignment (`alignas(16)`), verifying CMSIS-NN INT8 quantization scale and zero-point arithmetic, and organizing repository scripts and documentation for grading reproducibility.
