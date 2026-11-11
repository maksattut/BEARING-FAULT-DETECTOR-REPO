#!/usr/bin/env python3
"""
Deterministic Synthetic Dataset Generator for AIoT Bearing Fault Detection.
Generates 1,000 labeled multi-modal feature windows (250 per class) combining:
- High-frequency triaxial vibration features (BMI270 / LSM9DS1 IMU @ 1600 Hz)
- Acoustic ultrasonic/audible envelope features (MP34DT05 PDM Mic @ 16 kHz)
- Thermal drift features (TMP117 / On-die temperature sensor @ 0.2 Hz)
"""
import os
import numpy as np
import pandas as pd

def generate_bearing_dataset(output_path="../data/bearing_sensor_dataset.csv", n_per_class=250, seed=42):
    np.random.seed(seed)
    classes = ["Normal", "Inner_Race_Fault", "Outer_Race_Fault", "Ball_Defect"]
    rows = []
    for label_idx, label_name in enumerate(classes):
        for i in range(n_per_class):
            if label_idx == 0:
                rms_vib = np.random.normal(0.24, 0.11)
                kurtosis_vib = np.random.normal(3.10, 0.85)
                crest_factor = np.random.normal(3.25, 0.75)
                bpfo_energy = np.random.normal(0.20, 0.14)
                bpfi_energy = np.random.normal(0.13, 0.14)
                bsf_energy = np.random.normal(0.12, 0.14)
                acoustic_db = np.random.normal(51.0, 5.5)
                acoustic_hf_ratio = np.random.normal(0.22, 0.12)
                temp_delta_c = np.random.normal(5.0, 2.5)
            elif label_idx == 1:
                rms_vib = np.random.normal(0.58, 0.15)
                kurtosis_vib = np.random.normal(6.10, 0.85)
                crest_factor = np.random.normal(5.40, 0.95)
                bpfo_energy = np.random.normal(0.36, 0.12)
                bpfi_energy = np.random.normal(0.74, 0.13)
                bsf_energy = np.random.normal(0.38, 0.12)
                acoustic_db = np.random.normal(64.5, 5.5)
                acoustic_hf_ratio = np.random.normal(0.54, 0.15)
                temp_delta_c = np.random.normal(13.5, 3.2)
            elif label_idx == 2:
                rms_vib = np.random.normal(0.51, 0.15)
                kurtosis_vib = np.random.normal(5.10, 0.80)
                crest_factor = np.random.normal(4.85, 0.90)
                bpfo_energy = np.random.normal(0.76, 0.13)
                bpfi_energy = np.random.normal(0.34, 0.12)
                bsf_energy = np.random.normal(0.33, 0.12)
                acoustic_db = np.random.normal(61.0, 5.5)
                acoustic_hf_ratio = np.random.normal(0.46, 0.15)
                temp_delta_c = np.random.normal(10.8, 3.0)
            else:
                rms_vib = np.random.normal(0.46, 0.15)
                kurtosis_vib = np.random.normal(5.65, 0.80)
                crest_factor = np.random.normal(5.15, 0.90)
                bpfo_energy = np.random.normal(0.35, 0.12)
                bpfi_energy = np.random.normal(0.37, 0.12)
                bsf_energy = np.random.normal(0.73, 0.13)
                acoustic_db = np.random.normal(63.0, 5.5)
                acoustic_hf_ratio = np.random.normal(0.66, 0.16)
                temp_delta_c = np.random.normal(12.0, 3.1)

            rows.append({
                "sample_id": f"S_{label_idx}_{i:04d}",
                "rms_vib_g": round(rms_vib, 5),
                "kurtosis_vib": round(kurtosis_vib, 5),
                "crest_factor": round(crest_factor, 5),
                "bpfo_energy_norm": round(bpfo_energy, 5),
                "bpfi_energy_norm": round(bpfi_energy, 5),
                "bsf_energy_norm": round(bsf_energy, 5),
                "acoustic_spl_db": round(acoustic_db, 5),
                "acoustic_hf_ratio": round(acoustic_hf_ratio, 5),
                "temp_delta_c": round(temp_delta_c, 5),
                "label": label_idx,
                "class_name": label_name
            })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Generated {len(df)} samples across {len(classes)} classes -> {output_path}")

if __name__ == "__main__":
    generate_bearing_dataset()
