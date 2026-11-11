#ifndef SENSING_PIPELINE_H_
#define SENSING_PIPELINE_H_

#include <Arduino.h>
#include <stdint.h>

// Hardware & Sampling Configuration matching Report Section 2:
// - IMU (BMI270 / LSM9DS1) : Event-driven FIFO watermark + hardware wake-on-motion (>0.35g)
//                            Sampling rate: 1,600 Hz, Range: +/- 16g, 16-bit resolution
// - Acoustic Mic (MP34DT05): PDM via EasyDMA double-buffered streaming @ 16,000 Hz
// - Temperature (TMP117)   : Low-power I2C polling @ 0.2 Hz (every 5 seconds)

#define IMU_SAMPLING_FREQ_HZ       1600
#define AUDIO_SAMPLING_FREQ_HZ     16000
#define WINDOW_SAMPLES_IMU         512
#define SPECTROGRAM_BINS           32
#define SPECTROGRAM_FRAMES         32
#define WAKE_THRESHOLD_G           0.35f

struct BearingFeatureFrame {
    float rms_vib_g;
    float kurtosis_vib;
    float crest_factor;
    float bpfo_energy_norm;
    float bpfi_energy_norm;
    float bsf_energy_norm;
    float acoustic_spl_db;
    float acoustic_hf_ratio;
    float temp_delta_c;
    int8_t quantized_spectrogram[SPECTROGRAM_FRAMES][SPECTROGRAM_BINS][2];
};

class SensingPipeline {
public:
    bool begin();
    bool isWakeEventTriggered();
    void captureAndExtractFeatures(BearingFeatureFrame* out_frame);
    float readBearingTemperatureC();

private:
    float ambient_baseline_temp_c_ = 24.0f;
    void computeWindowedFFTAndQuantize(const float* accel_buf, const int16_t* audio_buf, BearingFeatureFrame* out_frame);
};

#endif  // SENSING_PIPELINE_H_
