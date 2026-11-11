#ifndef BEARING_FAULT_MODEL_DATA_H_
#define BEARING_FAULT_MODEL_DATA_H_

#include <stdint.h>

// Optimized On-Device AI Model Artifact:
// Architecture : Depthwise-Separable 2D CNN (DS-CNN)
// Optimization : Structured Magnitude Pruning (50% sparsity) + Full INT8 PTQ
// Input Shape  : [1, 32, 32, 2] (INT8, scale = 0.03125f, zero_point = -12)
// Output Shape : [1, 4] (Normal, Inner_Race_Fault, Outer_Race_Fault, Ball_Defect)
// Model Size   : 27033 bytes (26.4 KB Flash)
// Tensor Arena : 37,683 bytes (36.8 KB SRAM)

extern const unsigned char g_bearing_fault_model_data[];
extern const unsigned int g_bearing_fault_model_data_len;

#endif  // BEARING_FAULT_MODEL_DATA_H_
