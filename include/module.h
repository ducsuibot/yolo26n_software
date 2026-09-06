#ifndef MODULE_H
#define MODULE_H

#include <stdint.h>

// -------------------------------------------------------------
// YOLO26n MODULES HEADER (Hardcoded Shapes & DMA Mapping)
// -------------------------------------------------------------

// --- BACKBONE ---
// Module 0 là module duy nhất nhận input từ ngoài (ảnh gốc)
void module_0_conv_P1(const float *dram1, float *dram2, const float *input_image);

// Các module còn lại chỉ cần giao tiếp với DRAM1 (Weights) và DRAM2 (IFM/OFM)
void module_1_conv_P2(const float *dram1, float *dram2);
void module_2_c3k2_P2(const float *dram1, float *dram2);
void module_3_conv_P3(const float *dram1, float *dram2);
void module_4_c3k2_P3(const float *dram1, float *dram2);
void module_5_conv_P4(const float *dram1, float *dram2);
void module_6_c3k2_P4(const float *dram1, float *dram2);
void module_7_conv_P5(const float *dram1, float *dram2);
void module_8_c3k2_P5(const float *dram1, float *dram2);

// --- NECK ---
void module_9_sppf(const float *dram1, float *dram2);
void module_10_c2psa(const float *dram1, float *dram2);
void module_11_upsample(const float *dram1, float *dram2);
void module_12_concat(const float *dram1, float *dram2);
void module_13_c3k2(const float *dram1, float *dram2);
void module_14_upsample(const float *dram1, float *dram2);
void module_15_concat(const float *dram1, float *dram2);
void module_16_c3k2_detect_P3(const float *dram1, float *dram2);
void module_17_conv_downsample(const float *dram1, float *dram2);
void module_18_concat(const float *dram1, float *dram2);
void module_19_c3k2_detect_P4(const float *dram1, float *dram2);
void module_20_conv_downsample(const float *dram1, float *dram2);

// --- HEAD ---
void module_21_concat(const float *dram1, float *dram2);
void module_22_c3k2_detect_P5(const float *dram1, float *dram2);
void module_23_detect_head(const float *dram1, float *dram2);

#endif // MODULE_H