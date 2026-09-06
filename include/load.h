#ifndef LOAD_H
#define LOAD_H

#include <stddef.h>

// -------------------------------------------------------------
// Data Load & Store Interface
// -------------------------------------------------------------

// Tải dữ liệu ảnh/input từ file text vào buffer cục bộ
void load_input(const char *filename, float *dest, int size);

// Tải Weight & Bias từ bộ nhớ DRAM 1 (vùng chứa tham số tĩnh) vào BRAM/Buffer nội bộ
// tham số offset: địa chỉ bắt đầu của module trong DRAM 1
void load_weight_bias(const float *dram1, int offset, float *dest, int size);

// Tải Input Feature Map (IFM) từ bộ nhớ DRAM 2 (vùng chứa trung gian) vào BRAM/Buffer nội bộ
// tham số offset: địa chỉ bắt đầu của input module trong DRAM 2
void load_ifm(const float *dram2, int offset, float *dest, int size);

// Lưu Output Feature Map (OFM) từ BRAM/Buffer nội bộ trả về lại DRAM 2
// tham số offset: địa chỉ bắt đầu của output module trong DRAM 2
void store_ofm(float *dram2, int offset, const float *src, int size);

#endif // LOAD_H