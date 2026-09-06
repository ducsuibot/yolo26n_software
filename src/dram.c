#include "dram.h"
#include <stdio.h>
#include <stdlib.h>

// Định nghĩa giới hạn không gian nhớ (số lượng elements float)
#define INPUT_SIZE   1228800     // 1x3x640x640
#define DRAM1_SIZE   (2417124 + 1)     // Tổng tham số của YOLO26n
#define DRAM2_SIZE   (62623099 + 1)    // Đủ chứa toàn bộ khối lượng OFM nối tiếp của Backbone/ Neck
//2417073
//32000000
float *input_buffer = NULL;
float *dram1 = NULL;
float *dram2 = NULL;

void init_dram() {
    input_buffer = (float *)malloc(INPUT_SIZE * sizeof(float));
    dram1 = (float *)malloc(DRAM1_SIZE * sizeof(float));
    dram2 = (float *)malloc(DRAM2_SIZE * sizeof(float));

    if (!input_buffer || !dram1 || !dram2) {
        printf("[ERROR] Khong the cap phat bo nho DRAM (vui long kiem tra RAM cua he thong).\n");
        exit(EXIT_FAILURE);
    }
    printf("[INFO] Cap phat thanh cong %d MB cho DRAM.\n", 
           (INPUT_SIZE + DRAM1_SIZE + DRAM2_SIZE) * 4 / (1024 * 1024));
}

void free_dram() {
    if (input_buffer) free(input_buffer);
    if (dram1) free(dram1);
    if (dram2) free(dram2);
    printf("[INFO] Giai phong DRAM thanh cong.\n");
}

void load_txt_to_dram(const char *filename, float *dram_ptr, int offset, int size) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        printf("[ERROR] Khong the mo file: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < size; i++) {
        if (fscanf(f, "%f", &dram_ptr[offset + i]) != 1) {
            printf("[ERROR] Loi doc du lieu tai file %s (index %d)\n", filename, i);
            break;
        }
    }
    fclose(f);
    printf("[INFO] Load %d elements vao offset %d tu %s\n", size, offset, filename);
}