#include "load.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// -------------------------------------------------------------
// Data Load & Store Implementation
// -------------------------------------------------------------

void load_input(const char *filename, float *dest, int size) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        printf("[ERROR] Khong the mo file input: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < size; i++) {
        if (fscanf(f, "%f", &dest[i]) != 1) {
            printf("[ERROR] Doc loi file hoac file khong du %d elements tai index %d.\n", size, i);
            fclose(f);
            exit(EXIT_FAILURE);
        }
    }
    fclose(f);
    printf("[INFO] Load thanh cong %d elements tu %s\n", size, filename);
}

void load_weight_bias(const float *dram1, int offset, float *dest, int size) {
    if (size > 0) {
        // Mô phỏng DMA Read (MM2S) từ DRAM1 vào Local Buffer
        memcpy(dest, &dram1[offset], size * sizeof(float));
    }
}

void load_ifm(const float *dram2, int offset, float *dest, int size) {
    if (size > 0) {
        // Mô phỏng DMA Read (MM2S) từ DRAM2 vào Local Buffer
        memcpy(dest, &dram2[offset], size * sizeof(float));
    }
}

void store_ofm(float *dram2, int offset, const float *src, int size) {
    if (size > 0) {
        // Mô phỏng DMA Write (S2MM) từ Local Buffer ra DRAM2
        memcpy(&dram2[offset], src, size * sizeof(float));
    }
}