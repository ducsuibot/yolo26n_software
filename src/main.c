// #include <stdio.h>
// #include <stdlib.h>
// #include <math.h>
// #include "dram.h"
// #include "module.h"

// #define W_PATH "weight/backbone/"
// #define N_PATH "weight/neck/"

// // Cấu trúc map địa chỉ DRAM_2 cho mục đích đối chiếu
// typedef struct {
//     int id;
//     int offset;
//     int size;
//     const char *name;
// } NodeMap;

// // Hàm so sánh và tính sai số (Trả về 1 nếu Fail, 0 nếu Pass)
// int verify_golden(float *c_output, const char *golden_file, int size, const char *node_name) {
//     float *golden = (float *)malloc(size * sizeof(float));
//     if (!golden) {
//         printf("[ERROR] Malloc failed cho mảng Golden.\n");
//         return 1;
//     }
    
//     FILE *f = fopen(golden_file, "r");
//     if (!f) {
//         printf("[ERROR] Không tìm thấy %s. Bạn đã chạy extract_weights.py chưa?\n", golden_file);
//         free(golden);
//         return 1;
//     }
//     for (int i = 0; i < size; i++) {
//         fscanf(f, "%f", &golden[i]);
//     }
//     fclose(f);

//     float max_err = 0.0f;
//     int err_idx = -1;
//     int print_limit = 5; 

//     for (int i = 0; i < size; i++) {
//         float diff = fabs(c_output[i] - golden[i]);
//         if (diff > max_err) {
//             max_err = diff;
//             err_idx = i;
//         }
//         if (diff > 8e-3 && print_limit > 0) {
//             printf("   -> [MISMATCH] Index %d | C_Out: %.6f | Golden: %.6f | Diff: %.6f\n", 
//                 i, c_output[i], golden[i], diff);
//             print_limit--;
//         }
//     }

//     free(golden);

//     float threshold = 8e-3;// 0.008
//     if (max_err < threshold) {
//         printf("[PASS] %-20s | Size: %7d | Max Err: %.8f\n", node_name, size, max_err);
//         return 0;
//     } else {
//         printf("\n==================================================\n");
//         printf(" [FAIL] LỖI TẠI %s\n", node_name);
//         printf(" - File tham chiếu : %s\n", golden_file);
//         printf(" - Lệch lớn nhất   : %.8f (tại index %d)\n", max_err, err_idx);
//         printf("==================================================\n");
//         return 1;
//     }
// }

// int main() {
//     printf("==================================================\n");
//     printf("      YOLO26n BACKBONE & NECK C VERIFICATION ENGINE\n");
//     printf("==================================================\n\n");

//     init_dram();

//     printf("--- BƯỚC 1: LOAD DỮ LIỆU INPUT & TRỌNG SỐ ---\n");
//     load_txt_to_dram("input_images.txt", input_buffer, 0, 1228800);

//     // --- MODULE 0 ---
//     load_txt_to_dram(W_PATH "module_0_conv_P1/node_0_weight.txt", dram1, 0, 432);
//     load_txt_to_dram(W_PATH "module_0_conv_P1/node_0_bias.txt", dram1, 432, 16);
//     // --- MODULE 1 ---
//     load_txt_to_dram(W_PATH "module_1_conv_P2/node_3_weight.txt", dram1, 448, 4608);
//     load_txt_to_dram(W_PATH "module_1_conv_P2/node_3_bias.txt", dram1, 5056, 32);
//     // --- MODULE 2 ---
//     load_txt_to_dram(W_PATH "module_2_c3k2_P2/node_6_weight.txt", dram1, 5088, 1024);
//     load_txt_to_dram(W_PATH "module_2_c3k2_P2/node_6_bias.txt", dram1, 6112, 32);
//     load_txt_to_dram(W_PATH "module_2_c3k2_P2/node_10_weight.txt", dram1, 6146, 1152);
//     load_txt_to_dram(W_PATH "module_2_c3k2_P2/node_10_bias.txt", dram1, 7298, 8);
//     load_txt_to_dram(W_PATH "module_2_c3k2_P2/node_13_weight.txt", dram1, 7306, 1152);
//     load_txt_to_dram(W_PATH "module_2_c3k2_P2/node_13_bias.txt", dram1, 8458, 16);
//     load_txt_to_dram(W_PATH "module_2_c3k2_P2/node_18_weight.txt", dram1, 8474, 3072);
//     load_txt_to_dram(W_PATH "module_2_c3k2_P2/node_18_bias.txt", dram1, 11546, 64);
//     // --- MODULE 3 ---
//     load_txt_to_dram(W_PATH "module_3_conv_P3/node_21_weight.txt", dram1, 11610, 36864);
//     load_txt_to_dram(W_PATH "module_3_conv_P3/node_21_bias.txt", dram1, 48474, 64);
//     // --- MODULE 4 ---
//     load_txt_to_dram(W_PATH "module_4_c3k2_P3/node_24_weight.txt", dram1, 48538, 4096);
//     load_txt_to_dram(W_PATH "module_4_c3k2_P3/node_24_bias.txt", dram1, 52634, 64);
//     load_txt_to_dram(W_PATH "module_4_c3k2_P3/node_28_weight.txt", dram1, 52700, 4608);
//     load_txt_to_dram(W_PATH "module_4_c3k2_P3/node_28_bias.txt", dram1, 57308, 16);
//     load_txt_to_dram(W_PATH "module_4_c3k2_P3/node_31_weight.txt", dram1, 57324, 4608);
//     load_txt_to_dram(W_PATH "module_4_c3k2_P3/node_31_bias.txt", dram1, 61932, 32);
//     load_txt_to_dram(W_PATH "module_4_c3k2_P3/node_36_weight.txt", dram1, 61964, 12288);
//     load_txt_to_dram(W_PATH "module_4_c3k2_P3/node_36_bias.txt", dram1, 74252, 128);
//     // --- MODULE 5 ---
//     load_txt_to_dram(W_PATH "module_5_conv_P4/node_39_weight.txt", dram1, 74380, 147456);
//     load_txt_to_dram(W_PATH "module_5_conv_P4/node_39_bias.txt", dram1, 221836, 128);
//     // --- MODULE 6 ---
//     load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_42_weight.txt", dram1, 221964, 16384);
//     load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_42_bias.txt", dram1, 238348, 128);
//     load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_46_weight.txt", dram1, 238478, 2048);
//     load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_46_bias.txt", dram1, 240526, 32);
//     load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_47_weight.txt", dram1, 240558, 2048);
//     load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_47_bias.txt", dram1, 242606, 32);
//     load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_52_weight.txt", dram1, 242638, 9216);
//     load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_52_bias.txt", dram1, 251854, 32);
//     load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_55_weight.txt", dram1, 251886, 9216);
//     load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_55_bias.txt", dram1, 261102, 32);
//     load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_59_weight.txt", dram1, 261134, 9216);
//     load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_59_bias.txt", dram1, 270350, 32);
//     load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_62_weight.txt", dram1, 270382, 9216);
//     load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_62_bias.txt", dram1, 279598, 32);
//     load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_67_weight.txt", dram1, 279630, 4096);
//     load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_67_bias.txt", dram1, 283726, 64);
//     load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_71_weight.txt", dram1, 283790, 24576);
//     load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_71_bias.txt", dram1, 308366, 128);
//     // --- MODULE 7 ---
//     load_txt_to_dram(W_PATH "module_7_conv_P5/node_74_weight.txt", dram1, 308494, 294912);
//     load_txt_to_dram(W_PATH "module_7_conv_P5/node_74_bias.txt", dram1, 603406, 256);
//     // --- MODULE 8 ---
//     load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_77_weight.txt", dram1, 603662, 65536);
//     load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_77_bias.txt", dram1, 669198, 256);
//     load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_81_weight.txt", dram1, 669456, 8192);
//     load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_81_bias.txt", dram1, 677648, 64);
//     load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_82_weight.txt", dram1, 677712, 8192);
//     load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_82_bias.txt", dram1, 685904, 64);
//     load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_87_weight.txt", dram1, 685968, 36864);
//     load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_87_bias.txt", dram1, 722832, 64);
//     load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_90_weight.txt", dram1, 722896, 36864);
//     load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_90_bias.txt", dram1, 759760, 64);
//     load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_94_weight.txt", dram1, 759824, 36864);
//     load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_94_bias.txt", dram1, 796688, 64);
//     load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_97_weight.txt", dram1, 796752, 36864);
//     load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_97_bias.txt", dram1, 833616, 64);
//     load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_102_weight.txt", dram1, 833680, 16384);
//     load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_102_bias.txt", dram1, 850064, 128);
//     load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_106_weight.txt", dram1, 850192, 98304);
//     load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_106_bias.txt", dram1, 948496, 256);
    
//     // --- MODULE 9 ---
//     load_txt_to_dram(N_PATH "module_9_sppf/node_109_weight.txt", dram1, 948752, 32768);
//     load_txt_to_dram(N_PATH "module_9_sppf/node_109_bias.txt", dram1, 981520, 128);
//     load_txt_to_dram(N_PATH "module_9_sppf/node_114_weight.txt", dram1, 981648, 131072);
//     load_txt_to_dram(N_PATH "module_9_sppf/node_114_bias.txt", dram1, 1112720, 256);

//     // --- MODULE 10 ---
//     load_txt_to_dram(N_PATH "module_10_c2psa/node_118_weight.txt", dram1, 1112976, 65536);
//     load_txt_to_dram(N_PATH "module_10_c2psa/node_118_bias.txt", dram1, 1178512, 256);
//     load_txt_to_dram(N_PATH "module_10_c2psa/node_122_weight.txt", dram1, 1178770, 32768);
//     load_txt_to_dram(N_PATH "module_10_c2psa/node_122_bias.txt", dram1, 1211538, 256);
//     // load_txt_to_dram(N_PATH "module_10_c2psa/node_125_scale.txt", dram1, 1211801, 1);
//     load_txt_to_dram(N_PATH "module_10_c2psa/node_128_weight.txt", dram1, 1211806, 1152);
//     load_txt_to_dram(N_PATH "module_10_c2psa/node_128_bias.txt", dram1, 1212958, 128);
//     load_txt_to_dram(N_PATH "module_10_c2psa/node_135_weight.txt", dram1, 1213090, 16384);
//     load_txt_to_dram(N_PATH "module_10_c2psa/node_135_bias.txt", dram1, 1229474, 128);
//     load_txt_to_dram(N_PATH "module_10_c2psa/node_137_weight.txt", dram1, 1229602, 32768);
//     load_txt_to_dram(N_PATH "module_10_c2psa/node_137_bias.txt", dram1, 1262370, 256);
//     load_txt_to_dram(N_PATH "module_10_c2psa/node_140_weight.txt", dram1, 1262626, 32768);
//     load_txt_to_dram(N_PATH "module_10_c2psa/node_140_bias.txt", dram1, 1295394, 128);
//     load_txt_to_dram(N_PATH "module_10_c2psa/node_143_weight.txt", dram1, 1295522, 65536);
//     load_txt_to_dram(N_PATH "module_10_c2psa/node_143_bias.txt", dram1, 1361058, 256);

//     // --- MODULE 13 ---
//     load_txt_to_dram(N_PATH "module_13_c3k2/node_148_weight.txt", dram1, 1361318, 49152);
//     load_txt_to_dram(N_PATH "module_13_c3k2/node_148_bias.txt", dram1, 1410470, 128);
//     load_txt_to_dram(N_PATH "module_13_c3k2/node_152_weight.txt", dram1, 1410600, 2048);
//     load_txt_to_dram(N_PATH "module_13_c3k2/node_152_bias.txt", dram1, 1412648, 32);
//     load_txt_to_dram(N_PATH "module_13_c3k2/node_153_weight.txt", dram1, 1412680, 2048);
//     load_txt_to_dram(N_PATH "module_13_c3k2/node_153_bias.txt", dram1, 1414728, 32);
//     load_txt_to_dram(N_PATH "module_13_c3k2/node_158_weight.txt", dram1, 1414760, 9216);
//     load_txt_to_dram(N_PATH "module_13_c3k2/node_158_bias.txt", dram1, 1423976, 32);
//     load_txt_to_dram(N_PATH "module_13_c3k2/node_161_weight.txt", dram1, 1424008, 9216);
//     load_txt_to_dram(N_PATH "module_13_c3k2/node_161_bias.txt", dram1, 1433224, 32);
//     load_txt_to_dram(N_PATH "module_13_c3k2/node_165_weight.txt", dram1, 1433256, 9216);
//     load_txt_to_dram(N_PATH "module_13_c3k2/node_165_bias.txt", dram1, 1442472, 32);
//     load_txt_to_dram(N_PATH "module_13_c3k2/node_168_weight.txt", dram1, 1442504, 9216);
//     load_txt_to_dram(N_PATH "module_13_c3k2/node_168_bias.txt", dram1, 1451720, 32);
//     load_txt_to_dram(N_PATH "module_13_c3k2/node_173_weight.txt", dram1, 1451752, 4096);
//     load_txt_to_dram(N_PATH "module_13_c3k2/node_173_bias.txt", dram1, 1455848, 64);
//     load_txt_to_dram(N_PATH "module_13_c3k2/node_177_weight.txt", dram1, 1455912, 24576);
//     load_txt_to_dram(N_PATH "module_13_c3k2/node_177_bias.txt", dram1, 1480488, 128);

//     // --- MODULE 16 ---
//     load_txt_to_dram(N_PATH "module_16_c3k2/node_182_weight.txt", dram1, 1480620, 16384);
//     load_txt_to_dram(N_PATH "module_16_c3k2/node_182_bias.txt", dram1, 1497004, 64);
//     load_txt_to_dram(N_PATH "module_16_c3k2/node_186_weight.txt", dram1, 1497070, 512);
//     load_txt_to_dram(N_PATH "module_16_c3k2/node_186_bias.txt", dram1, 1497582, 16);
//     load_txt_to_dram(N_PATH "module_16_c3k2/node_187_weight.txt", dram1, 1497598, 512);
//     load_txt_to_dram(N_PATH "module_16_c3k2/node_187_bias.txt", dram1, 1498110, 16);
//     load_txt_to_dram(N_PATH "module_16_c3k2/node_192_weight.txt", dram1, 1498126, 2304);
//     load_txt_to_dram(N_PATH "module_16_c3k2/node_192_bias.txt", dram1, 1500430, 16);
//     load_txt_to_dram(N_PATH "module_16_c3k2/node_195_weight.txt", dram1, 1500446, 2304);
//     load_txt_to_dram(N_PATH "module_16_c3k2/node_195_bias.txt", dram1, 1502750, 16);
//     load_txt_to_dram(N_PATH "module_16_c3k2/node_199_weight.txt", dram1, 1502766, 2304);
//     load_txt_to_dram(N_PATH "module_16_c3k2/node_199_bias.txt", dram1, 1505070, 16);
//     load_txt_to_dram(N_PATH "module_16_c3k2/node_202_weight.txt", dram1, 1505086, 2304);
//     load_txt_to_dram(N_PATH "module_16_c3k2/node_202_bias.txt", dram1, 1507390, 16);
//     load_txt_to_dram(N_PATH "module_16_c3k2/node_207_weight.txt", dram1, 1507406, 1024);
//     load_txt_to_dram(N_PATH "module_16_c3k2/node_207_bias.txt", dram1, 1508430, 32);
//     load_txt_to_dram(N_PATH "module_16_c3k2/node_211_weight.txt", dram1, 1508462, 6144);
//     load_txt_to_dram(N_PATH "module_16_c3k2/node_211_bias.txt", dram1, 1514606, 64);

//     // --- MODULE 17 ---
//     load_txt_to_dram(N_PATH "module_17_conv/node_214_weight.txt", dram1, 1514670, 36864);
//     load_txt_to_dram(N_PATH "module_17_conv/node_214_bias.txt", dram1, 1551534, 64);

//     // --- MODULE 19 ---
//     load_txt_to_dram(N_PATH "module_19_c3k2/node_226_weight.txt", dram1, 1567950, 24576);
//     load_txt_to_dram(N_PATH "module_19_c3k2/node_226_bias.txt", dram1, 1592526, 128);
//     load_txt_to_dram(N_PATH "module_19_c3k2/node_238_weight.txt", dram1, 1593367, 2048);
//     load_txt_to_dram(N_PATH "module_19_c3k2/node_238_bias.txt", dram1, 1595415, 32);
//     load_txt_to_dram(N_PATH "module_19_c3k2/node_239_weight.txt", dram1, 1595447, 2048);
//     load_txt_to_dram(N_PATH "module_19_c3k2/node_239_bias.txt", dram1, 1597495, 32);
//     load_txt_to_dram(N_PATH "module_19_c3k2/node_247_weight.txt", dram1, 1601687, 9216);
//     load_txt_to_dram(N_PATH "module_19_c3k2/node_247_bias.txt", dram1, 1610903, 32);
//     load_txt_to_dram(N_PATH "module_19_c3k2/node_253_weight.txt", dram1, 1611003, 9216);
//     load_txt_to_dram(N_PATH "module_19_c3k2/node_253_bias.txt", dram1, 1620219, 32);
//     load_txt_to_dram(N_PATH "module_19_c3k2/node_257_weight.txt", dram1, 1620251, 9216);
//     load_txt_to_dram(N_PATH "module_19_c3k2/node_257_bias.txt", dram1, 1629467, 32);
//     load_txt_to_dram(N_PATH "module_19_c3k2/node_260_weight.txt", dram1, 1629499, 9216);
//     load_txt_to_dram(N_PATH "module_19_c3k2/node_260_bias.txt", dram1, 1638715, 32);
//     load_txt_to_dram(N_PATH "module_19_c3k2/node_265_weight.txt", dram1, 1638747, 4096);
//     load_txt_to_dram(N_PATH "module_19_c3k2/node_265_bias.txt", dram1, 1642843, 64);
//     load_txt_to_dram(N_PATH "module_19_c3k2/node_269_weight.txt", dram1, 1642907, 24576);
//     load_txt_to_dram(N_PATH "module_19_c3k2/node_269_bias.txt", dram1, 1667483, 128);

//     // --- MODULE 20 ---
//     load_txt_to_dram(N_PATH "module_20_conv/node_272_weight.txt", dram1, 1667611, 147456);
//     load_txt_to_dram(N_PATH "module_20_conv/node_272_bias.txt", dram1, 1815067, 128);

//     printf("\n--- BƯỚC 2: RUN BACKBONE AND NECK MODULES ---\n");
//     module_0_conv_P1(dram1, dram2, input_buffer);
//     module_1_conv_P2(dram1, dram2);
//     module_2_c3k2_P2(dram1, dram2);
//     module_3_conv_P3(dram1, dram2);
//     module_4_c3k2_P3(dram1, dram2);
//     module_5_conv_P4(dram1, dram2);
//     module_6_c3k2_P4(dram1, dram2);
//     module_7_conv_P5(dram1, dram2);
//     module_8_c3k2_P5(dram1, dram2);
//     // --- NECK ---
//     module_9_sppf(dram1, dram2);
//     module_10_c2psa(dram1, dram2);
//     module_11_upsample(dram1, dram2);
//     module_12_concat(dram1, dram2);
//     module_13_c3k2(dram1, dram2);
//     module_14_upsample(dram1, dram2);
//     module_15_concat(dram1, dram2);
//     module_16_c3k2_detect_P3(dram1, dram2);
//     module_17_conv_downsample(dram1, dram2);
//     module_18_concat(dram1, dram2);
//     module_19_c3k2_detect_P4(dram1, dram2);
//     module_20_conv_downsample(dram1, dram2);

//     printf("\n--- BƯỚC 3: DÒ LỖI TỪNG NODE (LAYER-BY-LAYER VERIFICATION) ---\n");

//     NodeMap nodes_map[] = {
//         // --- Backbone ---
//         {0, 0, 1638400, "Node 0: Conv"}, {1, 1638400, 1638400, "Node 1: Sigmoid"}, {2, 3276800, 1638400, "Node 2: Mul"},
//         {3, 4915200, 819200, "Node 3: Conv"}, {4, 5734400, 819200, "Node 4: Sigmoid"}, {5, 6553600, 819200, "Node 5: Mul"},
//         {6, 7372800, 819200, "Node 6: Conv"}, {7, 8192000, 819200, "Node 7: Sigmoid"}, {8, 9011200, 819200, "Node 8: Mul"},
//         {9, 9011200, 409600, "Node 9: Split"}, {10, 10240000, 204800, "Node 10: Conv"}, {11, 10444800, 204800, "Node 11: Sigmoid"},
//         {12, 10649600, 204800, "Node 12: Mul"}, {13, 10854400, 409600, "Node 13: Conv"}, {14, 11264000, 409600, "Node 14: Sigmoid"},
//         {15, 11673600, 409600, "Node 15: Mul"}, {16, 12083200, 409600, "Node 16: Add"}, {17, 12492800, 1228800, "Node 17: Concat"},
//         {18, 13721600, 1638400, "Node 18: Conv"}, {19, 15360000, 1638400, "Node 19: Sigmoid"}, {20, 16998400, 1638400, "Node 20: Mul"},
//         {21, 18636800, 409600, "Node 21: Conv"}, {22, 19046400, 409600, "Node 22: Sigmoid"}, {23, 19456000, 409600, "Node 23: Mul"},
//         {24, 19865600, 409600, "Node 24: Conv"}, {25, 20275200, 409600, "Node 25: Sigmoid"}, {26, 20684800, 409600, "Node 26: Mul"},
//         {27, 20684800, 204800, "Node 27: Split"}, {28, 21299200, 102400, "Node 28: Conv"}, {29, 21401600, 102400, "Node 29: Sigmoid"},
//         {30, 21504000, 102400, "Node 30: Mul"}, {31, 21606400, 204800, "Node 31: Conv"}, {32, 21811200, 204800, "Node 32: Sigmoid"},
//         {33, 22016000, 204800, "Node 33: Mul"}, {34, 22220800, 204800, "Node 34: Add"}, {35, 22425600, 614400, "Node 35: Concat"},
//         {36, 23040000, 819200, "Node 36: Conv"}, {37, 23859200, 819200, "Node 37: Sigmoid"}, {38, 24678400, 819200, "Node 38: Mul"},
//         {39, 25497600, 204800, "Node 39: Conv"}, {40, 25702400, 204800, "Node 40: Sigmoid"}, {41, 25907200, 204800, "Node 41: Mul"},
//         {42, 26112000, 204800, "Node 42: Conv"}, {43, 26316800, 204800, "Node 43: Sigmoid"}, {44, 26521600, 204800, "Node 44: Mul"},
//         {45, 26521600, 102400, "Node 45: Split"}, {46, 26828800, 51200, "Node 46: Conv"}, {47, 26880000, 51200, "Node 47: Conv"},
//         {48, 26931200, 51200, "Node 48: Sigmoid"}, {49, 26982400, 51200, "Node 49: Sigmoid"}, {50, 27033600, 51200, "Node 50: Mul"},
//         {51, 27084800, 51200, "Node 51: Mul"}, {52, 27136000, 51200, "Node 52: Conv"}, {53, 27187200, 51200, "Node 53: Sigmoid"},
//         {54, 27238400, 51200, "Node 54: Mul"}, {55, 27289600, 51200, "Node 55: Conv"}, {56, 27340800, 51200, "Node 56: Sigmoid"},
//         {57, 27392000, 51200, "Node 57: Mul"}, {58, 27443200, 51200, "Node 58: Add"}, {59, 27494400, 51200, "Node 59: Conv"},
//         {60, 27545600, 51200, "Node 60: Sigmoid"}, {61, 27596800, 51200, "Node 61: Mul"}, {62, 27648000, 51200, "Node 62: Conv"},
//         {63, 27699200, 51200, "Node 63: Sigmoid"}, {64, 27750400, 51200, "Node 64: Mul"}, {65, 27801600, 51200, "Node 65: Add"},
//         {66, 27852800, 102400, "Node 66: Concat"}, {67, 27955200, 102400, "Node 67: Conv"}, {68, 28057600, 102400, "Node 68: Sigmoid"},
//         {69, 28160000, 102400, "Node 69: Mul"}, {70, 28262400, 307200, "Node 70: Concat"}, {71, 28569600, 204800, "Node 71: Conv"},
//         {72, 28774400, 204800, "Node 72: Sigmoid"}, {73, 28979200, 204800, "Node 73: Mul"}, {74, 29184000, 102400, "Node 74: Conv"},
//         {75, 29286400, 102400, "Node 75: Sigmoid"}, {76, 29388800, 102400, "Node 76: Mul"}, {77, 29491200, 102400, "Node 77: Conv"},
//         {78, 29593600, 102400, "Node 78: Sigmoid"}, {79, 29696000, 102400, "Node 79: Mul"}, {80, 29696000, 51200, "Node 80: Split"},
//         {81, 29849600, 25600, "Node 81: Conv"}, {82, 29875200, 25600, "Node 82: Conv"}, {83, 29900800, 25600, "Node 83: Sigmoid"},
//         {84, 29926400, 25600, "Node 84: Sigmoid"}, {85, 29952000, 25600, "Node 85: Mul"}, {86, 29977600, 25600, "Node 86: Mul"},
//         {87, 30003200, 25600, "Node 87: Conv"}, {88, 30028800, 25600, "Node 88: Sigmoid"}, {89, 30054400, 25600, "Node 89: Mul"},
//         {90, 30080000, 25600, "Node 90: Conv"}, {91, 30105600, 25600, "Node 91: Sigmoid"}, {92, 30131200, 25600, "Node 92: Mul"},
//         {93, 30156800, 25600, "Node 93: Add"}, {94, 30182400, 25600, "Node 94: Conv"}, {95, 30208000, 25600, "Node 95: Sigmoid"},
//         {96, 30233600, 25600, "Node 96: Mul"}, {97, 30259200, 25600, "Node 97: Conv"}, {98, 30284800, 25600, "Node 98: Sigmoid"},
//         {99, 30310400, 25600, "Node 99: Mul"}, {100, 30336000, 25600, "Node 100: Add"}, {101, 30361600, 51200, "Node 101: Concat"},
//         {102, 30412800, 51200, "Node 102: Conv"}, {103, 30464000, 51200, "Node 103: Sigmoid"}, {104, 30515200, 51200, "Node 104: Mul"},
//         {105, 30566400, 153600, "Node 105: Concat"}, {106, 30720000, 102400, "Node 106: Conv"}, {107, 30822400, 102400, "Node 107: Sigmoid"},
//         {108, 30924800, 102400, "Node 108: Mul"},
        
//         // --- Neck ---
//         // Module 9
//         {109, 31027200, 51200, "Node 109: Conv"}, {110, 31078400, 51200, "Node 110: MaxPool"},
//         {111, 31129600, 51200, "Node 111: MaxPool"}, {112, 31180800, 51200, "Node 112: MaxPool"},
//         {113, 31232000, 204800, "Node 113: Concat"}, {114, 31436800, 102400, "Node 114: Conv"},
//         {115, 31539200, 102400, "Node 115: Sigmoid"}, {116, 31641600, 102400, "Node 116: Mul"},
//         {117, 31744000, 102400, "Node 117: Add"},
        
//         // Module 10
//         {118, 31846400, 102400, "Node 118: Conv"}, {119, 31948800, 102400, "Node 119: Sigmoid"},
//         {120, 32051200, 102400, "Node 120: Mul"}, {121, 32051200, 51200, "Node 121: Split"},
//         {122, 32204800, 102400, "Node 122: Conv"}, {123, 32307200, 102400, "Node 123: Reshape"},
//         // Đổi 32307200 thành 32409600
//         {124, 32409600, 25600, "Node 124: Split"}, {125, 32435200, 25600, "Node 125: Mul"},
//         {126, 32460800, 51200, "Node 126: Reshape"}, {127, 32512000, 25600, "Node 127: Transpose"},
//         {128, 32537600, 51200, "Node 128: Conv"}, {129, 32588800, 320000, "Node 129: MatMul"},
//         {130, 32908800, 320000, "Node 130: Softmax"}, {131, 33228800, 320000, "Node 131: Transpose"},
//         {132, 33548800, 51200, "Node 132: MatMul"}, {133, 33600000, 51200, "Node 133: Reshape"},
//         {134, 33651200, 51200, "Node 134: Add"}, {135, 33702400, 51200, "Node 135: Conv"},
//         {136, 33753600, 51200, "Node 136: Add"}, {137, 33804800, 102400, "Node 137: Conv"},
//         {138, 33907200, 102400, "Node 138: Sigmoid"}, {139, 34009600, 102400, "Node 139: Mul"},
//         {140, 34112000, 51200, "Node 140: Conv"}, {141, 34163200, 51200, "Node 141: Add"},
//         {142, 34214400, 102400, "Node 142: Concat"}, {143, 34316800, 102400, "Node 143: Conv"},
//         {144, 34419200, 102400, "Node 144: Sigmoid"}, {145, 34521600, 102400, "Node 145: Mul"},

//         // Module 11
//         {146, 34624000, 409600, "Node 146: Resize"},
        
//         // Module 12
//         {147, 35033600, 614400, "Node 147: Concat"},
        
//         // Module 13
//         {148, 35648000, 204800, "Node 148: Conv"}, {149, 35852800, 204800, "Node 149: Sigmoid"},
//         {150, 36057600, 204800, "Node 150: Mul"}, {151, 36057600, 102400, "Node 151: Split"},
//         {152, 36364800, 51200, "Node 152: Conv"}, {153, 36416000, 51200, "Node 153: Conv"},
//         {154, 36467200, 51200, "Node 154: Sigmoid"}, {155, 36518400, 51200, "Node 155: Sigmoid"},
//         {156, 36569600, 51200, "Node 156: Mul"}, {157, 36620800, 51200, "Node 157: Mul"},
//         {158, 36672000, 51200, "Node 158: Conv"}, {159, 36723200, 51200, "Node 159: Sigmoid"},
//         {160, 36774400, 51200, "Node 160: Mul"}, {161, 36825600, 51200, "Node 161: Conv"},
//         {162, 36876800, 51200, "Node 162: Sigmoid"}, {163, 36928000, 51200, "Node 163: Mul"},
//         {164, 36979200, 51200, "Node 164: Add"}, {165, 37030400, 51200, "Node 165: Conv"},
//         {166, 37081600, 51200, "Node 166: Sigmoid"}, {167, 37132800, 51200, "Node 167: Mul"},
//         {168, 37184000, 51200, "Node 168: Conv"}, {169, 37235200, 51200, "Node 169: Sigmoid"},
//         {170, 37286400, 51200, "Node 170: Mul"}, {171, 37337600, 51200, "Node 171: Add"},
//         {172, 37388800, 102400, "Node 172: Concat"}, {173, 37491200, 102400, "Node 173: Conv"},
//         {174, 37593600, 102400, "Node 174: Sigmoid"}, {175, 37696000, 102400, "Node 175: Mul"},
//         {176, 37798400, 307200, "Node 176: Concat"}, {177, 38105600, 204800, "Node 177: Conv"},
//         {178, 38310400, 204800, "Node 178: Sigmoid"}, {179, 38515200, 204800, "Node 179: Mul"},

//         // Module 14
//         {180, 38720000, 819200, "Node 180: Resize"},
        
//         // Module 15
//         {181, 39539200, 1638400, "Node 181: Concat"},
        
//         // Module 16
//         {182, 41177600, 409600, "Node 182: Conv"}, {183, 41587200, 409600, "Node 183: Sigmoid"},
//         {184, 41996800, 409600, "Node 184: Mul"}, {185, 41996800, 204800, "Node 185: Split"},
//         {186, 42611200, 102400, "Node 186: Conv"}, {187, 42713600, 102400, "Node 187: Conv"},
//         {188, 42816000, 102400, "Node 188: Sigmoid"}, {189, 42918400, 102400, "Node 189: Sigmoid"},
//         {190, 43020800, 102400, "Node 190: Mul"}, {191, 43123200, 102400, "Node 191: Mul"},
//         {192, 43225600, 102400, "Node 192: Conv"}, {193, 43328000, 102400, "Node 193: Sigmoid"},
//         {194, 43430400, 102400, "Node 194: Mul"}, {195, 43532800, 102400, "Node 195: Conv"},
//         {196, 43635200, 102400, "Node 196: Sigmoid"}, {197, 43737600, 102400, "Node 197: Mul"},
//         {198, 43840000, 102400, "Node 198: Add"}, {199, 43942400, 102400, "Node 199: Conv"},
//         {200, 44044800, 102400, "Node 200: Sigmoid"}, {201, 44147200, 102400, "Node 201: Mul"},
//         {202, 44249600, 102400, "Node 202: Conv"}, {203, 44352000, 102400, "Node 203: Sigmoid"},
//         {204, 44454400, 102400, "Node 204: Mul"}, {205, 44556800, 102400, "Node 205: Add"},
//         {206, 44659200, 204800, "Node 206: Concat"}, {207, 44864000, 204800, "Node 207: Conv"},
//         {208, 45068800, 204800, "Node 208: Sigmoid"}, {209, 45273600, 204800, "Node 209: Mul"},
//         {210, 45478400, 614400, "Node 210: Concat"}, {211, 46092800, 409600, "Node 211: Conv"},
//         {212, 46502400, 409600, "Node 212: Sigmoid"}, {213, 46912000, 409600, "Node 213: Mul"},

//         // Module 17
//         {214, 47321600, 102400, "Node 214: Conv"}, {217, 47936000, 102400, "Node 217: Sigmoid"},
//         {220, 48550400, 102400, "Node 220: Mul"},

//         // Module 18
//         {223, 49164800, 307200, "Node 223: Concat"},
        
//         // Module 19
//         {226, 49984000, 204800, "Node 226: Conv"}, {229, 50700800, 204800, "Node 229: Sigmoid"},
//         {232, 51417600, 204800, "Node 232: Mul"}, {235, 51417600, 102400, "Node 235: Split"},
//         {238, 52595200, 51200, "Node 238: Conv"}, {239, 52646400, 51200, "Node 239: Conv"},
//         {241, 53107200, 51200, "Node 241: Sigmoid"}, {244, 53619200, 51200, "Node 244: Mul"},
//         {247, 54131200, 51200, "Node 247: Conv"}, {249, 54592000, 51200, "Node 249: Sigmoid"},
//         {251, 54649600, 51200, "Node 251: Mul"}, {253, 54707200, 51200, "Node 253: Conv"},
//         {254, 54758400, 51200, "Node 254: Sigmoid"}, {255, 54809600, 51200, "Node 255: Mul"},
//         {256, 54860800, 51200, "Node 256: Add"}, {257, 54912000, 51200, "Node 257: Conv"},
//         {258, 54963200, 51200, "Node 258: Sigmoid"}, {259, 55014400, 51200, "Node 259: Mul"},
//         {260, 55065600, 51200, "Node 260: Conv"}, {261, 55116800, 51200, "Node 261: Sigmoid"},
//         {262, 55168000, 51200, "Node 262: Mul"}, {263, 55219200, 51200, "Node 263: Add"},
//         {264, 55270400, 102400, "Node 264: Concat"}, {265, 55372800, 102400, "Node 265: Conv"},
//         {266, 55475200, 102400, "Node 266: Sigmoid"}, {267, 55577600, 102400, "Node 267: Mul"},
//         {268, 55680000, 307200, "Node 268: Concat"}, {269, 55987200, 204800, "Node 269: Conv"},
//         {270, 56192000, 204800, "Node 270: Sigmoid"}, {271, 56396800, 204800, "Node 271: Mul"},

//         // Module 20
//         {272, 56601600, 51200, "Node 272: Conv"}, {275, 56883200, 51200, "Node 275: Sigmoid"},
//         {278, 57164800, 51200, "Node 278: Mul"}
//     };

//     int num_nodes = sizeof(nodes_map) / sizeof(NodeMap);
//     char filepath[256];

//     for (int i = 0; i < num_nodes; i++) {
//         sprintf(filepath, "golden_outputs/node_%d_golden.txt", nodes_map[i].id);
        
//         // Quét so sánh mảng C trong DRAM_2 với file tham chiếu ONNX
//         int status = verify_golden(&dram2[nodes_map[i].offset], filepath, nodes_map[i].size, nodes_map[i].name);
        
//         if (status != 0) {
//             printf("\n[!!!] Quá trình dò lỗi đã DỪNG LẠI tại %s do phát hiện sai số logic.\n", nodes_map[i].name);
//             break; 
//         }
//     }

//     free_dram();
//     return 0;
// }
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dram.h"
#include "module.h"

#define W_PATH "weight/backbone/"
#define N_PATH "weight/neck/"
#define H_PATH "weight/head/"

typedef struct {
    int id;
    int offset;
    int size;
    const char *name;
} NodeMap;

int verify_golden(float *c_output, const char *golden_file, int size, const char *node_name) {
    float *golden = (float *)malloc(size * sizeof(float));
    if (!golden) {
        printf("[ERROR] Malloc failed cho mảng Golden.\n");
        return 1;
    }
    
    FILE *f = fopen(golden_file, "r");
    if (!f) {
        printf("[ERROR] Không tìm thấy %s. Bạn đã chạy extract_weights.py chưa?\n", golden_file);
        free(golden);
        return 1;
    }
    for (int i = 0; i < size; i++) {
        fscanf(f, "%f", &golden[i]);
    }
    fclose(f);

    float max_err = 0.0f;
    int err_idx = -1;
    int print_limit = 5; 

    for (int i = 0; i < size; i++) {
        float diff = fabs(c_output[i] - golden[i]);
        if (diff > max_err) {
            max_err = diff;
            err_idx = i;
        }
        if (diff > 0.03 && print_limit > 0) {
            printf("   -> [MISMATCH] Index %d | C_Out: %.6f | Golden: %.6f | Diff: %.6f\n", 
                i, c_output[i], golden[i], diff);
            print_limit--;
        }
    }

    free(golden);

    float threshold = 0.03;// 0.03
    if (max_err < threshold) {
        printf("[PASS] %-20s | Size: %7d | Max Err: %.8f\n", node_name, size, max_err);
        return 0;
    } else {
        printf("\n==================================================\n");
        printf(" [FAIL] LỖI TẠI %s\n", node_name);
        printf(" - File tham chiếu : %s\n", golden_file);
        printf(" - Lệch lớn nhất   : %.8f (tại index %d)\n", max_err, err_idx);
        printf("==================================================\n");
        return 1;
    }
}

int main() {
    printf("==================================================\n");
    printf("  YOLO26n FULL NETWORK C VERIFICATION ENGINE\n");
    printf("==================================================\n\n");

    init_dram();

    printf("--- BƯỚC 1: LOAD DỮ LIỆU INPUT & TRỌNG SỐ ---\n");
    load_txt_to_dram("input_images.txt", input_buffer, 0, 1228800);

    // --- MODULE BACKBONE ---
    load_txt_to_dram(W_PATH "module_0_conv_P1/node_0_weight.txt", dram1, 0, 432);
    load_txt_to_dram(W_PATH "module_0_conv_P1/node_0_bias.txt", dram1, 432, 16);
    load_txt_to_dram(W_PATH "module_1_conv_P2/node_3_weight.txt", dram1, 448, 4608);
    load_txt_to_dram(W_PATH "module_1_conv_P2/node_3_bias.txt", dram1, 5056, 32);
    load_txt_to_dram(W_PATH "module_2_c3k2_P2/node_6_weight.txt", dram1, 5088, 1024);
    load_txt_to_dram(W_PATH "module_2_c3k2_P2/node_6_bias.txt", dram1, 6112, 32);
    load_txt_to_dram(W_PATH "module_2_c3k2_P2/node_10_weight.txt", dram1, 6146, 1152);
    load_txt_to_dram(W_PATH "module_2_c3k2_P2/node_10_bias.txt", dram1, 7298, 8);
    load_txt_to_dram(W_PATH "module_2_c3k2_P2/node_13_weight.txt", dram1, 7306, 1152);
    load_txt_to_dram(W_PATH "module_2_c3k2_P2/node_13_bias.txt", dram1, 8458, 16);
    load_txt_to_dram(W_PATH "module_2_c3k2_P2/node_18_weight.txt", dram1, 8474, 3072);
    load_txt_to_dram(W_PATH "module_2_c3k2_P2/node_18_bias.txt", dram1, 11546, 64);
    load_txt_to_dram(W_PATH "module_3_conv_P3/node_21_weight.txt", dram1, 11610, 36864);
    load_txt_to_dram(W_PATH "module_3_conv_P3/node_21_bias.txt", dram1, 48474, 64);
    load_txt_to_dram(W_PATH "module_4_c3k2_P3/node_24_weight.txt", dram1, 48538, 4096);
    load_txt_to_dram(W_PATH "module_4_c3k2_P3/node_24_bias.txt", dram1, 52634, 64);
    load_txt_to_dram(W_PATH "module_4_c3k2_P3/node_28_weight.txt", dram1, 52700, 4608);
    load_txt_to_dram(W_PATH "module_4_c3k2_P3/node_28_bias.txt", dram1, 57308, 16);
    load_txt_to_dram(W_PATH "module_4_c3k2_P3/node_31_weight.txt", dram1, 57324, 4608);
    load_txt_to_dram(W_PATH "module_4_c3k2_P3/node_31_bias.txt", dram1, 61932, 32);
    load_txt_to_dram(W_PATH "module_4_c3k2_P3/node_36_weight.txt", dram1, 61964, 12288);
    load_txt_to_dram(W_PATH "module_4_c3k2_P3/node_36_bias.txt", dram1, 74252, 128);
    load_txt_to_dram(W_PATH "module_5_conv_P4/node_39_weight.txt", dram1, 74380, 147456);
    load_txt_to_dram(W_PATH "module_5_conv_P4/node_39_bias.txt", dram1, 221836, 128);
    load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_42_weight.txt", dram1, 221964, 16384);
    load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_42_bias.txt", dram1, 238348, 128);
    load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_46_weight.txt", dram1, 238478, 2048);
    load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_46_bias.txt", dram1, 240526, 32);
    load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_47_weight.txt", dram1, 240558, 2048);
    load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_47_bias.txt", dram1, 242606, 32);
    load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_52_weight.txt", dram1, 242638, 9216);
    load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_52_bias.txt", dram1, 251854, 32);
    load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_55_weight.txt", dram1, 251886, 9216);
    load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_55_bias.txt", dram1, 261102, 32);
    load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_59_weight.txt", dram1, 261134, 9216);
    load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_59_bias.txt", dram1, 270350, 32);
    load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_62_weight.txt", dram1, 270382, 9216);
    load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_62_bias.txt", dram1, 279598, 32);
    load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_67_weight.txt", dram1, 279630, 4096);
    load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_67_bias.txt", dram1, 283726, 64);
    load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_71_weight.txt", dram1, 283790, 24576);
    load_txt_to_dram(W_PATH "module_6_c3k2_P4/node_71_bias.txt", dram1, 308366, 128);
    load_txt_to_dram(W_PATH "module_7_conv_P5/node_74_weight.txt", dram1, 308494, 294912);
    load_txt_to_dram(W_PATH "module_7_conv_P5/node_74_bias.txt", dram1, 603406, 256);
    load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_77_weight.txt", dram1, 603662, 65536);
    load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_77_bias.txt", dram1, 669198, 256);
    load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_81_weight.txt", dram1, 669456, 8192);
    load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_81_bias.txt", dram1, 677648, 64);
    load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_82_weight.txt", dram1, 677712, 8192);
    load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_82_bias.txt", dram1, 685904, 64);
    load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_87_weight.txt", dram1, 685968, 36864);
    load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_87_bias.txt", dram1, 722832, 64);
    load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_90_weight.txt", dram1, 722896, 36864);
    load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_90_bias.txt", dram1, 759760, 64);
    load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_94_weight.txt", dram1, 759824, 36864);
    load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_94_bias.txt", dram1, 796688, 64);
    load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_97_weight.txt", dram1, 796752, 36864);
    load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_97_bias.txt", dram1, 833616, 64);
    load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_102_weight.txt", dram1, 833680, 16384);
    load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_102_bias.txt", dram1, 850064, 128);
    load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_106_weight.txt", dram1, 850192, 98304);
    load_txt_to_dram(W_PATH "module_8_c3k2_P5/node_106_bias.txt", dram1, 948496, 256);
    
    // --- MODULE NECK ---
    load_txt_to_dram(N_PATH "module_9_sppf/node_109_weight.txt", dram1, 948752, 32768);
    load_txt_to_dram(N_PATH "module_9_sppf/node_109_bias.txt", dram1, 981520, 128);
    load_txt_to_dram(N_PATH "module_9_sppf/node_114_weight.txt", dram1, 981648, 131072);
    load_txt_to_dram(N_PATH "module_9_sppf/node_114_bias.txt", dram1, 1112720, 256);
    load_txt_to_dram(N_PATH "module_10_c2psa/node_118_weight.txt", dram1, 1112976, 65536);
    load_txt_to_dram(N_PATH "module_10_c2psa/node_118_bias.txt", dram1, 1178512, 256);
    load_txt_to_dram(N_PATH "module_10_c2psa/node_122_weight.txt", dram1, 1178770, 32768);
    load_txt_to_dram(N_PATH "module_10_c2psa/node_122_bias.txt", dram1, 1211538, 256);
    load_txt_to_dram(N_PATH "module_10_c2psa/node_128_weight.txt", dram1, 1211806, 1152);
    load_txt_to_dram(N_PATH "module_10_c2psa/node_128_bias.txt", dram1, 1212958, 128);
    load_txt_to_dram(N_PATH "module_10_c2psa/node_135_weight.txt", dram1, 1213090, 16384);
    load_txt_to_dram(N_PATH "module_10_c2psa/node_135_bias.txt", dram1, 1229474, 128);
    load_txt_to_dram(N_PATH "module_10_c2psa/node_137_weight.txt", dram1, 1229602, 32768);
    load_txt_to_dram(N_PATH "module_10_c2psa/node_137_bias.txt", dram1, 1262370, 256);
    load_txt_to_dram(N_PATH "module_10_c2psa/node_140_weight.txt", dram1, 1262626, 32768);
    load_txt_to_dram(N_PATH "module_10_c2psa/node_140_bias.txt", dram1, 1295394, 128);
    load_txt_to_dram(N_PATH "module_10_c2psa/node_143_weight.txt", dram1, 1295522, 65536);
    load_txt_to_dram(N_PATH "module_10_c2psa/node_143_bias.txt", dram1, 1361058, 256);
    load_txt_to_dram(N_PATH "module_13_c3k2/node_148_weight.txt", dram1, 1361318, 49152);
    load_txt_to_dram(N_PATH "module_13_c3k2/node_148_bias.txt", dram1, 1410470, 128);
    load_txt_to_dram(N_PATH "module_13_c3k2/node_152_weight.txt", dram1, 1410600, 2048);
    load_txt_to_dram(N_PATH "module_13_c3k2/node_152_bias.txt", dram1, 1412648, 32);
    load_txt_to_dram(N_PATH "module_13_c3k2/node_153_weight.txt", dram1, 1412680, 2048);
    load_txt_to_dram(N_PATH "module_13_c3k2/node_153_bias.txt", dram1, 1414728, 32);
    load_txt_to_dram(N_PATH "module_13_c3k2/node_158_weight.txt", dram1, 1414760, 9216);
    load_txt_to_dram(N_PATH "module_13_c3k2/node_158_bias.txt", dram1, 1423976, 32);
    load_txt_to_dram(N_PATH "module_13_c3k2/node_161_weight.txt", dram1, 1424008, 9216);
    load_txt_to_dram(N_PATH "module_13_c3k2/node_161_bias.txt", dram1, 1433224, 32);
    load_txt_to_dram(N_PATH "module_13_c3k2/node_165_weight.txt", dram1, 1433256, 9216);
    load_txt_to_dram(N_PATH "module_13_c3k2/node_165_bias.txt", dram1, 1442472, 32);
    load_txt_to_dram(N_PATH "module_13_c3k2/node_168_weight.txt", dram1, 1442504, 9216);
    load_txt_to_dram(N_PATH "module_13_c3k2/node_168_bias.txt", dram1, 1451720, 32);
    load_txt_to_dram(N_PATH "module_13_c3k2/node_173_weight.txt", dram1, 1451752, 4096);
    load_txt_to_dram(N_PATH "module_13_c3k2/node_173_bias.txt", dram1, 1455848, 64);
    load_txt_to_dram(N_PATH "module_13_c3k2/node_177_weight.txt", dram1, 1455912, 24576);
    load_txt_to_dram(N_PATH "module_13_c3k2/node_177_bias.txt", dram1, 1480488, 128);
    load_txt_to_dram(N_PATH "module_16_c3k2/node_182_weight.txt", dram1, 1480620, 16384);
    load_txt_to_dram(N_PATH "module_16_c3k2/node_182_bias.txt", dram1, 1497004, 64);
    load_txt_to_dram(N_PATH "module_16_c3k2/node_186_weight.txt", dram1, 1497070, 512);
    load_txt_to_dram(N_PATH "module_16_c3k2/node_186_bias.txt", dram1, 1497582, 16);
    load_txt_to_dram(N_PATH "module_16_c3k2/node_187_weight.txt", dram1, 1497598, 512);
    load_txt_to_dram(N_PATH "module_16_c3k2/node_187_bias.txt", dram1, 1498110, 16);
    load_txt_to_dram(N_PATH "module_16_c3k2/node_192_weight.txt", dram1, 1498126, 2304);
    load_txt_to_dram(N_PATH "module_16_c3k2/node_192_bias.txt", dram1, 1500430, 16);
    load_txt_to_dram(N_PATH "module_16_c3k2/node_195_weight.txt", dram1, 1500446, 2304);
    load_txt_to_dram(N_PATH "module_16_c3k2/node_195_bias.txt", dram1, 1502750, 16);
    load_txt_to_dram(N_PATH "module_16_c3k2/node_199_weight.txt", dram1, 1502766, 2304);
    load_txt_to_dram(N_PATH "module_16_c3k2/node_199_bias.txt", dram1, 1505070, 16);
    load_txt_to_dram(N_PATH "module_16_c3k2/node_202_weight.txt", dram1, 1505086, 2304);
    load_txt_to_dram(N_PATH "module_16_c3k2/node_202_bias.txt", dram1, 1507390, 16);
    load_txt_to_dram(N_PATH "module_16_c3k2/node_207_weight.txt", dram1, 1507406, 1024);
    load_txt_to_dram(N_PATH "module_16_c3k2/node_207_bias.txt", dram1, 1508430, 32);
    load_txt_to_dram(N_PATH "module_16_c3k2/node_211_weight.txt", dram1, 1508462, 6144);
    load_txt_to_dram(N_PATH "module_16_c3k2/node_211_bias.txt", dram1, 1514606, 64);
    load_txt_to_dram(N_PATH "module_17_conv/node_214_weight.txt", dram1, 1514670, 36864);
    load_txt_to_dram(N_PATH "module_17_conv/node_214_bias.txt", dram1, 1551534, 64);
    load_txt_to_dram(N_PATH "module_19_c3k2/node_226_weight.txt", dram1, 1567950, 24576);
    load_txt_to_dram(N_PATH "module_19_c3k2/node_226_bias.txt", dram1, 1592526, 128);
    load_txt_to_dram(N_PATH "module_19_c3k2/node_238_weight.txt", dram1, 1593367, 2048);
    load_txt_to_dram(N_PATH "module_19_c3k2/node_238_bias.txt", dram1, 1595415, 32);
    load_txt_to_dram(N_PATH "module_19_c3k2/node_239_weight.txt", dram1, 1595447, 2048);
    load_txt_to_dram(N_PATH "module_19_c3k2/node_239_bias.txt", dram1, 1597495, 32);
    load_txt_to_dram(N_PATH "module_19_c3k2/node_247_weight.txt", dram1, 1601687, 9216);
    load_txt_to_dram(N_PATH "module_19_c3k2/node_247_bias.txt", dram1, 1610903, 32);
    load_txt_to_dram(N_PATH "module_19_c3k2/node_253_weight.txt", dram1, 1611003, 9216);
    load_txt_to_dram(N_PATH "module_19_c3k2/node_253_bias.txt", dram1, 1620219, 32);
    load_txt_to_dram(N_PATH "module_19_c3k2/node_257_weight.txt", dram1, 1620251, 9216);
    load_txt_to_dram(N_PATH "module_19_c3k2/node_257_bias.txt", dram1, 1629467, 32);
    load_txt_to_dram(N_PATH "module_19_c3k2/node_260_weight.txt", dram1, 1629499, 9216);
    load_txt_to_dram(N_PATH "module_19_c3k2/node_260_bias.txt", dram1, 1638715, 32);
    load_txt_to_dram(N_PATH "module_19_c3k2/node_265_weight.txt", dram1, 1638747, 4096);
    load_txt_to_dram(N_PATH "module_19_c3k2/node_265_bias.txt", dram1, 1642843, 64);
    load_txt_to_dram(N_PATH "module_19_c3k2/node_269_weight.txt", dram1, 1642907, 24576);
    load_txt_to_dram(N_PATH "module_19_c3k2/node_269_bias.txt", dram1, 1667483, 128);
    load_txt_to_dram(N_PATH "module_20_conv/node_272_weight.txt", dram1, 1667611, 147456);
    load_txt_to_dram(N_PATH "module_20_conv/node_272_bias.txt", dram1, 1815067, 128);

    // --- MODULE HEAD ---
    load_txt_to_dram(H_PATH "module_22_c3k2/node_284_weight.txt", dram1, 1845499, 98304);
    load_txt_to_dram(H_PATH "module_22_c3k2/node_284_bias.txt", dram1, 1943803, 256);
    load_txt_to_dram(H_PATH "module_22_c3k2/node_296_weight.txt", dram1, 1944772, 73728);
    load_txt_to_dram(H_PATH "module_22_c3k2/node_296_bias.txt", dram1, 2018500, 64);
    load_txt_to_dram(H_PATH "module_22_c3k2/node_302_weight.txt", dram1, 2022724, 73728);
    load_txt_to_dram(H_PATH "module_22_c3k2/node_302_bias.txt", dram1, 2096452, 128);
    load_txt_to_dram(H_PATH "module_22_c3k2/node_309_weight.txt", dram1, 2096648, 32768);
    load_txt_to_dram(H_PATH "module_22_c3k2/node_309_bias.txt", dram1, 2129416, 256);
    load_txt_to_dram(H_PATH "module_22_c3k2/node_315_weight.txt", dram1, 2129684, 1152);
    load_txt_to_dram(H_PATH "module_22_c3k2/node_315_bias.txt", dram1, 2130836, 128);
    load_txt_to_dram(H_PATH "module_22_c3k2/node_322_weight.txt", dram1, 2130968, 16384);
    load_txt_to_dram(H_PATH "module_22_c3k2/node_322_bias.txt", dram1, 2147352, 128);
    load_txt_to_dram(H_PATH "module_22_c3k2/node_324_weight.txt", dram1, 2147480, 32768);
    load_txt_to_dram(H_PATH "module_22_c3k2/node_324_bias.txt", dram1, 2180248, 256);
    load_txt_to_dram(H_PATH "module_22_c3k2/node_327_weight.txt", dram1, 2180504, 32768);
    load_txt_to_dram(H_PATH "module_22_c3k2/node_327_bias.txt", dram1, 2213272, 128);
    load_txt_to_dram(H_PATH "module_22_c3k2/node_330_weight.txt", dram1, 2213400, 98304);
    load_txt_to_dram(H_PATH "module_22_c3k2/node_330_bias.txt", dram1, 2311704, 256);

    load_txt_to_dram(H_PATH "module_23_detect/node_215_weight.txt", dram1, 1551598, 9216);
    load_txt_to_dram(H_PATH "module_23_detect/node_215_bias.txt", dram1, 1560814, 16);
    load_txt_to_dram(H_PATH "module_23_detect/node_224_weight.txt", dram1, 1561470, 2304);
    load_txt_to_dram(H_PATH "module_23_detect/node_224_bias.txt", dram1, 1563774, 16);
    load_txt_to_dram(H_PATH "module_23_detect/node_233_weight.txt", dram1, 1592654, 64);
    load_txt_to_dram(H_PATH "module_23_detect/node_233_bias.txt", dram1, 1592718, 4);
    
    load_txt_to_dram(H_PATH "module_23_detect/node_216_weight.txt", dram1, 1560830, 576);
    load_txt_to_dram(H_PATH "module_23_detect/node_216_bias.txt", dram1, 1561406, 64);
    load_txt_to_dram(H_PATH "module_23_detect/node_225_weight.txt", dram1, 1563790, 4096);
    load_txt_to_dram(H_PATH "module_23_detect/node_225_bias.txt", dram1, 1567886, 64);
    load_txt_to_dram(H_PATH "module_23_detect/node_234_weight.txt", dram1, 1592722, 576);
    load_txt_to_dram(H_PATH "module_23_detect/node_234_bias.txt", dram1, 1593298, 64);
    load_txt_to_dram(H_PATH "module_23_detect/node_243_weight.txt", dram1, 1597527, 4096);
    load_txt_to_dram(H_PATH "module_23_detect/node_243_bias.txt", dram1, 1601623, 64);
    load_txt_to_dram(H_PATH "module_23_detect/node_250_weight.txt", dram1, 1610935, 64);
    load_txt_to_dram(H_PATH "module_23_detect/node_250_bias.txt", dram1, 1610999, 1);

    load_txt_to_dram(H_PATH "module_23_detect/node_273_weight.txt", dram1, 1815195, 18432);
    load_txt_to_dram(H_PATH "module_23_detect/node_273_bias.txt", dram1, 1833627, 16);
    load_txt_to_dram(H_PATH "module_23_detect/node_282_weight.txt", dram1, 1834923, 2304);
    load_txt_to_dram(H_PATH "module_23_detect/node_282_bias.txt", dram1, 1837227, 16);
    load_txt_to_dram(H_PATH "module_23_detect/node_291_weight.txt", dram1, 1944059, 64);
    load_txt_to_dram(H_PATH "module_23_detect/node_291_bias.txt", dram1, 1944123, 4);

    load_txt_to_dram(H_PATH "module_23_detect/node_274_weight.txt", dram1, 1833643, 1152);
    load_txt_to_dram(H_PATH "module_23_detect/node_274_bias.txt", dram1, 1834795, 128);
    load_txt_to_dram(H_PATH "module_23_detect/node_283_weight.txt", dram1, 1837243, 8192);
    load_txt_to_dram(H_PATH "module_23_detect/node_283_bias.txt", dram1, 1845435, 64);
    load_txt_to_dram(H_PATH "module_23_detect/node_292_weight.txt", dram1, 1944127, 576);
    load_txt_to_dram(H_PATH "module_23_detect/node_292_bias.txt", dram1, 1944703, 64);
    load_txt_to_dram(H_PATH "module_23_detect/node_299_weight.txt", dram1, 2018564, 4096);
    load_txt_to_dram(H_PATH "module_23_detect/node_299_bias.txt", dram1, 2022660, 64);
    load_txt_to_dram(H_PATH "module_23_detect/node_305_weight.txt", dram1, 2096580, 64);
    load_txt_to_dram(H_PATH "module_23_detect/node_305_bias.txt", dram1, 2096644, 1);

    load_txt_to_dram(H_PATH "module_23_detect/node_333_weight.txt", dram1, 2311960, 36864);
    load_txt_to_dram(H_PATH "module_23_detect/node_333_bias.txt", dram1, 2348824, 16);
    load_txt_to_dram(H_PATH "module_23_detect/node_339_weight.txt", dram1, 2351400, 2304);
    load_txt_to_dram(H_PATH "module_23_detect/node_339_bias.txt", dram1, 2353704, 16);
    load_txt_to_dram(H_PATH "module_23_detect/node_345_weight.txt", dram1, 2370168, 64);
    load_txt_to_dram(H_PATH "module_23_detect/node_345_bias.txt", dram1, 2370232, 4);

    load_txt_to_dram(H_PATH "module_23_detect/node_334_weight.txt", dram1, 2348840, 2304);
    load_txt_to_dram(H_PATH "module_23_detect/node_334_bias.txt", dram1, 2351144, 256);
    load_txt_to_dram(H_PATH "module_23_detect/node_340_weight.txt", dram1, 2353720, 16384);
    load_txt_to_dram(H_PATH "module_23_detect/node_340_bias.txt", dram1, 2370104, 64);
    load_txt_to_dram(H_PATH "module_23_detect/node_346_weight.txt", dram1, 2370236, 576);
    load_txt_to_dram(H_PATH "module_23_detect/node_346_bias.txt", dram1, 2370812, 64);
    load_txt_to_dram(H_PATH "module_23_detect/node_351_weight.txt", dram1, 2370879, 4096);
    load_txt_to_dram(H_PATH "module_23_detect/node_351_bias.txt", dram1, 2374975, 64);
    load_txt_to_dram(H_PATH "module_23_detect/node_359_weight.txt", dram1, 2408645, 64);
    load_txt_to_dram(H_PATH "module_23_detect/node_359_bias.txt", dram1, 2408709, 1);

    // Anchors & Strides
    load_txt_to_dram(H_PATH "module_23_detect/node_355_weight.txt", dram1, 2375045, 16800);
    load_txt_to_dram(H_PATH "module_23_detect/node_356_weight.txt", dram1, 2391845, 16800);
    load_txt_to_dram(H_PATH "module_23_detect/node_360_weight.txt", dram1, 2408710, 8400);


    printf("\n--- BƯỚC 2: RUN BACKBONE, NECK, HEAD MODULES ---\n");
    module_0_conv_P1(dram1, dram2, input_buffer);
    module_1_conv_P2(dram1, dram2);
    module_2_c3k2_P2(dram1, dram2);
    module_3_conv_P3(dram1, dram2);
    module_4_c3k2_P3(dram1, dram2);
    module_5_conv_P4(dram1, dram2);
    module_6_c3k2_P4(dram1, dram2);
    module_7_conv_P5(dram1, dram2);
    module_8_c3k2_P5(dram1, dram2);
    
    module_9_sppf(dram1, dram2);
    module_10_c2psa(dram1, dram2);
    module_11_upsample(dram1, dram2);
    module_12_concat(dram1, dram2);
    module_13_c3k2(dram1, dram2);
    module_14_upsample(dram1, dram2);
    module_15_concat(dram1, dram2);
    module_16_c3k2_detect_P3(dram1, dram2);
    module_17_conv_downsample(dram1, dram2);
    module_18_concat(dram1, dram2);
    module_19_c3k2_detect_P4(dram1, dram2);
    module_20_conv_downsample(dram1, dram2);
    
    module_21_concat(dram1, dram2);
    module_22_c3k2_detect_P5(dram1, dram2);
    module_23_detect_head(dram1, dram2);

    printf("\n--- BƯỚC 3: DÒ LỖI TỪNG NODE ---\n");

    NodeMap nodes_map[] = {
        // --- Backbone ---
        {0, 0, 1638400, "Node 0: Conv"}, {1, 1638400, 1638400, "Node 1: Sigmoid"}, {2, 3276800, 1638400, "Node 2: Mul"},
        {3, 4915200, 819200, "Node 3: Conv"}, {4, 5734400, 819200, "Node 4: Sigmoid"}, {5, 6553600, 819200, "Node 5: Mul"},
        {6, 7372800, 819200, "Node 6: Conv"}, {7, 8192000, 819200, "Node 7: Sigmoid"}, {8, 9011200, 819200, "Node 8: Mul"},
        {9, 9011200, 409600, "Node 9: Split"}, {10, 10240000, 204800, "Node 10: Conv"}, {11, 10444800, 204800, "Node 11: Sigmoid"},
        {12, 10649600, 204800, "Node 12: Mul"}, {13, 10854400, 409600, "Node 13: Conv"}, {14, 11264000, 409600, "Node 14: Sigmoid"},
        {15, 11673600, 409600, "Node 15: Mul"}, {16, 12083200, 409600, "Node 16: Add"}, {17, 12492800, 1228800, "Node 17: Concat"},
        {18, 13721600, 1638400, "Node 18: Conv"}, {19, 15360000, 1638400, "Node 19: Sigmoid"}, {20, 16998400, 1638400, "Node 20: Mul"},
        {21, 18636800, 409600, "Node 21: Conv"}, {22, 19046400, 409600, "Node 22: Sigmoid"}, {23, 19456000, 409600, "Node 23: Mul"},
        {24, 19865600, 409600, "Node 24: Conv"}, {25, 20275200, 409600, "Node 25: Sigmoid"}, {26, 20684800, 409600, "Node 26: Mul"},
        {27, 20684800, 204800, "Node 27: Split"}, {28, 21299200, 102400, "Node 28: Conv"}, {29, 21401600, 102400, "Node 29: Sigmoid"},
        {30, 21504000, 102400, "Node 30: Mul"}, {31, 21606400, 204800, "Node 31: Conv"}, {32, 21811200, 204800, "Node 32: Sigmoid"},
        {33, 22016000, 204800, "Node 33: Mul"}, {34, 22220800, 204800, "Node 34: Add"}, {35, 22425600, 614400, "Node 35: Concat"},
        {36, 23040000, 819200, "Node 36: Conv"}, {37, 23859200, 819200, "Node 37: Sigmoid"}, {38, 24678400, 819200, "Node 38: Mul"},
        {39, 25497600, 204800, "Node 39: Conv"}, {40, 25702400, 204800, "Node 40: Sigmoid"}, {41, 25907200, 204800, "Node 41: Mul"},
        {42, 26112000, 204800, "Node 42: Conv"}, {43, 26316800, 204800, "Node 43: Sigmoid"}, {44, 26521600, 204800, "Node 44: Mul"},
        {45, 26521600, 102400, "Node 45: Split"}, {46, 26828800, 51200, "Node 46: Conv"}, {47, 26880000, 51200, "Node 47: Conv"},
        {48, 26931200, 51200, "Node 48: Sigmoid"}, {49, 26982400, 51200, "Node 49: Sigmoid"}, {50, 27033600, 51200, "Node 50: Mul"},
        {51, 27084800, 51200, "Node 51: Mul"}, {52, 27136000, 51200, "Node 52: Conv"}, {53, 27187200, 51200, "Node 53: Sigmoid"},
        {54, 27238400, 51200, "Node 54: Mul"}, {55, 27289600, 51200, "Node 55: Conv"}, {56, 27340800, 51200, "Node 56: Sigmoid"},
        {57, 27392000, 51200, "Node 57: Mul"}, {58, 27443200, 51200, "Node 58: Add"}, {59, 27494400, 51200, "Node 59: Conv"},
        {60, 27545600, 51200, "Node 60: Sigmoid"}, {61, 27596800, 51200, "Node 61: Mul"}, {62, 27648000, 51200, "Node 62: Conv"},
        {63, 27699200, 51200, "Node 63: Sigmoid"}, {64, 27750400, 51200, "Node 64: Mul"}, {65, 27801600, 51200, "Node 65: Add"},
        {66, 27852800, 102400, "Node 66: Concat"}, {67, 27955200, 102400, "Node 67: Conv"}, {68, 28057600, 102400, "Node 68: Sigmoid"},
        {69, 28160000, 102400, "Node 69: Mul"}, {70, 28262400, 307200, "Node 70: Concat"}, {71, 28569600, 204800, "Node 71: Conv"},
        {72, 28774400, 204800, "Node 72: Sigmoid"}, {73, 28979200, 204800, "Node 73: Mul"}, {74, 29184000, 102400, "Node 74: Conv"},
        {75, 29286400, 102400, "Node 75: Sigmoid"}, {76, 29388800, 102400, "Node 76: Mul"}, {77, 29491200, 102400, "Node 77: Conv"},
        {78, 29593600, 102400, "Node 78: Sigmoid"}, {79, 29696000, 102400, "Node 79: Mul"}, {80, 29696000, 51200, "Node 80: Split"},
        {81, 29849600, 25600, "Node 81: Conv"}, {82, 29875200, 25600, "Node 82: Conv"}, {83, 29900800, 25600, "Node 83: Sigmoid"},
        {84, 29926400, 25600, "Node 84: Sigmoid"}, {85, 29952000, 25600, "Node 85: Mul"}, {86, 29977600, 25600, "Node 86: Mul"},
        {87, 30003200, 25600, "Node 87: Conv"}, {88, 30028800, 25600, "Node 88: Sigmoid"}, {89, 30054400, 25600, "Node 89: Mul"},
        {90, 30080000, 25600, "Node 90: Conv"}, {91, 30105600, 25600, "Node 91: Sigmoid"}, {92, 30131200, 25600, "Node 92: Mul"},
        {93, 30156800, 25600, "Node 93: Add"}, {94, 30182400, 25600, "Node 94: Conv"}, {95, 30208000, 25600, "Node 95: Sigmoid"},
        {96, 30233600, 25600, "Node 96: Mul"}, {97, 30259200, 25600, "Node 97: Conv"}, {98, 30284800, 25600, "Node 98: Sigmoid"},
        {99, 30310400, 25600, "Node 99: Mul"}, {100, 30336000, 25600, "Node 100: Add"}, {101, 30361600, 51200, "Node 101: Concat"},
        {102, 30412800, 51200, "Node 102: Conv"}, {103, 30464000, 51200, "Node 103: Sigmoid"}, {104, 30515200, 51200, "Node 104: Mul"},
        {105, 30566400, 153600, "Node 105: Concat"}, {106, 30720000, 102400, "Node 106: Conv"}, {107, 30822400, 102400, "Node 107: Sigmoid"},
        {108, 30924800, 102400, "Node 108: Mul"},
        
        // --- Neck ---
        {109, 31027200, 51200, "Node 109: Conv"}, {110, 31078400, 51200, "Node 110: MaxPool"},
        {111, 31129600, 51200, "Node 111: MaxPool"}, {112, 31180800, 51200, "Node 112: MaxPool"},
        {113, 31232000, 204800, "Node 113: Concat"}, {114, 31436800, 102400, "Node 114: Conv"},
        {115, 31539200, 102400, "Node 115: Sigmoid"}, {116, 31641600, 102400, "Node 116: Mul"},
        {117, 31744000, 102400, "Node 117: Add"}, {118, 31846400, 102400, "Node 118: Conv"},
        {119, 31948800, 102400, "Node 119: Sigmoid"}, {120, 32051200, 102400, "Node 120: Mul"},
        {121, 32051200, 51200, "Node 121: Split"}, {122, 32204800, 102400, "Node 122: Conv"},
        {123, 32307200, 102400, "Node 123: Reshape"}, {124, 32409600, 25600, "Node 124: Split"},
        {125, 32435200, 25600, "Node 125: Mul"}, {126, 32460800, 51200, "Node 126: Reshape"},
        {127, 32512000, 25600, "Node 127: Transpose"}, {128, 32537600, 51200, "Node 128: Conv"},
        {129, 32588800, 320000, "Node 129: MatMul"}, {130, 32908800, 320000, "Node 130: Softmax"},
        {131, 33228800, 320000, "Node 131: Transpose"}, {132, 33548800, 51200, "Node 132: MatMul"},
        {133, 33600000, 51200, "Node 133: Reshape"}, {134, 33651200, 51200, "Node 134: Add"},
        {135, 33702400, 51200, "Node 135: Conv"}, {136, 33753600, 51200, "Node 136: Add"},
        {137, 33804800, 102400, "Node 137: Conv"}, {138, 33907200, 102400, "Node 138: Sigmoid"},
        {139, 34009600, 102400, "Node 139: Mul"}, {140, 34112000, 51200, "Node 140: Conv"},
        {141, 34163200, 51200, "Node 141: Add"}, {142, 34214400, 102400, "Node 142: Concat"},
        {143, 34316800, 102400, "Node 143: Conv"}, {144, 34419200, 102400, "Node 144: Sigmoid"},
        {145, 34521600, 102400, "Node 145: Mul"}, {146, 34624000, 409600, "Node 146: Resize"},
        {147, 35033600, 614400, "Node 147: Concat"}, {148, 35648000, 204800, "Node 148: Conv"},
        {149, 35852800, 204800, "Node 149: Sigmoid"}, {150, 36057600, 204800, "Node 150: Mul"},
        {151, 36057600, 102400, "Node 151: Split"}, {152, 36364800, 51200, "Node 152: Conv"},
        {153, 36416000, 51200, "Node 153: Conv"}, {154, 36467200, 51200, "Node 154: Sigmoid"},
        {155, 36518400, 51200, "Node 155: Sigmoid"}, {156, 36569600, 51200, "Node 156: Mul"},
        {157, 36620800, 51200, "Node 157: Mul"}, {158, 36672000, 51200, "Node 158: Conv"},
        {159, 36723200, 51200, "Node 159: Sigmoid"}, {160, 36774400, 51200, "Node 160: Mul"},
        {161, 36825600, 51200, "Node 161: Conv"}, {162, 36876800, 51200, "Node 162: Sigmoid"},
        {163, 36928000, 51200, "Node 163: Mul"}, {164, 36979200, 51200, "Node 164: Add"},
        {165, 37030400, 51200, "Node 165: Conv"}, {166, 37081600, 51200, "Node 166: Sigmoid"},
        {167, 37132800, 51200, "Node 167: Mul"}, {168, 37184000, 51200, "Node 168: Conv"},
        {169, 37235200, 51200, "Node 169: Sigmoid"}, {170, 37286400, 51200, "Node 170: Mul"},
        {171, 37337600, 51200, "Node 171: Add"}, {172, 37388800, 102400, "Node 172: Concat"},
        {173, 37491200, 102400, "Node 173: Conv"}, {174, 37593600, 102400, "Node 174: Sigmoid"},
        {175, 37696000, 102400, "Node 175: Mul"}, {176, 37798400, 307200, "Node 176: Concat"},
        {177, 38105600, 204800, "Node 177: Conv"}, {178, 38310400, 204800, "Node 178: Sigmoid"},
        {179, 38515200, 204800, "Node 179: Mul"}, {180, 38720000, 819200, "Node 180: Resize"},
        {181, 39539200, 1638400, "Node 181: Concat"}, {182, 41177600, 409600, "Node 182: Conv"},
        {183, 41587200, 409600, "Node 183: Sigmoid"}, {184, 41996800, 409600, "Node 184: Mul"},
        {185, 41996800, 204800, "Node 185: Split"}, {186, 42611200, 102400, "Node 186: Conv"},
        {187, 42713600, 102400, "Node 187: Conv"}, {188, 42816000, 102400, "Node 188: Sigmoid"},
        {189, 42918400, 102400, "Node 189: Sigmoid"}, {190, 43020800, 102400, "Node 190: Mul"},
        {191, 43123200, 102400, "Node 191: Mul"}, {192, 43225600, 102400, "Node 192: Conv"},
        {193, 43328000, 102400, "Node 193: Sigmoid"}, {194, 43430400, 102400, "Node 194: Mul"},
        {195, 43532800, 102400, "Node 195: Conv"}, {196, 43635200, 102400, "Node 196: Sigmoid"},
        {197, 43737600, 102400, "Node 197: Mul"}, {198, 43840000, 102400, "Node 198: Add"},
        {199, 43942400, 102400, "Node 199: Conv"}, {200, 44044800, 102400, "Node 200: Sigmoid"},
        {201, 44147200, 102400, "Node 201: Mul"}, {202, 44249600, 102400, "Node 202: Conv"},
        {203, 44352000, 102400, "Node 203: Sigmoid"}, {204, 44454400, 102400, "Node 204: Mul"},
        {205, 44556800, 102400, "Node 205: Add"}, {206, 44659200, 204800, "Node 206: Concat"},
        {207, 44864000, 204800, "Node 207: Conv"}, {208, 45068800, 204800, "Node 208: Sigmoid"},
        {209, 45273600, 204800, "Node 209: Mul"}, {210, 45478400, 614400, "Node 210: Concat"},
        {211, 46092800, 409600, "Node 211: Conv"}, {212, 46502400, 409600, "Node 212: Sigmoid"},
        {213, 46912000, 409600, "Node 213: Mul"}, {214, 47321600, 102400, "Node 214: Conv"},
        {217, 47936000, 102400, "Node 217: Sigmoid"}, {220, 48550400, 102400, "Node 220: Mul"},
        {223, 49164800, 307200, "Node 223: Concat"}, {226, 49984000, 204800, "Node 226: Conv"},
        {229, 50700800, 204800, "Node 229: Sigmoid"}, {232, 51417600, 204800, "Node 232: Mul"},
        {235, 51417600, 102400, "Node 235: Split"}, {238, 52595200, 51200, "Node 238: Conv"},
        {239, 52646400, 51200, "Node 239: Conv"}, {241, 53107200, 51200, "Node 241: Sigmoid"},
        {244, 53619200, 51200, "Node 244: Mul"}, {247, 54131200, 51200, "Node 247: Conv"},
        {249, 54592000, 51200, "Node 249: Sigmoid"}, {251, 54649600, 51200, "Node 251: Mul"},
        {253, 54707200, 51200, "Node 253: Conv"}, {254, 54758400, 51200, "Node 254: Sigmoid"},
        {255, 54809600, 51200, "Node 255: Mul"}, {256, 54860800, 51200, "Node 256: Add"},
        {257, 54912000, 51200, "Node 257: Conv"}, {258, 54963200, 51200, "Node 258: Sigmoid"},
        {259, 55014400, 51200, "Node 259: Mul"}, {260, 55065600, 51200, "Node 260: Conv"},
        {261, 55116800, 51200, "Node 261: Sigmoid"}, {262, 55168000, 51200, "Node 262: Mul"},
        {263, 55219200, 51200, "Node 263: Add"}, {264, 55270400, 102400, "Node 264: Concat"},
        {265, 55372800, 102400, "Node 265: Conv"}, {266, 55475200, 102400, "Node 266: Sigmoid"},
        {267, 55577600, 102400, "Node 267: Mul"}, {268, 55680000, 307200, "Node 268: Concat"},
        {269, 55987200, 204800, "Node 269: Conv"}, {270, 56192000, 204800, "Node 270: Sigmoid"},
        {271, 56396800, 204800, "Node 271: Mul"}, {272, 56601600, 51200, "Node 272: Conv"},
        {275, 56883200, 51200, "Node 275: Sigmoid"}, {278, 57164800, 51200, "Node 278: Mul"},

        // --- Head (Module 21, 22) ---
        {281, 57446400, 153600, "Node 281: Concat"}, {284, 57728000, 102400, "Node 284: Conv"},
        {293, 58400000, 51200, "Node 293: Split"}, {296, 58560000, 25600, "Node 296: Conv"},
        {302, 58944000, 51200, "Node 302: Conv"}, {308, 59203200, 51200, "Node 308: Add"},
        {309, 59254400, 102400, "Node 309: Conv"}, {310, 59356800, 102400, "Node 310: Reshape"},
        {311, 59459200, 25600, "Node 311: Split"}, {312, 59484800, 25600, "Node 312: Mul"},
        {314, 59561600, 25600, "Node 314: Transpose"}, {315, 59587200, 51200, "Node 315: Conv"},
        {316, 59638400, 320000, "Node 316: MatMul"}, {317, 59958400, 320000, "Node 317: Softmax"},
        {318, 60278400, 320000, "Node 318: Transpose"}, {319, 60598400, 51200, "Node 319: MatMul"},
        {320, 60649600, 51200, "Node 320: Reshape"}, {321, 60700800, 51200, "Node 321: Add"},
        {322, 60752000, 51200, "Node 322: Conv"}, {323, 60803200, 51200, "Node 323: Add"},
        {324, 60854400, 102400, "Node 324: Conv"}, {327, 61161600, 51200, "Node 327: Conv"},
        {328, 61212800, 51200, "Node 328: Add"}, {329, 61264000, 153600, "Node 329: Concat"},
        {330, 61417600, 102400, "Node 330: Conv"},

        // --- Head (Module 23 Detect Branches) ---
        // P3
        {215, 47424000, 102400, "Node 215: Conv_Box"}, {224, 49472000, 102400, "Node 224: Conv_Box"},
        {233, 51622400, 25600, "Node 233: Conv_Box"}, {216, 47526400, 409600, "Node 216: Conv_Cls"},
        {225, 49574400, 409600, "Node 225: Conv_Cls"}, {234, 51648000, 409600, "Node 234: Conv_Cls"},
        {243, 53209600, 409600, "Node 243: Conv_Cls"}, {250, 54643200, 6400, "Node 250: Conv_Cls"},
        
        // P4
        {273, 56652800, 25600, "Node 273: Conv_Box"}, {282, 57600000, 25600, "Node 282: Conv_Box"},
        {291, 58291200, 6400, "Node 291: Conv_Box"}, {274, 56678400, 204800, "Node 274: Conv_Cls"},
        {283, 57625600, 102400, "Node 283: Conv_Cls"}, {292, 58297600, 102400, "Node 292: Conv_Cls"},
        {299, 58713600, 102400, "Node 299: Conv_Cls"}, {305, 59148800, 1600, "Node 305: Conv_Cls"},
        
        // P5
        {333, 61724800, 6400, "Node 333: Conv_Box"}, {339, 62051200, 6400, "Node 339: Conv_Box"},
        {345, 62147200, 1600, "Node 345: Conv_Box"}, {334, 61731200, 102400, "Node 334: Conv_Cls"},
        {340, 62057600, 25600, "Node 340: Conv_Cls"}, {346, 62148800, 25600, "Node 346: Conv_Cls"},
        {351, 62260800, 25600, "Node 351: Conv_Cls"}, {359, 62438400, 400, "Node 359: Conv_Cls"},

        // DFL & Post Processing (NMS prep)
        {349, 62201600, 33600, "Node 349: Concat Box"}, {362, 62472800, 8400, "Node 362: Concat Cls"},
        {364, 62489600, 42000, "Node 364: Concat B+C"}, {365, 62531600, 42000, "Node 365: Transpose"},
        {366, 62573600, 33600, "Node 366: Split Box"}, {367, 62607200, 8400, "Node 367: ReduceMax"}
    };

    int num_nodes = sizeof(nodes_map) / sizeof(NodeMap);
    char filepath[256];

    for (int i = 0; i < num_nodes; i++) {
        sprintf(filepath, "golden_outputs/node_%d_golden.txt", nodes_map[i].id);
        int status = verify_golden(&dram2[nodes_map[i].offset], filepath, nodes_map[i].size, nodes_map[i].name);
        
        if (status != 0) {
            printf("\n[!!!] Quá trình dò lỗi đã DỪNG LẠI tại %s do phát hiện sai số logic.\n", nodes_map[i].name);
            break; 
        }
    }


    // ========================================================================
    // BƯỚC 4: XUẤT OUTPUT TỪNG NODE TỪ DRAM RA THƯ MỤC BACKBONE / NECK / HEAD
    // ========================================================================
    printf("\n--- BƯỚC 4: ĐANG GHI OUTPUT TỪNG NODE RA TỆP TEXT ---\n");
    
    // Tạo cấu trúc thư mục nếu chưa có
    system("mkdir -p output_nodes/backbone output_nodes/neck output_nodes/head");

    for (int i = 0; i < num_nodes; i++) {
        char out_path[256];
        // Phân loại thư mục dựa theo ID của Node
        if (nodes_map[i].id <= 108) {
            sprintf(out_path, "output_nodes/backbone/node_%d_output.txt", nodes_map[i].id);
        } else if (nodes_map[i].id <= 278) {
            sprintf(out_path, "output_nodes/neck/node_%d_output.txt", nodes_map[i].id);
        } else {
            sprintf(out_path, "output_nodes/head/node_%d_output.txt", nodes_map[i].id);
        }

        FILE *f_out = fopen(out_path, "w");
        if (f_out) {
            float *ptr = &dram2[nodes_map[i].offset];
            for (int j = 0; j < nodes_map[i].size; j++) {
                fprintf(f_out, "%.6f ", ptr[j]);
                if ((j + 1) % 10 == 0) fprintf(f_out, "\n");
            }
            fclose(f_out);
        }
    }
    printf("✅ Đã ghi thành công toàn bộ node output vào thư mục 'output_nodes/'!\n");

    free_dram();
    return 0;
}