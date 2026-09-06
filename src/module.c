#include "module.h"
#include "node.h"
#include "load.h"

// ============================================================================
// MODULE 0: Conv (P1, k=3 s=2)
// ============================================================================
void module_0_conv_P1(const float *dram1, float *dram2, const float *input_buffer) {
    
    // --------------------------------------------------------
    // NODE 0: Conv
    // --------------------------------------------------------
    // Cấp phát BRAM tĩnh cho Node 0
    static float n0_weight[432];
    static float n0_bias[16];
    static float n0_output[1638400];
    
    // Load tham số từ DRAM_1 vào BRAM
    load_weight_bias(dram1, 0, n0_weight, 432);        // Địa chỉ: 0 -> 431
    load_weight_bias(dram1, 432, n0_bias, 16);         // Địa chỉ: 432 -> 447
    
    // Thực thi Compute (Lấy input_buffer làm đầu vào trực tiếp)
    node_conv((float *)input_buffer, n0_weight, n0_bias, n0_output, 1, 3, 640, 640, 16, 3, 2, 1);
    
    // Store OFM ra DRAM_2
    store_ofm(dram2, 0, n0_output, 1638400);           // Địa chỉ: 0 -> 1638399


    // --------------------------------------------------------
    // NODE 1: Sigmoid
    // --------------------------------------------------------
    // Cấp phát BRAM tĩnh cho Node 1
    static float n1_input[1638400];
    static float n1_output[1638400];
    
    // Load IFM từ DRAM_2 (Chính là output của Node 0 vừa ghi)
    load_ifm(dram2, 0, n1_input, 1638400);             // Địa chỉ: 0 -> 1638399
    
    // Thực thi Compute
    node_sigmoid(n1_input, n1_output, 1638400);
    
    // Store OFM ra DRAM_2
    store_ofm(dram2, 1638400, n1_output, 1638400);     // Địa chỉ: 1638400 -> 3276799


    // --------------------------------------------------------
    // NODE 2: Mul (SiLU = Input * Sigmoid)
    // --------------------------------------------------------
    // Cấp phát BRAM tĩnh cho Node 2
    static float n2_input1[1638400];
    static float n2_input2[1638400];
    static float n2_output[1638400];
    
    // Load IFM từ DRAM_2
    load_ifm(dram2, 0, n2_input1, 1638400);            // Nhánh 1: Lấy lại Output của Node 0
    load_ifm(dram2, 1638400, n2_input2, 1638400);      // Nhánh 2: Lấy Output của Node 1 (Sigmoid)
    
    // Thực thi Compute
    node_mul(n2_input1, n2_input2, n2_output, 1638400);
    
    // Store OFM ra DRAM_2
    store_ofm(dram2, 3276800, n2_output, 1638400);     // Địa chỉ: 3276800 -> 4915199
}

// ============================================================================
// MODULE 1: Conv (P2, k=3 s=2)
// Input: 1x16x320x320 | Output: 1x32x160x160 | Params: 4,640
// ============================================================================
void module_1_conv_P2(const float *dram1, float *dram2) {
    
    // --------------------------------------------------------
    // NODE 3: Conv
    // --------------------------------------------------------
    // Cấp phát BRAM tĩnh cho Node 3
    static float n3_input[1638400];    // Kích thước từ output Node 2
    static float n3_weight[4608];
    static float n3_bias[32];
    static float n3_output[819200];
    
    // Load IFM từ DRAM_2 (Output của Node 2: địa chỉ 3276800 -> 4915199)
    load_ifm(dram2, 3276800, n3_input, 1638400);

    // Load tham số từ DRAM_1 vào BRAM
    load_weight_bias(dram1, 448, n3_weight, 4608);     // Địa chỉ: 448 -> 5055
    load_weight_bias(dram1, 5056, n3_bias, 32);        // Địa chỉ: 5056 -> 5087
    
    // Thực thi Compute
    node_conv(n3_input, n3_weight, n3_bias, n3_output, 1, 16, 320, 320, 32, 3, 2, 1);
    
    // Store OFM ra DRAM_2
    store_ofm(dram2, 4915200, n3_output, 819200);      // Địa chỉ: 4915200 -> 5734399


    // --------------------------------------------------------
    // NODE 4: Sigmoid
    // --------------------------------------------------------
    // Cấp phát BRAM tĩnh cho Node 4
    static float n4_input[819200];
    static float n4_output[819200];
    
    // Load IFM từ DRAM_2 (Output của Node 3 vừa ghi)
    load_ifm(dram2, 4915200, n4_input, 819200);        // Địa chỉ: 4915200 -> 5734399
    
    // Thực thi Compute
    node_sigmoid(n4_input, n4_output, 819200);
    
    // Store OFM ra DRAM_2
    store_ofm(dram2, 5734400, n4_output, 819200);      // Địa chỉ: 5734400 -> 6553599


    // --------------------------------------------------------
    // NODE 5: Mul (SiLU = Input * Sigmoid)
    // --------------------------------------------------------
    // Cấp phát BRAM tĩnh cho Node 5
    static float n5_input1[819200];
    static float n5_input2[819200];
    static float n5_output[819200];
    
    // Load IFM từ DRAM_2
    load_ifm(dram2, 4915200, n5_input1, 819200);       // Nhánh 1: Lấy lại Output của Node 3
    load_ifm(dram2, 5734400, n5_input2, 819200);       // Nhánh 2: Lấy Output của Node 4 (Sigmoid)
    
    // Thực thi Compute
    node_mul(n5_input1, n5_input2, n5_output, 819200);
    
    // Store OFM ra DRAM_2
    store_ofm(dram2, 6553600, n5_output, 819200);      // Địa chỉ: 6553600 -> 7372799
}

// ============================================================================
// MODULE 2: C3k2 (P2)
// Input: 1x32x160x160 | Output: 1x64x160x160
// ============================================================================
void module_2_c3k2_P2(const float *dram1, float *dram2) {
    
    // --------------------------------------------------------
    // NODE 6: Conv (cv1)
    // --------------------------------------------------------
    static float n6_input[819200];     // Lấy Output của Module 1 (Node 5)
    static float n6_weight[1024];
    static float n6_bias[32];
    static float n6_output[819200];
    
    // Địa chỉ Output Node 5 ở module trước kết thúc tại 7372799, nên input bắt đầu từ 6553600
    load_ifm(dram2, 6553600, n6_input, 819200);

    // Load Weights & Biases từ DRAM_1[cite: 6]
    load_weight_bias(dram1, 5088, n6_weight, 1024);    // Địa chỉ: 5088 -> 6111[cite: 6]
    load_weight_bias(dram1, 6112, n6_bias, 32);        // Địa chỉ: 6112 -> 6143[cite: 6]
    
    // Thực thi Conv (k=1, s=1, p=0)[cite: 6]
    node_conv(n6_input, n6_weight, n6_bias, n6_output, 1, 32, 160, 160, 32, 1, 1, 0);
    store_ofm(dram2, 7372800, n6_output, 819200);      // Địa chỉ: 7372800 -> 8191999[cite: 6]


    // --------------------------------------------------------
    // NODE 7: Sigmoid
    // --------------------------------------------------------
    static float n7_input[819200];
    static float n7_output[819200];
    
    load_ifm(dram2, 7372800, n7_input, 819200);        //[cite: 6]
    node_sigmoid(n7_input, n7_output, 819200);
    store_ofm(dram2, 8192000, n7_output, 819200);      // Địa chỉ: 8192000 -> 9011199[cite: 6]


    // --------------------------------------------------------
    // NODE 8: Mul (cv1 output)
    // --------------------------------------------------------
    static float n8_input1[819200];
    static float n8_input2[819200];
    static float n8_output[819200];
    
    load_ifm(dram2, 7372800, n8_input1, 819200);       // Lấy Conv output[cite: 6]
    load_ifm(dram2, 8192000, n8_input2, 819200);       // Lấy Sigmoid output[cite: 6]
    
    node_mul(n8_input1, n8_input2, n8_output, 819200);
    store_ofm(dram2, 9011200, n8_output, 819200);      // Địa chỉ: 9011200 -> 9830399[cite: 6]


    // --------------------------------------------------------
    // NODE 9: Split
    // --------------------------------------------------------
    // Bảng DRAM_2 cấp phát khối 409600 element ở offset 9830400 cho Node 9[cite: 6].
    // Đây chính là Split_output_1 (Nửa sau) để nạp vào hệ thống Bottleneck (Node 10).
    // Split_output_0 (Nửa đầu) vẫn nằm an toàn ở 9011200 (Output Node 8).
    static float n9_input[819200];
    static float n9_split0[409600];
    static float n9_split1[409600];
    
    load_ifm(dram2, 9011200, n9_input, 819200);        //[cite: 6]
    
    float *splits[2] = {n9_split0, n9_split1};
    node_split(n9_input, splits, 2, 409600);
    
    store_ofm(dram2, 9830400, n9_split1, 409600);      // Lưu nửa sau vào: 9830400 -> 10239999[cite: 6]


    // --------------------------------------------------------
    // NODE 10: Conv (m.0.cv1)
    // --------------------------------------------------------
    static float n10_input[409600];
    static float n10_weight[1152];
    static float n10_bias[8];
    static float n10_output[204800];
    
    load_ifm(dram2, 9830400, n10_input, 409600);       // Lấy Input từ Split[cite: 6]
    
    load_weight_bias(dram1, 6146, n10_weight, 1152);   // Địa chỉ: 6146 -> 7297[cite: 6]
    load_weight_bias(dram1, 7298, n10_bias, 8);        // Địa chỉ: 7298 -> 7305[cite: 6]
    
    // Thực thi Conv (k=3, s=1, p=1)[cite: 6]
    node_conv(n10_input, n10_weight, n10_bias, n10_output, 1, 16, 160, 160, 8, 3, 1, 1);
    store_ofm(dram2, 10240000, n10_output, 204800);    // Địa chỉ: 10240000 -> 10444799[cite: 6]


    // --------------------------------------------------------
    // NODE 11: Sigmoid
    // --------------------------------------------------------
    static float n11_input[204800];
    static float n11_output[204800];
    
    load_ifm(dram2, 10240000, n11_input, 204800);      //[cite: 6]
    node_sigmoid(n11_input, n11_output, 204800);
    store_ofm(dram2, 10444800, n11_output, 204800);    // Địa chỉ: 10444800 -> 10649599[cite: 6]


    // --------------------------------------------------------
    // NODE 12: Mul (m.0.cv1 output)
    // --------------------------------------------------------
    static float n12_input1[204800];
    static float n12_input2[204800];
    static float n12_output[204800];
    
    load_ifm(dram2, 10240000, n12_input1, 204800);     //[cite: 6]
    load_ifm(dram2, 10444800, n12_input2, 204800);     //[cite: 6]
    
    node_mul(n12_input1, n12_input2, n12_output, 204800);
    store_ofm(dram2, 10649600, n12_output, 204800);    // Địa chỉ: 10649600 -> 10854399[cite: 6]


    // --------------------------------------------------------
    // NODE 13: Conv (m.0.cv2)
    // --------------------------------------------------------
    static float n13_input[204800];
    static float n13_weight[1152];
    static float n13_bias[16];
    static float n13_output[409600];
    
    load_ifm(dram2, 10649600, n13_input, 204800);      //[cite: 6]
    
    load_weight_bias(dram1, 7306, n13_weight, 1152);   // Địa chỉ: 7306 -> 8457[cite: 6]
    load_weight_bias(dram1, 8458, n13_bias, 16);       // Địa chỉ: 8458 -> 8473[cite: 6]
    
    node_conv(n13_input, n13_weight, n13_bias, n13_output, 1, 8, 160, 160, 16, 3, 1, 1);
    store_ofm(dram2, 10854400, n13_output, 409600);    // Địa chỉ: 10854400 -> 11263999[cite: 6]


    // --------------------------------------------------------
    // NODE 14: Sigmoid
    // --------------------------------------------------------
    static float n14_input[409600];
    static float n14_output[409600];
    
    load_ifm(dram2, 10854400, n14_input, 409600);      //[cite: 6]
    node_sigmoid(n14_input, n14_output, 409600);
    store_ofm(dram2, 11264000, n14_output, 409600);    // Địa chỉ: 11264000 -> 11673599[cite: 6]


    // --------------------------------------------------------
    // NODE 15: Mul (m.0.cv2 output)
    // --------------------------------------------------------
    static float n15_input1[409600];
    static float n15_input2[409600];
    static float n15_output[409600];
    
    load_ifm(dram2, 10854400, n15_input1, 409600);     //[cite: 6]
    load_ifm(dram2, 11264000, n15_input2, 409600);     //[cite: 6]
    
    node_mul(n15_input1, n15_input2, n15_output, 409600);
    store_ofm(dram2, 11673600, n15_output, 409600);    // Địa chỉ: 11673600 -> 12083199[cite: 6]


    // --------------------------------------------------------
    // NODE 16: Add (Shortcut)
    // --------------------------------------------------------
    static float n16_input1[409600];
    static float n16_input2[409600];
    static float n16_output[409600];
    
    // Nạp Split_output_1 từ Node 9[cite: 6]
    load_ifm(dram2, 9830400, n16_input1, 409600);      
    // Nạp Output từ Node 15[cite: 6]
    load_ifm(dram2, 11673600, n16_input2, 409600);     
    
    node_add(n16_input1, n16_input2, n16_output, 409600);
    store_ofm(dram2, 12083200, n16_output, 409600);    // Địa chỉ: 12083200 -> 12492799[cite: 6]


    // --------------------------------------------------------
    // NODE 17: Concat
    // --------------------------------------------------------
    static float n17_in_split0[409600];
    static float n17_in_split1[409600];
    static float n17_in_add[409600];
    static float n17_output[1228800];
    
    // Load Split_output_0 trực tiếp từ nửa đầu của Node 8 (Địa chỉ 9011200)[cite: 6]
    load_ifm(dram2, 9011200, n17_in_split0, 409600);   
    
    // Load Split_output_1 từ Node 9 (Địa chỉ 9830400)[cite: 6]
    load_ifm(dram2, 9830400, n17_in_split1, 409600);   
    
    // Load Node 16 Add output (Địa chỉ 12083200)[cite: 6]
    load_ifm(dram2, 12083200, n17_in_add, 409600);     
    
    float *n17_inputs[3] = {n17_in_split0, n17_in_split1, n17_in_add};
    int n17_sizes[3] = {409600, 409600, 409600};
    
    node_concat(n17_inputs, 3, n17_output, n17_sizes, 1228800);
    store_ofm(dram2, 12492800, n17_output, 1228800);   // Địa chỉ: 12492800 -> 13721599[cite: 6]


    // --------------------------------------------------------
    // NODE 18: Conv (cv2)
    // --------------------------------------------------------
    static float n18_input[1228800];
    static float n18_weight[3072];
    static float n18_bias[64];
    static float n18_output[1638400];
    
    load_ifm(dram2, 12492800, n18_input, 1228800);     // Lấy Output Concat[cite: 6]
    
    load_weight_bias(dram1, 8474, n18_weight, 3072);   // Địa chỉ: 8474 -> 11545[cite: 6]
    load_weight_bias(dram1, 11546, n18_bias, 64);      // Địa chỉ: 11546 -> 11609[cite: 6]
    
    // Thực thi Conv (k=1, s=1, p=0)[cite: 6]
    node_conv(n18_input, n18_weight, n18_bias, n18_output, 1, 48, 160, 160, 64, 1, 1, 0);
    store_ofm(dram2, 13721600, n18_output, 1638400);   // Địa chỉ: 13721600 -> 15359999[cite: 6]


    // --------------------------------------------------------
    // NODE 19: Sigmoid
    // --------------------------------------------------------
    static float n19_input[1638400];
    static float n19_output[1638400];
    
    load_ifm(dram2, 13721600, n19_input, 1638400);     //[cite: 6]
    node_sigmoid(n19_input, n19_output, 1638400);
    store_ofm(dram2, 15360000, n19_output, 1638400);   // Địa chỉ: 15360000 -> 16998399[cite: 6]


    // --------------------------------------------------------
    // NODE 20: Mul (cv2 output)
    // --------------------------------------------------------
    static float n20_input1[1638400];
    static float n20_input2[1638400];
    static float n20_output[1638400];
    
    load_ifm(dram2, 13721600, n20_input1, 1638400);    // Lấy Conv cv2[cite: 6]
    load_ifm(dram2, 15360000, n20_input2, 1638400);    // Lấy Sigmoid[cite: 6]
    
    node_mul(n20_input1, n20_input2, n20_output, 1638400);
    store_ofm(dram2, 16998400, n20_output, 1638400);   // Địa chỉ: 16998400 -> 18636799[cite: 6]
}


// ============================================================================
// MODULE 3: Conv (P3, k=3 s=2)
// Input: 1x64x160x160 | Output: 1x64x80x80 | Params: 36,928
// ============================================================================
void module_3_conv_P3(const float *dram1, float *dram2) {
    
    // --------------------------------------------------------
    // NODE 21: Conv (k=3, s=2, p=1)
    // --------------------------------------------------------
    // Cấp phát BRAM tĩnh cho Node 21
    static float n21_input[1638400];   // 1x64x160x160
    static float n21_weight[36864];    // 64x64x3x3
    static float n21_bias[64];         // 64
    static float n21_output[409600];   // 1x64x80x80
    
    // Load IFM từ DRAM_2 (Output của Node 20 ở module trước)
    load_ifm(dram2, 16998400, n21_input, 1638400);

    // Load Weights & Biases từ DRAM_1
    load_weight_bias(dram1, 11610, n21_weight, 36864); // Địa chỉ: 11610 -> 48473
    load_weight_bias(dram1, 48474, n21_bias, 64);      // Địa chỉ: 48474 -> 48537
    
    // Thực thi Compute
    node_conv(n21_input, n21_weight, n21_bias, n21_output, 1, 64, 160, 160, 64, 3, 2, 1);
    
    // Store OFM ra DRAM_2
    store_ofm(dram2, 18636800, n21_output, 409600);    // Địa chỉ: 18636800 -> 19046399


    // --------------------------------------------------------
    // NODE 22: Sigmoid
    // --------------------------------------------------------
    // Cấp phát BRAM tĩnh cho Node 22
    static float n22_input[409600];
    static float n22_output[409600];
    
    // Load IFM từ DRAM_2 (Output của Node 21 vừa ghi)
    load_ifm(dram2, 18636800, n22_input, 409600);      
    
    // Thực thi Compute
    node_sigmoid(n22_input, n22_output, 409600);
    
    // Store OFM ra DRAM_2
    store_ofm(dram2, 19046400, n22_output, 409600);    // Địa chỉ: 19046400 -> 19455999


    // --------------------------------------------------------
    // NODE 23: Mul (SiLU = Conv_Output * Sigmoid_Output)
    // --------------------------------------------------------
    // Cấp phát BRAM tĩnh cho Node 23
    static float n23_input1[409600];
    static float n23_input2[409600];
    static float n23_output[409600];
    
    // Load IFM từ DRAM_2
    load_ifm(dram2, 18636800, n23_input1, 409600);     // Nhánh 1: Lấy Output của Node 21 (Conv)
    load_ifm(dram2, 19046400, n23_input2, 409600);     // Nhánh 2: Lấy Output của Node 22 (Sigmoid)
    
    // Thực thi Compute
    node_mul(n23_input1, n23_input2, n23_output, 409600);
    
    // Store OFM ra DRAM_2
    store_ofm(dram2, 19456000, n23_output, 409600);    // Địa chỉ: 19456000 -> 19865599
}

// ============================================================================
// MODULE 4: C3k2 (P3)
// Input: 1x64x80x80 | Output: 1x128x80x80
// ============================================================================
void module_4_c3k2_P3(const float *dram1, float *dram2) {
    
    // --------------------------------------------------------
    // NODE 24: Conv (cv1)
    // --------------------------------------------------------
    static float n24_input[409600];    // Lấy Output của Module 3 (Node 23)
    static float n24_weight[4096];
    static float n24_bias[64];
    static float n24_output[409600];
    
    // Output Node 23 ở module trước bắt đầu từ 19456000
    load_ifm(dram2, 19456000, n24_input, 409600);

    // Load Weights & Biases từ DRAM_1
    load_weight_bias(dram1, 48538, n24_weight, 4096);  // Địa chỉ: 48538 -> 52633
    load_weight_bias(dram1, 52634, n24_bias, 64);      // Địa chỉ: 52634 -> 52697
    
    // Thực thi Conv (k=1, s=1, p=0)
    node_conv(n24_input, n24_weight, n24_bias, n24_output, 1, 64, 80, 80, 64, 1, 1, 0);
    store_ofm(dram2, 19865600, n24_output, 409600);    // Địa chỉ: 19865600 -> 20275199


    // --------------------------------------------------------
    // NODE 25: Sigmoid
    // --------------------------------------------------------
    static float n25_input[409600];
    static float n25_output[409600];
    
    load_ifm(dram2, 19865600, n25_input, 409600);      
    node_sigmoid(n25_input, n25_output, 409600);
    store_ofm(dram2, 20275200, n25_output, 409600);    // Địa chỉ: 20275200 -> 20684799


    // --------------------------------------------------------
    // NODE 26: Mul (cv1 output)
    // --------------------------------------------------------
    static float n26_input1[409600];
    static float n26_input2[409600];
    static float n26_output[409600];
    
    load_ifm(dram2, 19865600, n26_input1, 409600);     // Lấy Conv output
    load_ifm(dram2, 20275200, n26_input2, 409600);     // Lấy Sigmoid output
    
    node_mul(n26_input1, n26_input2, n26_output, 409600);
    store_ofm(dram2, 20684800, n26_output, 409600);    // Địa chỉ: 20684800 -> 21094399


    // --------------------------------------------------------
    // NODE 27: Split
    // --------------------------------------------------------
    static float n27_input[409600];
    static float n27_split0[204800];                   // 1x32x80x80
    static float n27_split1[204800];                   // 1x32x80x80
    
    load_ifm(dram2, 20684800, n27_input, 409600);      
    
    float *splits_27[2] = {n27_split0, n27_split1};
    node_split(n27_input, splits_27, 2, 204800);
    
    // Lưu nửa sau (Split_output_1) ra vùng nhớ mới để dùng tiếp
    store_ofm(dram2, 21094400, n27_split1, 204800);    // Địa chỉ: 21094400 -> 21299199


    // --------------------------------------------------------
    // NODE 28: Conv (m.0.cv1)
    // --------------------------------------------------------
    static float n28_input[204800];
    static float n28_weight[4608];
    static float n28_bias[16];
    static float n28_output[102400];                   // 1x16x80x80
    
    load_ifm(dram2, 21094400, n28_input, 204800);      // Lấy Input từ Split_1
    
    load_weight_bias(dram1, 52700, n28_weight, 4608);  // Địa chỉ: 52700 -> 57307
    load_weight_bias(dram1, 57308, n28_bias, 16);      // Địa chỉ: 57308 -> 57323
    
    // Thực thi Conv (k=3, s=1, p=1)
    node_conv(n28_input, n28_weight, n28_bias, n28_output, 1, 32, 80, 80, 16, 3, 1, 1);
    store_ofm(dram2, 21299200, n28_output, 102400);    // Địa chỉ: 21299200 -> 21401599


    // --------------------------------------------------------
    // NODE 29: Sigmoid
    // --------------------------------------------------------
    static float n29_input[102400];
    static float n29_output[102400];
    
    load_ifm(dram2, 21299200, n29_input, 102400);      
    node_sigmoid(n29_input, n29_output, 102400);
    store_ofm(dram2, 21401600, n29_output, 102400);    // Địa chỉ: 21401600 -> 21503999


    // --------------------------------------------------------
    // NODE 30: Mul (m.0.cv1 output)
    // --------------------------------------------------------
    static float n30_input1[102400];
    static float n30_input2[102400];
    static float n30_output[102400];
    
    load_ifm(dram2, 21299200, n30_input1, 102400);     
    load_ifm(dram2, 21401600, n30_input2, 102400);     
    
    node_mul(n30_input1, n30_input2, n30_output, 102400);
    store_ofm(dram2, 21504000, n30_output, 102400);    // Địa chỉ: 21504000 -> 21606399


    // --------------------------------------------------------
    // NODE 31: Conv (m.0.cv2)
    // --------------------------------------------------------
    static float n31_input[102400];
    static float n31_weight[4608];
    static float n31_bias[32];
    static float n31_output[204800];
    
    load_ifm(dram2, 21504000, n31_input, 102400);      
    
    load_weight_bias(dram1, 57324, n31_weight, 4608);  // Địa chỉ: 57324 -> 61931
    load_weight_bias(dram1, 61932, n31_bias, 32);      // Địa chỉ: 61932 -> 61963
    
    node_conv(n31_input, n31_weight, n31_bias, n31_output, 1, 16, 80, 80, 32, 3, 1, 1);
    store_ofm(dram2, 21606400, n31_output, 204800);    // Địa chỉ: 21606400 -> 21811199


    // --------------------------------------------------------
    // NODE 32: Sigmoid
    // --------------------------------------------------------
    static float n32_input[204800];
    static float n32_output[204800];
    
    load_ifm(dram2, 21606400, n32_input, 204800);      
    node_sigmoid(n32_input, n32_output, 204800);
    store_ofm(dram2, 21811200, n32_output, 204800);    // Địa chỉ: 21811200 -> 22015999


    // --------------------------------------------------------
    // NODE 33: Mul (m.0.cv2 output)
    // --------------------------------------------------------
    static float n33_input1[204800];
    static float n33_input2[204800];
    static float n33_output[204800];
    
    load_ifm(dram2, 21606400, n33_input1, 204800);     
    load_ifm(dram2, 21811200, n33_input2, 204800);     
    
    node_mul(n33_input1, n33_input2, n33_output, 204800);
    store_ofm(dram2, 22016000, n33_output, 204800);    // Địa chỉ: 22016000 -> 22220799


    // --------------------------------------------------------
    // NODE 34: Add (Shortcut)
    // --------------------------------------------------------
    static float n34_input1[204800];
    static float n34_input2[204800];
    static float n34_output[204800];
    
    // Nạp Split_output_1 từ Node 27
    load_ifm(dram2, 21094400, n34_input1, 204800);      
    // Nạp Output từ Node 33
    load_ifm(dram2, 22016000, n34_input2, 204800);     
    
    node_add(n34_input1, n34_input2, n34_output, 204800);
    store_ofm(dram2, 22220800, n34_output, 204800);    // Địa chỉ: 22220800 -> 22425599


    // --------------------------------------------------------
    // NODE 35: Concat
    // --------------------------------------------------------
    static float n35_in_split0[204800];
    static float n35_in_split1[204800];
    static float n35_in_add[204800];
    static float n35_output[614400];                   // 1x96x80x80
    
    // Load Split_output_0 trực tiếp từ nửa đầu của Node 26 (Địa chỉ 20684800)
    load_ifm(dram2, 20684800, n35_in_split0, 204800);   
    // Load Split_output_1 từ Node 27 (Địa chỉ 21094400)
    load_ifm(dram2, 21094400, n35_in_split1, 204800);   
    // Load Node 34 Add output (Địa chỉ 22220800)
    load_ifm(dram2, 22220800, n35_in_add, 204800);     
    
    float *n35_inputs[3] = {n35_in_split0, n35_in_split1, n35_in_add};
    int n35_sizes[3] = {204800, 204800, 204800};
    
    node_concat(n35_inputs, 3, n35_output, n35_sizes, 614400);
    store_ofm(dram2, 22425600, n35_output, 614400);    // Địa chỉ: 22425600 -> 23039999


    // --------------------------------------------------------
    // NODE 36: Conv (cv2)
    // --------------------------------------------------------
    static float n36_input[614400];
    static float n36_weight[12288];
    static float n36_bias[128];
    static float n36_output[819200];                   // 1x128x80x80
    
    load_ifm(dram2, 22425600, n36_input, 614400);      // Lấy Output Concat
    
    load_weight_bias(dram1, 61964, n36_weight, 12288); // Địa chỉ: 61964 -> 74251
    load_weight_bias(dram1, 74252, n36_bias, 128);     // Địa chỉ: 74252 -> 74379
    
    // Thực thi Conv (k=1, s=1, p=0)
    node_conv(n36_input, n36_weight, n36_bias, n36_output, 1, 96, 80, 80, 128, 1, 1, 0);
    store_ofm(dram2, 23040000, n36_output, 819200);    // Địa chỉ: 23040000 -> 23859199


    // --------------------------------------------------------
    // NODE 37: Sigmoid
    // --------------------------------------------------------
    static float n37_input[819200];
    static float n37_output[819200];
    
    load_ifm(dram2, 23040000, n37_input, 819200);      
    node_sigmoid(n37_input, n37_output, 819200);
    store_ofm(dram2, 23859200, n37_output, 819200);    // Địa chỉ: 23859200 -> 24678399


    // --------------------------------------------------------
    // NODE 38: Mul (cv2 output)
    // --------------------------------------------------------
    static float n38_input1[819200];
    static float n38_input2[819200];
    static float n38_output[819200];
    
    load_ifm(dram2, 23040000, n38_input1, 819200);     // Lấy Conv cv2
    load_ifm(dram2, 23859200, n38_input2, 819200);     // Lấy Sigmoid
    
    node_mul(n38_input1, n38_input2, n38_output, 819200);
    store_ofm(dram2, 24678400, n38_output, 819200);    // Địa chỉ: 24678400 -> 25497599
}

// ============================================================================
// MODULE 5: Conv (P4, k=3 s=2)
// Input: 1x128x80x80 | Output: 1x128x40x40 | Params: 147,584
// ============================================================================
void module_5_conv_P4(const float *dram1, float *dram2) {
    
    // --------------------------------------------------------
    // NODE 39: Conv (k=3, s=2, p=1)
    // --------------------------------------------------------
    // Cấp phát BRAM tĩnh cho Node 39
    static float n39_input[819200];    // 1x128x80x80 (Output của Module 4 - Node 38)
    static float n39_weight[147456];   // 128x128x3x3
    static float n39_bias[128];        // 128
    static float n39_output[204800];   // 1x128x40x40
    
    // ĐÃ FIX: Lấy IFM từ DRAM_2 tại địa chỉ kết thúc của Node 38 (24678400)
    load_ifm(dram2, 24678400, n39_input, 819200);

    // Load Weights & Biases từ DRAM_1
    load_weight_bias(dram1, 74380, n39_weight, 147456); // Địa chỉ: 74380 -> 221835
    load_weight_bias(dram1, 221836, n39_bias, 128);     // Địa chỉ: 221836 -> 221963
    
    // Thực thi Compute (Batch=1, InC=128, InH=80, InW=80, OutC=128, K=3, S=2, P=1)
    node_conv(n39_input, n39_weight, n39_bias, n39_output, 1, 128, 80, 80, 128, 3, 2, 1);
    
    // Store OFM ra DRAM_2
    store_ofm(dram2, 25497600, n39_output, 204800);    // Địa chỉ: 25497600 -> 25702399


    // --------------------------------------------------------
    // NODE 40: Sigmoid
    // --------------------------------------------------------
    // Cấp phát BRAM tĩnh cho Node 40
    static float n40_input[204800];
    static float n40_output[204800];
    
    // Load IFM từ DRAM_2 (Output của Node 39 vừa ghi)
    load_ifm(dram2, 25497600, n40_input, 204800);      
    
    // Thực thi Compute
    node_sigmoid(n40_input, n40_output, 204800);
    
    // Store OFM ra DRAM_2
    store_ofm(dram2, 25702400, n40_output, 204800);    // Địa chỉ: 25702400 -> 25907199


    // --------------------------------------------------------
    // NODE 41: Mul (SiLU = Conv_Output * Sigmoid_Output)
    // --------------------------------------------------------
    // Cấp phát BRAM tĩnh cho Node 41
    static float n41_input1[204800];
    static float n41_input2[204800];
    static float n41_output[204800];
    
    // Load IFM từ DRAM_2
    load_ifm(dram2, 25497600, n41_input1, 204800);     // Nhánh 1: Lấy Output của Node 39 (Conv)
    load_ifm(dram2, 25702400, n41_input2, 204800);     // Nhánh 2: Lấy Output của Node 40 (Sigmoid)
    
    // Thực thi Compute
    node_mul(n41_input1, n41_input2, n41_output, 204800);
    
    // Store OFM ra DRAM_2
    store_ofm(dram2, 25907200, n41_output, 204800);    // Địa chỉ: 25907200 -> 26111999
}
// ============================================================================
// MODULE 6: C3k2 (P4, n=2xd, c3k=True)
// Input: 1x128x40x40 | Output: 1x128x40x40
// ============================================================================
void module_6_c3k2_P4(const float *dram1, float *dram2) {
    
    // --------------------------------------------------------
    // NODE 42: Conv (cv1)
    // --------------------------------------------------------
    static float n42_input[204800];    // Lấy Output từ Module 5 (Node 41 kết thúc ở 26111999)
    static float n42_weight[16384];
    static float n42_bias[128];
    static float n42_output[204800];
    load_ifm(dram2, 25907200, n42_input, 204800);
    // load_ifm(dram2, 26112000, n42_input, 204800);
    load_weight_bias(dram1, 221964, n42_weight, 16384); // Địa chỉ: 221964 -> 238347
    load_weight_bias(dram1, 238348, n42_bias, 128);     // Địa chỉ: 238348 -> 238475
    
    node_conv(n42_input, n42_weight, n42_bias, n42_output, 1, 128, 40, 40, 128, 1, 1, 0);
    store_ofm(dram2, 26112000, n42_output, 204800);     // Địa chỉ: 26112000 -> 26316799


    // --------------------------------------------------------
    // NODE 43: Sigmoid
    // --------------------------------------------------------
    static float n43_input[204800];
    static float n43_output[204800];
    
    load_ifm(dram2, 26112000, n43_input, 204800);
    node_sigmoid(n43_input, n43_output, 204800);
    store_ofm(dram2, 26316800, n43_output, 204800);     // Địa chỉ: 26316800 -> 26521599


    // --------------------------------------------------------
    // NODE 44: Mul (cv1 output -> SiLU)
    // --------------------------------------------------------
    static float n44_input1[204800];
    static float n44_input2[204800];
    static float n44_output[204800];
    
    load_ifm(dram2, 26112000, n44_input1, 204800);
    load_ifm(dram2, 26316800, n44_input2, 204800);
    node_mul(n44_input1, n44_input2, n44_output, 204800);
    store_ofm(dram2, 26521600, n44_output, 204800);     // Địa chỉ: 26521600 -> 26726399


    // --------------------------------------------------------
    // NODE 45: Split (Tách đôi 128 kênh thành 2 luồng 64 kênh)
    // --------------------------------------------------------
    static float n45_input[204800];
    static float n45_split0[102400];
    static float n45_split1[102400];
    
    load_ifm(dram2, 26521600, n45_input, 204800);
    float *splits_45[2] = {n45_split0, n45_split1};
    node_split(n45_input, splits_45, 2, 102400);
    store_ofm(dram2, 26726400, n45_split1, 102400);    // Lưu split_1 tại: 26726400 -> 26828799


    // --------------------------------------------------------
    // NODE 46: Conv (m.0.cv1)
    // --------------------------------------------------------
    static float n46_input[102400];
    static float n46_weight[2048];
    static float n46_bias[32];
    static float n46_output[51200];
    
    load_ifm(dram2, 26726400, n46_input, 102400);
    load_weight_bias(dram1, 238478, n46_weight, 2048); // Địa chỉ: 238478 -> 240525
    load_weight_bias(dram1, 240526, n46_bias, 32);     // Địa chỉ: 240526 -> 240557
    node_conv(n46_input, n46_weight, n46_bias, n46_output, 1, 64, 40, 40, 32, 1, 1, 0);
    store_ofm(dram2, 26828800, n46_output, 51200);     // Địa chỉ: 26828800 -> 26879999


    // --------------------------------------------------------
    // NODE 47: Conv (m.0.cv2)
    // --------------------------------------------------------
    static float n47_input[102400];
    static float n47_weight[2048];
    static float n47_bias[32];
    static float n47_output[51200];
    
    load_ifm(dram2, 26726400, n47_input, 102400);
    load_weight_bias(dram1, 240558, n47_weight, 2048); // Địa chỉ: 240558 -> 242605
    load_weight_bias(dram1, 242606, n47_bias, 32);     // Địa chỉ: 242606 -> 242637
    node_conv(n47_input, n47_weight, n47_bias, n47_output, 1, 64, 40, 40, 32, 1, 1, 0);
    store_ofm(dram2, 26880000, n47_output, 51200);     // Địa chỉ: 26880000 -> 26931199


    // --------------------------------------------------------
    // NODE 48 & 49: Sigmoid (cho m.0.cv1 và m.0.cv2)
    // --------------------------------------------------------
    static float n48_in[51200], n48_out[51200];
    static float n49_in[51200], n49_out[51200];
    
    load_ifm(dram2, 26828800, n48_in, 51200);
    node_sigmoid(n48_in, n48_out, 51200);
    store_ofm(dram2, 26931200, n48_out, 51200);        // Địa chỉ: 26931200 -> 26982399

    load_ifm(dram2, 26880000, n49_in, 51200);
    node_sigmoid(n49_in, n49_out, 51200);
    store_ofm(dram2, 26982400, n49_out, 51200);        // Địa chỉ: 26982400 -> 27033599


    // --------------------------------------------------------
    // NODE 50 & 51: Mul (SiLU cho m.0.cv1 và m.0.cv2)
    // --------------------------------------------------------
    static float n50_i1[51200], n50_i2[51200], n50_o[51200];
    static float n51_i1[51200], n51_i2[51200], n51_o[51200];
    
    load_ifm(dram2, 26828800, n50_i1, 51200); load_ifm(dram2, 26931200, n50_i2, 51200);
    node_mul(n50_i1, n50_i2, n50_o, 51200);
    store_ofm(dram2, 27033600, n50_o, 51200);          // Địa chỉ: 27033600 -> 27084799

    load_ifm(dram2, 26880000, n51_i1, 51200); load_ifm(dram2, 26982400, n51_i2, 51200);
    node_mul(n51_i1, n51_i2, n51_o, 51200);
    store_ofm(dram2, 27084800, n51_o, 51200);          // Địa chỉ: 27084800 -> 27135999


    // --------------------------------------------------------
    // NODE 52, 53, 54: m.0.m.0 (C3k block đầu tiên)
    // --------------------------------------------------------
    static float n52_in[51200], n52_w[9216], n52_b[32], n52_out[51200];
    load_ifm(dram2, 27033600, n52_in, 51200);
    load_weight_bias(dram1, 242638, n52_w, 9216);      // Địa chỉ: 242638 -> 251853
    load_weight_bias(dram1, 251854, n52_b, 32);        // Địa chỉ: 251854 -> 251885
    node_conv(n52_in, n52_w, n52_b, n52_out, 1, 32, 40, 40, 32, 3, 1, 1);
    store_ofm(dram2, 27136000, n52_out, 51200);        // Địa chỉ: 27136000 -> 27187199

    // Sigmoid + Mul cho m.0.m.0 cv1
    static float n53_in[51200], n53_out[51200];
    load_ifm(dram2, 27136000, n53_in, 51200);
    node_sigmoid(n53_in, n53_out, 51200);
    store_ofm(dram2, 27187200, n53_out, 51200);        // Địa chỉ: 27187200 -> 27238399

    static float n54_i1[51200], n54_i2[51200], n54_out[51200];
    load_ifm(dram2, 27136000, n54_i1, 51200); load_ifm(dram2, 27187200, n54_i2, 51200);
    node_mul(n54_i1, n54_i2, n54_out, 51200);
    store_ofm(dram2, 27238400, n54_out, 51200);        // Địa chỉ: 27238400 -> 27289599


    // --------------------------------------------------------
    // NODE 55, 56, 57, 58: m.0.m.0 cv2 và Add (Shortcut block 1)
    // --------------------------------------------------------
    static float n55_in[51200], n55_w[9216], n55_b[32], n55_out[51200];
    load_ifm(dram2, 27238400, n55_in, 51200);
    load_weight_bias(dram1, 251886, n55_w, 9216);      // Địa chỉ: 251886 -> 261101
    load_weight_bias(dram1, 261102, n55_b, 32);        // Địa chỉ: 261102 -> 261133
    node_conv(n55_in, n55_w, n55_b, n55_out, 1, 32, 40, 40, 32, 3, 1, 1);
    store_ofm(dram2, 27289600, n55_out, 51200);        // Địa chỉ: 27289600 -> 27340799

    // Sigmoid + Mul cho m.0.m.0 cv2
    static float n56_in[51200], n56_out[51200];
    load_ifm(dram2, 27289600, n56_in, 51200);
    node_sigmoid(n56_in, n56_out, 51200);
    store_ofm(dram2, 27340800, n56_out, 51200);        // Địa chỉ: 27340800 -> 27391999

    static float n57_i1[51200], n57_i2[51200], n57_out[51200];
    load_ifm(dram2, 27289600, n57_i1, 51200); load_ifm(dram2, 27340800, n57_i2, 51200);
    node_mul(n57_i1, n57_i2, n57_out, 51200);
    store_ofm(dram2, 27392000, n57_out, 51200);        // Địa chỉ: 27392000 -> 27443199

    // Node 58: Add (input m.0.cv1 sau khi mul + output m.0.m.0)
    static float n58_i1[51200], n58_i2[51200], n58_out[51200];
    load_ifm(dram2, 27033600, n58_i1, 51200); load_ifm(dram2, 27392000, n58_i2, 51200);
    node_add(n58_i1, n58_i2, n58_out, 51200);
    store_ofm(dram2, 27443200, n58_out, 51200);        // Địa chỉ: 27443200 -> 27494399


    // --------------------------------------------------------
    // NODE 59 đến 65: m.0.m.1 (C3k block thứ hai) tương tự block 1
    // --------------------------------------------------------
    static float n59_in[51200], n59_w[9216], n59_b[32], n59_out[51200];
    load_ifm(dram2, 27443200, n59_in, 51200);
    load_weight_bias(dram1, 261134, n59_w, 9216);      // Địa chỉ: 261134 -> 270349
    load_weight_bias(dram1, 270350, n59_b, 32);        // Địa chỉ: 270350 -> 270381
    node_conv(n59_in, n59_w, n59_b, n59_out, 1, 32, 40, 40, 32, 3, 1, 1);
    store_ofm(dram2, 27494400, n59_out, 51200);        // Địa chỉ: 27494400 -> 27545599

    static float n60_in[51200], n60_out[51200];
    load_ifm(dram2, 27494400, n60_in, 51200);
    node_sigmoid(n60_in, n60_out, 51200);
    store_ofm(dram2, 27545600, n60_out, 51200);        // Địa chỉ: 27545600 -> 27596799

    static float n61_i1[51200], n61_i2[51200], n61_out[51200];
    load_ifm(dram2, 27494400, n61_i1, 51200); load_ifm(dram2, 27545600, n61_i2, 51200);
    node_mul(n61_i1, n61_i2, n61_out, 51200);
    store_ofm(dram2, 27596800, n61_out, 51200);        // Địa chỉ: 27596800 -> 27647999

    static float n62_in[51200], n62_w[9216], n62_b[32], n62_out[51200];
    load_ifm(dram2, 27596800, n62_in, 51200);
    load_weight_bias(dram1, 270382, n62_w, 9216);      // Địa chỉ: 270382 -> 279597
    load_weight_bias(dram1, 279598, n62_b, 32);        // Địa chỉ: 279598 -> 279629
    node_conv(n62_in, n62_w, n62_b, n62_out, 1, 32, 40, 40, 32, 3, 1, 1);
    store_ofm(dram2, 27648000, n62_out, 51200);        // Địa chỉ: 27648000 -> 27699199

    static float n63_in[51200], n63_out[51200];
    load_ifm(dram2, 27648000, n63_in, 51200);
    node_sigmoid(n63_in, n63_out, 51200);
    store_ofm(dram2, 27699200, n63_out, 51200);        // Địa chỉ: 27699200 -> 27750399

    static float n64_i1[51200], n64_i2[51200], n64_out[51200];
    load_ifm(dram2, 27648000, n64_i1, 51200); load_ifm(dram2, 27699200, n64_i2, 51200);
    node_mul(n64_i1, n64_i2, n64_out, 51200);
    store_ofm(dram2, 27750400, n64_out, 51200);        // Địa chỉ: 27750400 -> 27801599

    // Node 65: Add (Shortcut block 2)
    static float n65_i1[51200], n65_i2[51200], n65_out[51200];
    load_ifm(dram2, 27443200, n65_i1, 51200); load_ifm(dram2, 27750400, n65_i2, 51200);
    node_add(n65_i1, n65_i2, n65_out, 51200);
    store_ofm(dram2, 27801600, n65_out, 51200);        // Địa chỉ: 27801600 -> 27852799


    // --------------------------------------------------------
    // NODE 66: Concat (m.0 output: nối block 2 + m.0.cv2 đầu ra)
    // --------------------------------------------------------
    static float n66_in1[51200], n66_in2[51200], n66_out[102400];
    load_ifm(dram2, 27801600, n66_in1, 51200);          // Output block 2
    load_ifm(dram2, 27084800, n66_in2, 51200);          // Output m.0.cv2 (Node 51)
    float *n66_inputs[2] = {n66_in1, n66_in2};
    int n66_sizes[2] = {51200, 51200};
    node_concat(n66_inputs, 2, n66_out, n66_sizes, 102400);
    store_ofm(dram2, 27852800, n66_out, 102400);       // Địa chỉ: 27852800 -> 27955199


    // --------------------------------------------------------
    // NODE 67, 68, 69: cv3 (Conv -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n67_in[102400], n67_w[4096], n67_b[64], n67_out[102400];
    load_ifm(dram2, 27852800, n67_in, 102400);
    load_weight_bias(dram1, 279630, n67_w, 4096);      // Địa chỉ: 279630 -> 283725
    load_weight_bias(dram1, 283726, n67_b, 64);        // Địa chỉ: 283726 -> 283789
    node_conv(n67_in, n67_w, n67_b, n67_out, 1, 64, 40, 40, 64, 1, 1, 0);
    store_ofm(dram2, 27955200, n67_out, 102400);       // Địa chỉ: 27955200 -> 28057599

    static float n68_in[102400], n68_out[102400];
    load_ifm(dram2, 27955200, n68_in, 102400);
    node_sigmoid(n68_in, n68_out, 102400);
    store_ofm(dram2, 28057600, n68_out, 102400);       // Địa chỉ: 28057600 -> 28159999

    static float n69_i1[102400], n69_i2[102400], n69_out[102400];
    load_ifm(dram2, 27955200, n69_i1, 102400); load_ifm(dram2, 28057600, n69_i2, 102400);
    node_mul(n69_i1, n69_i2, n69_out, 102400);
    store_ofm(dram2, 28160000, n69_out, 102400);       // Địa chỉ: 28160000 -> 28262399


    // --------------------------------------------------------
    // NODE 70: Concat (Gộp split_0, split_1, và cv3 output)
    // --------------------------------------------------------
    static float n70_in1[102400], n70_in2[102400], n70_in3[102400], n70_out[307200];
    
    // ĐÃ FIX OFFSETS:
    load_ifm(dram2, 26521600, n70_in1, 102400); // Split_0 (Lấy trực tiếp từ Output Node 44)
    load_ifm(dram2, 26726400, n70_in2, 102400); // Split_1 (Lấy từ Output Node 45)
    load_ifm(dram2, 28160000, n70_in3, 102400); // cv3 output (Lấy từ Node 69)
    
    float *n70_inputs[3] = {n70_in1, n70_in2, n70_in3};
    int n70_sizes[3] = {102400, 102400, 102400};
    node_concat(n70_inputs, 3, n70_out, n70_sizes, 307200);
    store_ofm(dram2, 28262400, n70_out, 307200);       // Địa chỉ: 28262400 -> 28569599

    // --------------------------------------------------------
    // NODE 71, 72, 73: cv2 (Conv -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n71_in[307200], n71_weight[24576], n71_bias[128], n71_output[204800];
    load_ifm(dram2, 28262400, n71_in, 307200);
    load_weight_bias(dram1, 283790, n71_weight, 24576); // Địa chỉ: 283790 -> 308365
    load_weight_bias(dram1, 308366, n71_bias, 128);     // Địa chỉ: 308366 -> 308493
    node_conv(n71_in, n71_weight, n71_bias, n71_output, 1, 192, 40, 40, 128, 1, 1, 0);
    store_ofm(dram2, 28569600, n71_output, 204800);     // Địa chỉ: 28569600 -> 28774399

    static float n72_in[204800], n72_out[204800];
    load_ifm(dram2, 28569600, n72_in, 204800);
    node_sigmoid(n72_in, n72_out, 204800);
    store_ofm(dram2, 28774400, n72_out, 204800);        // Địa chỉ: 28774400 -> 28979199

    static float n73_i1[204800], n73_i2[204800], n73_out[204800];
    load_ifm(dram2, 28569600, n73_i1, 204800); load_ifm(dram2, 28774400, n73_i2, 204800);
    node_mul(n73_i1, n73_i2, n73_out, 204800);
    store_ofm(dram2, 28979200, n73_out, 204800);        // Địa chỉ: 28979200 -> 29183999
}

// ============================================================================
// MODULE 7: Conv (P5, k=3 s=2)
// Input: 1x128x40x40 | Output: 1x256x20x20 | Params: 295,168
// ============================================================================
void module_7_conv_P5(const float *dram1, float *dram2) {
    
    // --------------------------------------------------------
    // NODE 74: Conv (k=3, s=2, p=1)
    // --------------------------------------------------------
    static float n74_input[204800];    // 1x128x40x40 (Output của Module 6)
    static float n74_weight[294912];   // 256x128x3x3
    static float n74_bias[256];        // 256
    static float n74_output[102400];   // 1x256x20x20
    
    // ĐÃ FIX: Lấy IFM từ DRAM_2 (Địa chỉ kết thúc của Node 73)
    load_ifm(dram2, 28979200, n74_input, 204800);

    // Load Weights & Biases từ DRAM_1
    load_weight_bias(dram1, 308494, n74_weight, 294912); // Địa chỉ: 308494 -> 603405
    load_weight_bias(dram1, 603406, n74_bias, 256);      // Địa chỉ: 603406 -> 603661
    
    // Thực thi Compute
    node_conv(n74_input, n74_weight, n74_bias, n74_output, 1, 128, 40, 40, 256, 3, 2, 1);
    
    // Store OFM ra DRAM_2
    store_ofm(dram2, 29184000, n74_output, 102400);    // Địa chỉ: 29184000 -> 29286399


    // --------------------------------------------------------
    // NODE 75: Sigmoid
    // --------------------------------------------------------
    static float n75_input[102400];
    static float n75_output[102400];
    
    // ĐÃ FIX: Lấy IFM từ DRAM_2 (Địa chỉ Output của Node 74)
    load_ifm(dram2, 29184000, n75_input, 102400);      
    
    // Thực thi Compute
    node_sigmoid(n75_input, n75_output, 102400);
    
    // Store OFM ra DRAM_2
    store_ofm(dram2, 29286400, n75_output, 102400);    // Địa chỉ: 29286400 -> 29388799


    // --------------------------------------------------------
    // NODE 76: Mul (SiLU = Conv_Output * Sigmoid_Output)
    // --------------------------------------------------------
    static float n76_input1[102400];
    static float n76_input2[102400];
    static float n76_output[102400];
    
    // Load IFM từ DRAM_2
    load_ifm(dram2, 29184000, n76_input1, 102400);     // Nhánh 1: Lấy Output của Node 74 (Conv)
    load_ifm(dram2, 29286400, n76_input2, 102400);     // Nhánh 2: Lấy Output của Node 75 (Sigmoid)
    
    // Thực thi Compute
    node_mul(n76_input1, n76_input2, n76_output, 102400);
    
    // Store OFM ra DRAM_2
    store_ofm(dram2, 29388800, n76_output, 102400);    // Địa chỉ: 29388800 -> 29491199
}

// ============================================================================
// MODULE 8: C3k2 (P5, n=2xd, c3k=True)
// Input: 1x256x20x20 | Output: 1x256x20x20 | Params: 345,090
// ============================================================================
void module_8_c3k2_P5(const float *dram1, float *dram2) {
    
    // --------------------------------------------------------
    // NODE 77: Conv (cv1)
    // --------------------------------------------------------
    static float n77_input[102400];    // 1x256x20x20 (Output của Module 7)
    static float n77_weight[65536];    // 256x256x1x1
    static float n77_bias[256];        // 256
    static float n77_output[102400];   // 1x256x20x20
    load_ifm(dram2, 29388800, n77_input, 102400);
    // load_ifm(dram2, 29491200, n77_input, 102400);
    load_weight_bias(dram1, 603662, n77_weight, 65536); // Địa chỉ: 603662 -> 669197
    load_weight_bias(dram1, 669198, n77_bias, 256);     // Địa chỉ: 669198 -> 669453
    
    node_conv(n77_input, n77_weight, n77_bias, n77_output, 1, 256, 20, 20, 256, 1, 1, 0);
    store_ofm(dram2, 29491200, n77_output, 102400);     // Địa chỉ: 29491200 -> 29593599


    // --------------------------------------------------------
    // NODE 78: Sigmoid
    // --------------------------------------------------------
    static float n78_input[102400];
    static float n78_output[102400];
    
    load_ifm(dram2, 29491200, n78_input, 102400);
    node_sigmoid(n78_input, n78_output, 102400);
    store_ofm(dram2, 29593600, n78_output, 102400);     // Địa chỉ: 29593600 -> 29695999


    // --------------------------------------------------------
    // NODE 79: Mul (cv1 output -> SiLU)
    // --------------------------------------------------------
    static float n79_input1[102400];
    static float n79_input2[102400];
    static float n79_output[102400];
    
    load_ifm(dram2, 29491200, n79_input1, 102400);
    load_ifm(dram2, 29593600, n79_input2, 102400);
    node_mul(n79_input1, n79_input2, n79_output, 102400);
    store_ofm(dram2, 29696000, n79_output, 102400);     // Địa chỉ: 29696000 -> 29798399


    // --------------------------------------------------------
    // NODE 80: Split (Tách đôi 256 kênh thành 2 luồng 128 kênh)
    // --------------------------------------------------------
    static float n80_input[102400];
    static float n80_split0[51200];
    static float n80_split1[51200];
    
    load_ifm(dram2, 29696000, n80_input, 102400);
    float *splits_80[2] = {n80_split0, n80_split1};
    node_split(n80_input, splits_80, 2, 51200);
    store_ofm(dram2, 29798400, n80_split1, 51200);     // Lưu split_1 tại: 29798400 -> 29849599


    // --------------------------------------------------------
    // NODE 81: Conv (m.0.cv1)
    // --------------------------------------------------------
    static float n81_input[51200];
    static float n81_weight[8192];
    static float n81_bias[64];
    static float n81_output[25600];
    
    load_ifm(dram2, 29798400, n81_input, 51200);
    load_weight_bias(dram1, 669456, n81_weight, 8192);  // Địa chỉ: 669456 -> 677647
    load_weight_bias(dram1, 677648, n81_bias, 64);      // Địa chỉ: 677648 -> 677711
    node_conv(n81_input, n81_weight, n81_bias, n81_output, 1, 128, 20, 20, 64, 1, 1, 0);
    store_ofm(dram2, 29849600, n81_output, 25600);     // Địa chỉ: 29849600 -> 29875199


    // --------------------------------------------------------
    // NODE 82: Conv (m.0.cv2)
    // --------------------------------------------------------
    static float n82_input[51200];
    static float n82_weight[8192];
    static float n82_bias[64];
    static float n82_output[25600];
    
    load_ifm(dram2, 29798400, n82_input, 51200);
    load_weight_bias(dram1, 677712, n82_weight, 8192);  // Địa chỉ: 677712 -> 685903
    load_weight_bias(dram1, 685904, n82_bias, 64);      // Địa chỉ: 685904 -> 685967
    node_conv(n82_input, n82_weight, n82_bias, n82_output, 1, 128, 20, 20, 64, 1, 1, 0);
    store_ofm(dram2, 29875200, n82_output, 25600);     // Địa chỉ: 29875200 -> 29900799


    // --------------------------------------------------------
    // NODE 83 & 84: Sigmoid (cho m.0.cv1 và m.0.cv2)
    // --------------------------------------------------------
    static float n83_in[25600], n83_out[25600];
    static float n84_in[25600], n84_out[25600];
    
    load_ifm(dram2, 29849600, n83_in, 25600);
    node_sigmoid(n83_in, n83_out, 25600);
    store_ofm(dram2, 29900800, n83_out, 25600);        // Địa chỉ: 29900800 -> 29926399

    load_ifm(dram2, 29875200, n84_in, 25600);
    node_sigmoid(n84_in, n84_out, 25600);
    store_ofm(dram2, 29926400, n84_out, 25600);        // Địa chỉ: 29926400 -> 29951999


    // --------------------------------------------------------
    // NODE 85 & 86: Mul (SiLU)
    // --------------------------------------------------------
    static float n85_i1[25600], n85_i2[25600], n85_o[25600];
    static float n86_i1[25600], n86_i2[25600], n86_o[25600];
    
    load_ifm(dram2, 29849600, n85_i1, 25600); load_ifm(dram2, 29900800, n85_i2, 25600);
    node_mul(n85_i1, n85_i2, n85_o, 25600);
    store_ofm(dram2, 29952000, n85_o, 25600);          // Địa chỉ: 29952000 -> 29977599

    load_ifm(dram2, 29875200, n86_i1, 25600); load_ifm(dram2, 29926400, n86_i2, 25600);
    node_mul(n86_i1, n86_i2, n86_o, 25600);
    store_ofm(dram2, 29977600, n86_o, 25600);          // Địa chỉ: 29977600 -> 30003199


    // --------------------------------------------------------
    // NODE 87, 88, 89: m.0.m.0 (Block 1 - Conv -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n87_in[25600], n87_w[36864], n87_b[64], n87_out[25600];
    load_ifm(dram2, 29952000, n87_in, 25600);
    load_weight_bias(dram1, 685968, n87_w, 36864);     // Địa chỉ: 685968 -> 722831
    load_weight_bias(dram1, 722832, n87_b, 64);        // Địa chỉ: 722832 -> 722895
    node_conv(n87_in, n87_w, n87_b, n87_out, 1, 64, 20, 20, 64, 3, 1, 1);
    store_ofm(dram2, 30003200, n87_out, 25600);        // Địa chỉ: 30003200 -> 30028799

    static float n88_in[25600], n88_out[25600];
    load_ifm(dram2, 30003200, n88_in, 25600);
    node_sigmoid(n88_in, n88_out, 25600);
    store_ofm(dram2, 30028800, n88_out, 25600);        // Địa chỉ: 30028800 -> 30054399

    static float n89_i1[25600], n89_i2[25600], n89_out[25600];
    load_ifm(dram2, 30003200, n89_i1, 25600); load_ifm(dram2, 30028800, n89_i2, 25600);
    node_mul(n89_i1, n89_i2, n89_out, 25600);
    store_ofm(dram2, 30054400, n89_out, 25600);        // Địa chỉ: 30054400 -> 30079999


    // --------------------------------------------------------
    // NODE 90, 91, 92, 93: m.0.m.0 phần 2 (Conv -> Sigmoid -> Mul -> Add Shortcut)
    // --------------------------------------------------------
    static float n90_in[25600], n90_w[36864], n90_b[64], n90_out[25600];
    load_ifm(dram2, 30054400, n90_in, 25600);
    load_weight_bias(dram1, 722896, n90_w, 36864);     // Địa chỉ: 722896 -> 759759
    load_weight_bias(dram1, 759760, n90_b, 64);        // Địa chỉ: 759760 -> 759823
    node_conv(n90_in, n90_w, n90_b, n90_out, 1, 64, 20, 20, 64, 3, 1, 1);
    store_ofm(dram2, 30080000, n90_out, 25600);        // Địa chỉ: 30080000 -> 30105599

    static float n91_in[25600], n91_out[25600];
    load_ifm(dram2, 30080000, n91_in, 25600);
    node_sigmoid(n91_in, n91_out, 25600);
    store_ofm(dram2, 30105600, n91_out, 25600);        // Địa chỉ: 30105600 -> 30131199

    static float n92_i1[25600], n92_i2[25600], n92_out[25600];
    load_ifm(dram2, 30080000, n92_i1, 25600); load_ifm(dram2, 30105600, n92_i2, 25600);
    node_mul(n92_i1, n92_i2, n92_out, 25600);
    store_ofm(dram2, 30131200, n92_out, 25600);        // Địa chỉ: 30131200 -> 30156799

    // Node 93: Add (Shortcut block 1)
    static float n93_i1[25600], n93_i2[25600], n93_out[25600];
    
    // ĐÃ FIX: Đổi 29977600 thành 29952000 (Lấy Output của Node 85)
    load_ifm(dram2, 29952000, n93_i1, 25600); 
    load_ifm(dram2, 30131200, n93_i2, 25600); // Lấy Output của Node 92
    
    node_add(n93_i1, n93_i2, n93_out, 25600);
    store_ofm(dram2, 30156800, n93_out, 25600);        // Địa chỉ: 30156800 -> 30182399

    // --------------------------------------------------------
    // NODE 94 đến 100: m.0.m.1 (Block 2 - C3k block thứ hai)
    // --------------------------------------------------------
    static float n94_in[25600], n94_w[36864], n94_b[64], n94_out[25600];
    load_ifm(dram2, 30156800, n94_in, 25600);
    load_weight_bias(dram1, 759824, n94_w, 36864);     // Địa chỉ: 759824 -> 796687
    load_weight_bias(dram1, 796688, n94_b, 64);        // Địa chỉ: 796688 -> 796751
    node_conv(n94_in, n94_w, n94_b, n94_out, 1, 64, 20, 20, 64, 3, 1, 1);
    store_ofm(dram2, 30182400, n94_out, 25600);        // Địa chỉ: 30182400 -> 30207999

    static float n95_in[25600], n95_out[25600];
    load_ifm(dram2, 30182400, n95_in, 25600);
    node_sigmoid(n95_in, n95_out, 25600);
    store_ofm(dram2, 30208000, n95_out, 25600);        // Địa chỉ: 30208000 -> 30233599

    static float n96_i1[25600], n96_i2[25600], n96_out[25600];
    load_ifm(dram2, 30182400, n96_i1, 25600); load_ifm(dram2, 30208000, n96_i2, 25600);
    node_mul(n96_i1, n96_i2, n96_out, 25600);
    store_ofm(dram2, 30233600, n96_out, 25600);        // Địa chỉ: 30233600 -> 30259199

    static float n97_in[25600], n97_w[36864], n97_b[64], n97_out[25600];
    load_ifm(dram2, 30233600, n97_in, 25600);
    load_weight_bias(dram1, 796752, n97_w, 36864);     // Địa chỉ: 796752 -> 833615
    load_weight_bias(dram1, 833616, n97_b, 64);        // Địa chỉ: 833616 -> 833679
    node_conv(n97_in, n97_w, n97_b, n97_out, 1, 64, 20, 20, 64, 3, 1, 1);
    store_ofm(dram2, 30259200, n97_out, 25600);        // Địa chỉ: 30259200 -> 30284799

    static float n98_in[25600], n98_out[25600];
    load_ifm(dram2, 30259200, n98_in, 25600);
    node_sigmoid(n98_in, n98_out, 25600);
    store_ofm(dram2, 30284800, n98_out, 25600);        // Địa chỉ: 30284800 -> 30310399

    static float n99_i1[25600], n99_i2[25600], n99_out[25600];
    load_ifm(dram2, 30259200, n99_i1, 25600); load_ifm(dram2, 30284800, n99_i2, 25600);
    node_mul(n99_i1, n99_i2, n99_out, 25600);
    store_ofm(dram2, 30310400, n99_out, 25600);        // Địa chỉ: 30310400 -> 30335999

    // Node 100: Add (Shortcut block 2)
    static float n100_i1[25600], n100_i2[25600], n100_out[25600];
    load_ifm(dram2, 30156800, n100_i1, 25600); load_ifm(dram2, 30310400, n100_i2, 25600);
    node_add(n100_i1, n100_i2, n100_out, 25600);
    store_ofm(dram2, 30336000, n100_out, 25600);       // Địa chỉ: 30336000 -> 30361599


    // --------------------------------------------------------
    // NODE 101: Concat (Nối block 2 + m.0.cv2 đầu ra)
    // --------------------------------------------------------
    static float n101_in1[25600], n101_in2[25600], n101_out[51200];
    load_ifm(dram2, 30336000, n101_in1, 25600);         // Output block 2
    load_ifm(dram2, 29977600, n101_in2, 25600);         // Output m.0.cv2 (Node 86)
    float *n101_inputs[2] = {n101_in1, n101_in2};
    int n101_sizes[2] = {25600, 25600};
    node_concat(n101_inputs, 2, n101_out, n101_sizes, 51200);
    store_ofm(dram2, 30361600, n101_out, 51200);        // Địa chỉ: 30361600 -> 30412799


    // --------------------------------------------------------
    // NODE 102, 103, 104: cv3 (Conv -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n102_in[51200], n102_weight[16384], n102_bias[128], n102_output[51200];
    load_ifm(dram2, 30361600, n102_in, 51200);
    load_weight_bias(dram1, 833680, n102_weight, 16384); // Địa chỉ: 833680 -> 850063
    load_weight_bias(dram1, 850064, n102_bias, 128);     // Địa chỉ: 850064 -> 850191
    node_conv(n102_in, n102_weight, n102_bias, n102_output, 1, 128, 20, 20, 128, 1, 1, 0);
    store_ofm(dram2, 30412800, n102_output, 51200);      // Địa chỉ: 30412800 -> 30463999

    static float n103_in[51200], n103_out[51200];
    load_ifm(dram2, 30412800, n103_in, 51200);
    node_sigmoid(n103_in, n103_out, 51200);
    store_ofm(dram2, 30464000, n103_out, 51200);        // Địa chỉ: 30464000 -> 30515199

    static float n104_i1[51200], n104_i2[51200], n104_out[51200];
    load_ifm(dram2, 30412800, n104_i1, 51200); load_ifm(dram2, 30464000, n104_i2, 51200);
    node_mul(n104_i1, n104_i2, n104_out, 51200);
    store_ofm(dram2, 30515200, n104_out, 51200);        // Địa chỉ: 30515200 -> 30566399


    // --------------------------------------------------------
    // NODE 105: Concat (Gộp split_0, split_1 và cv3 output)
    // --------------------------------------------------------
    static float n105_in1[51200], n105_in2[51200], n105_in3[51200], n105_out[153600];
    
    // ĐÃ FIX OFFSETS:
    load_ifm(dram2, 29696000, n105_in1, 51200); // Split_0 (Lấy từ nửa đầu Output Node 79)
    load_ifm(dram2, 29798400, n105_in2, 51200); // Split_1 (Lấy từ Output Node 80)
    load_ifm(dram2, 30515200, n105_in3, 51200); // cv3 output (Lấy từ Node 104)
    
    float *n105_inputs[3] = {n105_in1, n105_in2, n105_in3};
    int n105_sizes[3] = {51200, 51200, 51200};
    node_concat(n105_inputs, 3, n105_out, n105_sizes, 153600);
    store_ofm(dram2, 30566400, n105_out, 153600);       // Địa chỉ: 30566400 -> 30719999


    // --------------------------------------------------------
    // NODE 106, 107, 108: cv2 (Conv -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n106_in[153600], n106_weight[98304], n106_bias[256], n106_output[102400];
    load_ifm(dram2, 30566400, n106_in, 153600);
    load_weight_bias(dram1, 850192, n106_weight, 98304); // Địa chỉ: 850192 -> 948495
    load_weight_bias(dram1, 948496, n106_bias, 256);     // Địa chỉ: 948496 -> 948751
    node_conv(n106_in, n106_weight, n106_bias, n106_output, 1, 384, 20, 20, 256, 1, 1, 0);
    store_ofm(dram2, 30720000, n106_output, 102400);     // Địa chỉ: 30720000 -> 30822399

    static float n107_in[102400], n107_out[102400];
    load_ifm(dram2, 30720000, n107_in, 102400);
    node_sigmoid(n107_in, n107_out, 102400);
    store_ofm(dram2, 30822400, n107_out, 102400);        // Địa chỉ: 30822400 -> 30924799

    static float n108_i1[102400], n108_i2[102400], n108_out[102400];
    load_ifm(dram2, 30720000, n108_i1, 102400); load_ifm(dram2, 30822400, n108_i2, 102400);
    node_mul(n108_i1, n108_i2, n108_out, 102400);
    store_ofm(dram2, 30924800, n108_out, 102400);        // Địa chỉ: 30924800 -> 31027199
}

// ============================================================================
// MODULE 9: SPPF (Spatial Pyramid Pooling - Fast)
// Input: 1x256x20x20 | Output: 1x256x20x20 | Params: 164,224
// ============================================================================
void module_9_sppf(const float *dram1, float *dram2) {
    
    // --------------------------------------------------------
    // NODE 109: Conv (cv1) - Giảm số kênh từ 256 xuống 128
    // --------------------------------------------------------
    static float n109_input[102400];   // 1x256x20x20 (Output Node 108)
    static float n109_weight[32768];   // 128x256x1x1
    static float n109_bias[128];       // 128
    static float n109_output[51200];   // 1x128x20x20
    
    load_ifm(dram2, 30924800, n109_input, 102400);
    load_weight_bias(dram1, 948752, n109_weight, 32768); // Địa chỉ: 948752 -> 981519
    load_weight_bias(dram1, 981520, n109_bias, 128);     // Địa chỉ: 981520 -> 981647
    
    node_conv(n109_input, n109_weight, n109_bias, n109_output, 1, 256, 20, 20, 128, 1, 1, 0);
    store_ofm(dram2, 31027200, n109_output, 51200);      // Địa chỉ: 31027200 -> 31078399

    // --------------------------------------------------------
    // NODE 110: MaxPool 1 (k=5, s=1, p=2)
    // --------------------------------------------------------
    static float n110_input[51200], n110_output[51200];
    load_ifm(dram2, 31027200, n110_input, 51200);
    node_maxpool(n110_input, n110_output, 128, 20, 20, 5, 1, 2);
    store_ofm(dram2, 31078400, n110_output, 51200);      

    // --------------------------------------------------------
    // NODE 111: MaxPool 2 (k=5, s=1, p=2)
    // --------------------------------------------------------
    static float n111_input[51200], n111_output[51200];
    load_ifm(dram2, 31078400, n111_input, 51200);        // Đọc từ Output của MaxPool 1
    node_maxpool(n111_input, n111_output, 128, 20, 20, 5, 1, 2);
    store_ofm(dram2, 31129600, n111_output, 51200);      

    // --------------------------------------------------------
    // NODE 112: MaxPool 3 (k=5, s=1, p=2)
    // --------------------------------------------------------
    static float n112_input[51200], n112_output[51200];
    load_ifm(dram2, 31129600, n112_input, 51200);        // Đọc từ Output của MaxPool 2
    node_maxpool(n112_input, n112_output, 128, 20, 20, 5, 1, 2);
    store_ofm(dram2, 31180800, n112_output, 51200);      

    // --------------------------------------------------------
    // NODE 113: Concat (Nối cv1, maxpool1, maxpool2, maxpool3)
    // --------------------------------------------------------
    static float n113_in1[51200], n113_in2[51200], n113_in3[51200], n113_in4[51200], n113_out[204800];
    load_ifm(dram2, 31027200, n113_in1, 51200); // Output Conv (Node 109)
    load_ifm(dram2, 31078400, n113_in2, 51200); // Output MaxPool 1 (Node 110)
    load_ifm(dram2, 31129600, n113_in3, 51200); // Output MaxPool 2 (Node 111)
    load_ifm(dram2, 31180800, n113_in4, 51200); // Output MaxPool 3 (Node 112)
    
    float *n113_inputs[4] = {n113_in1, n113_in2, n113_in3, n113_in4};
    int n113_sizes[4] = {51200, 51200, 51200, 51200};
    node_concat(n113_inputs, 4, n113_out, n113_sizes, 204800);
    store_ofm(dram2, 31232000, n113_out, 204800);        

    // --------------------------------------------------------
    // NODE 114: Conv (cv2) - Tăng kênh từ 512 lên 256
    // --------------------------------------------------------
    static float n114_input[204800], n114_weight[131072], n114_bias[256], n114_output[102400];
    load_ifm(dram2, 31232000, n114_input, 204800);
    load_weight_bias(dram1, 981648, n114_weight, 131072);
    load_weight_bias(dram1, 1112720, n114_bias, 256);     
    node_conv(n114_input, n114_weight, n114_bias, n114_output, 1, 512, 20, 20, 256, 1, 1, 0);
    store_ofm(dram2, 31436800, n114_output, 102400);     

    // --------------------------------------------------------
    // NODE 115: Sigmoid
    // --------------------------------------------------------
    static float n115_input[102400], n115_output[102400];
    load_ifm(dram2, 31436800, n115_input, 102400);      
    node_sigmoid(n115_input, n115_output, 102400);
    store_ofm(dram2, 31539200, n115_output, 102400);     

    // --------------------------------------------------------
    // NODE 116: Mul (SiLU = Conv_Output * Sigmoid_Output)
    // --------------------------------------------------------
    static float n116_input1[102400], n116_input2[102400], n116_output[102400];
    load_ifm(dram2, 31436800, n116_input1, 102400);
    load_ifm(dram2, 31539200, n116_input2, 102400);
    node_mul(n116_input1, n116_input2, n116_output, 102400);
    store_ofm(dram2, 31641600, n116_output, 102400);     

    // --------------------------------------------------------
    // NODE 117: Add (Residual Shortcut)
    // --------------------------------------------------------
    static float n117_input1[102400], n117_input2[102400], n117_output[102400];
    load_ifm(dram2, 30924800, n117_input1, 102400); // Input ban đầu của SPPF (Node 108)
    load_ifm(dram2, 31641600, n117_input2, 102400); // Output SiLU (Node 116)
    node_add(n117_input1, n117_input2, n117_output, 102400);
    store_ofm(dram2, 31744000, n117_output, 102400);     // Địa chỉ: 31744000 -> 31846399
}

// ============================================================================
// MODULE 10: C2PSA (Cross Stage Partial Spatial Attention)
// Input: 1x256x20x20 | Output: 1x256x20x20 
// ============================================================================
void module_10_c2psa(const float *dram1, float *dram2) {
    
    // --------------------------------------------------------
    // NODE 118, 119, 120: cv1 (Conv -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n118_in[102400], n118_w[65536], n118_b[256], n118_out[102400];
    load_ifm(dram2, 31744000, n118_in, 102400);
    load_weight_bias(dram1, 1112976, n118_w, 65536);
    load_weight_bias(dram1, 1178512, n118_b, 256);
    node_conv(n118_in, n118_w, n118_b, n118_out, 1, 256, 20, 20, 256, 1, 1, 0);
    store_ofm(dram2, 31846400, n118_out, 102400);

    static float n119_in[102400], n119_out[102400];
    load_ifm(dram2, 31846400, n119_in, 102400);
    node_sigmoid(n119_in, n119_out, 102400);
    store_ofm(dram2, 31948800, n119_out, 102400);

    static float n120_i1[102400], n120_i2[102400], n120_out[102400];
    load_ifm(dram2, 31846400, n120_i1, 102400);
    load_ifm(dram2, 31948800, n120_i2, 102400);
    node_mul(n120_i1, n120_i2, n120_out, 102400);
    store_ofm(dram2, 32051200, n120_out, 102400);

    // --------------------------------------------------------
    // NODE 121: Split (Tách đôi 256 kênh thành 2 luồng 128 kênh)
    // --------------------------------------------------------
    static float n121_in[102400], n121_split0[51200], n121_split1[51200];
    load_ifm(dram2, 32051200, n121_in, 102400);
    float *splits_121[2] = {n121_split0, n121_split1};
    node_split(n121_in, splits_121, 2, 51200);
    store_ofm(dram2, 32153600, n121_split1, 51200);

    // --------------------------------------------------------
    // NODE 122: Conv (QKV gen)
    // --------------------------------------------------------
    static float n122_in[51200], n122_w[32768], n122_b[256], n122_out[102400];
    load_ifm(dram2, 32153600, n122_in, 51200);
    load_weight_bias(dram1, 1178770, n122_w, 32768);
    load_weight_bias(dram1, 1211538, n122_b, 256);
    node_conv(n122_in, n122_w, n122_b, n122_out, 1, 128, 20, 20, 256, 1, 1, 0);
    store_ofm(dram2, 32204800, n122_out, 102400);

    // --------------------------------------------------------
    // NODE 123: Reshape
    // --------------------------------------------------------
    static float n123_in[102400], n123_out[102400];
    load_ifm(dram2, 32204800, n123_in, 102400);
    node_reshape(n123_in, n123_out, 102400); 
    store_ofm(dram2, 32307200, n123_out, 102400);

    // --------------------------------------------------------
    // NODE 124: Split (Tách Q, K, V dọc theo trục Channel=128)
    // --------------------------------------------------------
    static float n124_in[102400], n124_q[25600], n124_k[25600], n124_v[51200];
    load_ifm(dram2, 32307200, n124_in, 102400);
    
    // Tách mảng đa chiều [1, 2, 128, 400] thành Q,K,V dọc theo chiều 128
    int offset_q = 0, offset_k = 0, offset_v = 0;
    
    for (int h = 0; h < 2; h++) { // Lặp qua 2 Heads
        int head_offset = h * 128 * 400; // Head 0 bắt đầu từ 0, Head 1 bắt đầu từ 51200
        
        // Q: Lấy 32 kênh đầu tiên (32 * 400 = 12800 phần tử)
        for (int i = 0; i < 12800; i++) {
            n124_q[offset_q++] = n124_in[head_offset + i];
        }
        
        // K: Lấy 32 kênh tiếp theo
        for (int i = 0; i < 12800; i++) {
            n124_k[offset_k++] = n124_in[head_offset + 12800 + i];
        }
        
        // V: Lấy 64 kênh cuối cùng (64 * 400 = 25600 phần tử)
        for (int i = 0; i < 25600; i++) {
            n124_v[offset_v++] = n124_in[head_offset + 25600 + i];
        }
    }
    
    store_ofm(dram2, 32409600, n124_q, 25600);

    // --------------------------------------------------------
    // NODE 125: Mul (Scale Q bằng node_mul_scalar)
    // --------------------------------------------------------
    static float n125_in[25600], n125_out[25600];
    load_ifm(dram2, 32409600, n125_in, 25600);
    
    // ĐÃ FIX: Hardcode hệ số scale = 1 / sqrt(32)
    float scale = 0.176776695f; 
    node_mul_scalar(n125_in, scale, n125_out, 25600);
    
    store_ofm(dram2, 32435200, n125_out, 25600);

    // --------------------------------------------------------
    // NODE 126: Reshape V
    // --------------------------------------------------------
    static float n126_out[51200];
    node_reshape(n124_v, n126_out, 51200); 
    store_ofm(dram2, 32460800, n126_out, 51200);

    // --------------------------------------------------------
    // NODE 127: Transpose Q_scaled (Manual Loop)
    // Input Q_scaled (n125_out): [1, 2, 32, 400] -> Output Q_T: [1, 2, 400, 32]
    // --------------------------------------------------------
    static float n127_out[25600];
    int idx127 = 0;
    for (int h = 0; h < 2; h++) {
        for (int w = 0; w < 400; w++) {
            for (int c = 0; c < 32; c++) {
                // Đọc từ n125_out (Q_scaled)
                n127_out[idx127++] = n125_out[h * 12800 + c * 400 + w];
            }
        }
    }
    store_ofm(dram2, 32512000, n127_out, 25600);

    // --------------------------------------------------------
    // NODE 128: Depthwise Conv PE
    // --------------------------------------------------------
    static float n128_in[51200], n128_w[1152], n128_b[128], n128_out[51200];
    load_ifm(dram2, 32460800, n128_in, 51200);
    load_weight_bias(dram1, 1211806, n128_w, 1152);
    load_weight_bias(dram1, 1212958, n128_b, 128);
    node_conv_depthwise(n128_in, n128_w, n128_b, n128_out, 1, 128, 20, 20, 128, 3, 1, 1);
    store_ofm(dram2, 32537600, n128_out, 51200);

    // --------------------------------------------------------
    // NODE 129: MatMul (Q_scaled_T * K) (Manual Loop)
    // Q_T: [1, 2, 400, 32] x K: [1, 2, 32, 400] -> [1, 2, 400, 400]
    // --------------------------------------------------------
    static float n129_out[320000];
    for (int h = 0; h < 2; h++) {
        int offset_q_t = h * 12800;  // Q_T: 400x32
        int offset_k   = h * 12800;  // K: 32x400
        int offset_out = h * 160000; // Out: 400x400
        
        for (int i = 0; i < 400; i++) {
            for (int j = 0; j < 400; j++) {
                float sum = 0.0f;
                for (int p = 0; p < 32; p++) {
                    // Q_T[i, p] * K[p, j]
                    float q_val = n127_out[offset_q_t + i * 32 + p];
                    float k_val = n124_k[offset_k + p * 400 + j];
                    sum += q_val * k_val;
                }
                n129_out[offset_out + i * 400 + j] = sum;
            }
        }
    }
    store_ofm(dram2, 32588800, n129_out, 320000);

    // --------------------------------------------------------
    // NODE 130: Softmax
    // --------------------------------------------------------
    static float n130_in[320000], n130_out[320000];
    load_ifm(dram2, 32588800, n130_in, 320000);
    node_softmax(n130_in, n130_out, 800, 400); // 800 hàng, 400 cột
    store_ofm(dram2, 32908800, n130_out, 320000);

    // --------------------------------------------------------
    // NODE 131: Transpose Attention Map (Manual Loop)
    // Attn: [1, 2, 400, 400] -> Hoán vị thành [1, 2, 400, 400] 
    // --------------------------------------------------------
    static float n131_out[320000];
    int idx131 = 0;
    for (int h = 0; h < 2; h++) {
        for (int w = 0; w < 400; w++) {
            for (int h2 = 0; h2 < 400; h2++) {
                n131_out[idx131++] = n130_out[h * 160000 + h2 * 400 + w];
            }
        }
    }
    store_ofm(dram2, 33228800, n131_out, 320000);

    // --------------------------------------------------------
    // NODE 132: MatMul (V * Attn_T) (Manual Loop)
    // V: [1, 2, 64, 400] x Attn_T: [1, 2, 400, 400] -> [1, 2, 64, 400]
    // --------------------------------------------------------
    static float n132_out[51200];
    for (int h = 0; h < 2; h++) {
        int offset_v = h * 25600;       // V: 64x400
        int offset_attn_t = h * 160000; // Attn_T: 400x400
        int offset_out = h * 25600;     // Out: 64x400
        
        for (int i = 0; i < 64; i++) {
            for (int j = 0; j < 400; j++) {
                float sum = 0.0f;
                for (int p = 0; p < 400; p++) {
                    // V[i, p] * Attn_T[p, j]
                    float v_val = n124_v[offset_v + i * 400 + p];
                    float attn_val = n131_out[offset_attn_t + p * 400 + j];
                    sum += v_val * attn_val;
                }
                n132_out[offset_out + i * 400 + j] = sum;
            }
        }
    }
    store_ofm(dram2, 33548800, n132_out, 51200);
    // --------------------------------------------------------
    // NODE 133: Reshape
    // --------------------------------------------------------
    static float n133_in[51200], n133_out[51200];
    load_ifm(dram2, 33548800, n133_in, 51200);
    node_reshape(n133_in, n133_out, 51200);
    store_ofm(dram2, 33600000, n133_out, 51200);

    // --------------------------------------------------------
    // NODE 134: Add (PE + Attn_out)
    // --------------------------------------------------------
    static float n134_i1[51200], n134_i2[51200], n134_out[51200];
    load_ifm(dram2, 33600000, n134_i1, 51200); 
    load_ifm(dram2, 32537600, n134_i2, 51200); 
    node_add(n134_i1, n134_i2, n134_out, 51200);
    store_ofm(dram2, 33651200, n134_out, 51200);

    // --------------------------------------------------------
    // NODE 135: Conv (Proj)
    // --------------------------------------------------------
    static float n135_in[51200], n135_w[16384], n135_b[128], n135_out[51200];
    load_ifm(dram2, 33651200, n135_in, 51200);
    load_weight_bias(dram1, 1213090, n135_w, 16384);
    load_weight_bias(dram1, 1229474, n135_b, 128);
    node_conv(n135_in, n135_w, n135_b, n135_out, 1, 128, 20, 20, 128, 1, 1, 0);
    store_ofm(dram2, 33702400, n135_out, 51200);

    // --------------------------------------------------------
    // NODE 136: Add (Shortcut 1)
    // --------------------------------------------------------
    static float n136_i1[51200], n136_i2[51200], n136_out[51200];
    load_ifm(dram2, 33702400, n136_i1, 51200);
    load_ifm(dram2, 32153600, n136_i2, 51200); 
    node_add(n136_i1, n136_i2, n136_out, 51200);
    store_ofm(dram2, 33753600, n136_out, 51200);

    // --------------------------------------------------------
    // NODE 137, 138, 139: FFN cv1
    // --------------------------------------------------------
    static float n137_in[51200], n137_w[32768], n137_b[256], n137_out[102400];
    load_ifm(dram2, 33753600, n137_in, 51200);
    load_weight_bias(dram1, 1229602, n137_w, 32768);
    load_weight_bias(dram1, 1262370, n137_b, 256);
    node_conv(n137_in, n137_w, n137_b, n137_out, 1, 128, 20, 20, 256, 1, 1, 0);
    store_ofm(dram2, 33804800, n137_out, 102400);

    static float n138_in[102400], n138_out[102400];
    load_ifm(dram2, 33804800, n138_in, 102400);
    node_sigmoid(n138_in, n138_out, 102400);
    store_ofm(dram2, 33907200, n138_out, 102400);

    static float n139_i1[102400], n139_i2[102400], n139_out[102400];
    load_ifm(dram2, 33804800, n139_i1, 102400);
    load_ifm(dram2, 33907200, n139_i2, 102400);
    node_mul(n139_i1, n139_i2, n139_out, 102400);
    store_ofm(dram2, 34009600, n139_out, 102400);

    // --------------------------------------------------------
    // NODE 140: FFN cv2
    // --------------------------------------------------------
    static float n140_in[102400], n140_w[32768], n140_b[128], n140_out[51200];
    load_ifm(dram2, 34009600, n140_in, 102400);
    load_weight_bias(dram1, 1262626, n140_w, 32768);
    load_weight_bias(dram1, 1295394, n140_b, 128);
    node_conv(n140_in, n140_w, n140_b, n140_out, 1, 256, 20, 20, 128, 1, 1, 0);
    store_ofm(dram2, 34112000, n140_out, 51200);

    // --------------------------------------------------------
    // NODE 141: Add (Shortcut 2)
    // --------------------------------------------------------
    static float n141_i1[51200], n141_i2[51200], n141_out[51200];
    load_ifm(dram2, 34112000, n141_i1, 51200); 
    load_ifm(dram2, 33753600, n141_i2, 51200); 
    node_add(n141_i1, n141_i2, n141_out, 51200);
    store_ofm(dram2, 34163200, n141_out, 51200);

    // --------------------------------------------------------
    // NODE 142: Concat
    // --------------------------------------------------------
    static float n142_in1[51200], n142_in2[51200], n142_out[102400];
    float *n142_inputs[2] = {n121_split0, n141_out};
    int n142_sizes[2] = {51200, 51200};
    node_concat(n142_inputs, 2, n142_out, n142_sizes, 102400);
    store_ofm(dram2, 34214400, n142_out, 102400);

    // --------------------------------------------------------
    // NODE 143, 144, 145: cv2 (Conv -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n143_in[102400], n143_w[65536], n143_b[256], n143_out[102400];
    load_ifm(dram2, 34214400, n143_in, 102400);
    load_weight_bias(dram1, 1295522, n143_w, 65536);
    load_weight_bias(dram1, 1361058, n143_b, 256);
    node_conv(n143_in, n143_w, n143_b, n143_out, 1, 256, 20, 20, 256, 1, 1, 0);
    store_ofm(dram2, 34316800, n143_out, 102400);

    static float n144_in[102400], n144_out[102400];
    load_ifm(dram2, 34316800, n144_in, 102400);
    node_sigmoid(n144_in, n144_out, 102400);
    store_ofm(dram2, 34419200, n144_out, 102400);

    static float n145_i1[102400], n145_i2[102400], n145_out[102400];
    load_ifm(dram2, 34316800, n145_i1, 102400);
    load_ifm(dram2, 34419200, n145_i2, 102400);
    node_mul(n145_i1, n145_i2, n145_out, 102400);
    store_ofm(dram2, 34521600, n145_out, 102400); 
}

// ============================================================================
// MODULE 11: Upsample
// Input: 1x256x20x20 | Output: 1x256x40x40 
// ============================================================================
void module_11_upsample(const float *dram1, float *dram2) {
    
    // --------------------------------------------------------
    // NODE 146: Resize (Upsample 2x)
    // --------------------------------------------------------
    static float n146_input[102400];   // 1x256x20x20
    static float n146_output[409600];  // 1x256x40x40
    
    // Đọc Output của Node 145 (Kết thúc C2PSA)
    load_ifm(dram2, 34521600, n146_input, 102400);
    
    // Resize Nearest Neighbor: 256 kênh, từ 20x20 lên 40x40
    node_resize(n146_input, n146_output, 256, 20, 20, 40, 40);
    
    // Store OFM ra DRAM_2
    store_ofm(dram2, 34624000, n146_output, 409600);   // Địa chỉ: 34624000 -> 35033599
}

// ============================================================================
// MODULE 12: Concat
// Input 1: 1x256x40x40 (Upsampled) | Input 2: 1x128x40x40 (Backbone P4)
// Output: 1x384x40x40
// ============================================================================
void module_12_concat(const float *dram1, float *dram2) {
    
    // --------------------------------------------------------
    // NODE 147: Concat (FPN / PANet skip connection)
    // --------------------------------------------------------
    static float n147_in1[409600]; // 256 kênh từ Node 146 (Upsample)
    static float n147_in2[204800]; // 128 kênh từ Node 73 (Module 6 P4)
    static float n147_output[614400]; // 384 kênh sau khi nối
    
    // Load Input 1 (Upsample)
    load_ifm(dram2, 34624000, n147_in1, 409600);
    
    // Load Input 2 (Backbone P4 - Địa chỉ của Node 73)
    load_ifm(dram2, 28979200, n147_in2, 204800);
    
    float *n147_inputs[2] = {n147_in1, n147_in2};
    int n147_sizes[2] = {409600, 204800};
    
    // Gộp 2 tensor lại (mảng C CHW memory format copy tuần tự cực nhanh)
    node_concat(n147_inputs, 2, n147_output, n147_sizes, 614400);
    
    // Store OFM ra DRAM_2
    store_ofm(dram2, 35033600, n147_output, 614400);   // Địa chỉ: 35033600 -> 35647999
}

// ============================================================================
// MODULE 13: C3k2 (Neck - P4)
// Input: 1x384x40x40 (Từ Module 12) | Output: 1x128x40x40
// ============================================================================
void module_13_c3k2(const float *dram1, float *dram2) {
    
    // --------------------------------------------------------
    // NODE 148, 149, 150: cv1 (Conv -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n148_in[614400], n148_w[49152], n148_b[128], n148_out[204800];
    
    // Lấy Input từ Output của Module 12 (Node 147)
    load_ifm(dram2, 35033600, n148_in, 614400);
    load_weight_bias(dram1, 1361318, n148_w, 49152);
    load_weight_bias(dram1, 1410470, n148_b, 128);
    
    node_conv(n148_in, n148_w, n148_b, n148_out, 1, 384, 40, 40, 128, 1, 1, 0);
    store_ofm(dram2, 35648000, n148_out, 204800);

    static float n149_in[204800], n149_out[204800];
    load_ifm(dram2, 35648000, n149_in, 204800);
    node_sigmoid(n149_in, n149_out, 204800);
    store_ofm(dram2, 35852800, n149_out, 204800);

    static float n150_i1[204800], n150_i2[204800], n150_out[204800];
    load_ifm(dram2, 35648000, n150_i1, 204800);
    load_ifm(dram2, 35852800, n150_i2, 204800);
    node_mul(n150_i1, n150_i2, n150_out, 204800);
    store_ofm(dram2, 36057600, n150_out, 204800);

    // --------------------------------------------------------
    // NODE 151: Split (Tách 128 kênh thành 2 luồng 64 kênh)
    // --------------------------------------------------------
    static float n151_in[204800], n151_split0[102400], n151_split1[102400];
    load_ifm(dram2, 36057600, n151_in, 204800);
    
    float *splits_151[2] = {n151_split0, n151_split1};
    node_split(n151_in, splits_151, 2, 102400);
    // Theo cấu trúc YOLO C3k2, split_1 đi vào luồng Bottleneck
    store_ofm(dram2, 36262400, n151_split1, 102400); 

    // --------------------------------------------------------
    // NODE 152, 154, 156: m.0.cv1 (Conv -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n152_in[102400], n152_w[2048], n152_b[32], n152_out[51200];
    load_ifm(dram2, 36262400, n152_in, 102400);
    load_weight_bias(dram1, 1410600, n152_w, 2048);
    load_weight_bias(dram1, 1412648, n152_b, 32);
    node_conv(n152_in, n152_w, n152_b, n152_out, 1, 64, 40, 40, 32, 1, 1, 0);
    store_ofm(dram2, 36364800, n152_out, 51200);

    static float n154_in[51200], n154_out[51200];
    load_ifm(dram2, 36364800, n154_in, 51200);
    node_sigmoid(n154_in, n154_out, 51200);
    store_ofm(dram2, 36467200, n154_out, 51200);

    static float n156_i1[51200], n156_i2[51200], n156_out[51200];
    load_ifm(dram2, 36364800, n156_i1, 51200);
    load_ifm(dram2, 36467200, n156_i2, 51200);
    node_mul(n156_i1, n156_i2, n156_out, 51200);
    store_ofm(dram2, 36569600, n156_out, 51200);

    // --------------------------------------------------------
    // NODE 153, 155, 157: m.0.cv2 (Conv -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n153_w[2048], n153_b[32], n153_out[51200];
    // n152_in (split1) được dùng chung làm input cho cv2
    load_weight_bias(dram1, 1412680, n153_w, 2048);
    load_weight_bias(dram1, 1414728, n153_b, 32);
    node_conv(n152_in, n153_w, n153_b, n153_out, 1, 64, 40, 40, 32, 1, 1, 0);
    store_ofm(dram2, 36416000, n153_out, 51200);

    static float n155_in[51200], n155_out[51200];
    load_ifm(dram2, 36416000, n155_in, 51200);
    node_sigmoid(n155_in, n155_out, 51200);
    store_ofm(dram2, 36518400, n155_out, 51200);

    static float n157_i1[51200], n157_i2[51200], n157_out[51200];
    load_ifm(dram2, 36416000, n157_i1, 51200);
    load_ifm(dram2, 36518400, n157_i2, 51200);
    node_mul(n157_i1, n157_i2, n157_out, 51200);
    store_ofm(dram2, 36620800, n157_out, 51200);

    // --------------------------------------------------------
    // NODE 158, 159, 160: m.0.m.0 phần 1 (Conv 3x3 -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n158_in[51200], n158_w[9216], n158_b[32], n158_out[51200];
    load_ifm(dram2, 36569600, n158_in, 51200); // Lấy input từ SiLU cv1 (Node 156)
    load_weight_bias(dram1, 1414760, n158_w, 9216);
    load_weight_bias(dram1, 1423976, n158_b, 32);
    node_conv(n158_in, n158_w, n158_b, n158_out, 1, 32, 40, 40, 32, 3, 1, 1);
    store_ofm(dram2, 36672000, n158_out, 51200);

    static float n159_in[51200], n159_out[51200];
    load_ifm(dram2, 36672000, n159_in, 51200);
    node_sigmoid(n159_in, n159_out, 51200);
    store_ofm(dram2, 36723200, n159_out, 51200);

    static float n160_i1[51200], n160_i2[51200], n160_out[51200];
    load_ifm(dram2, 36672000, n160_i1, 51200);
    load_ifm(dram2, 36723200, n160_i2, 51200);
    node_mul(n160_i1, n160_i2, n160_out, 51200);
    store_ofm(dram2, 36774400, n160_out, 51200);

    // --------------------------------------------------------
    // NODE 161, 162, 163: m.0.m.0 phần 2 (Conv 3x3 -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n161_in[51200], n161_w[9216], n161_b[32], n161_out[51200];
    load_ifm(dram2, 36774400, n161_in, 51200);
    load_weight_bias(dram1, 1424008, n161_w, 9216);
    load_weight_bias(dram1, 1433224, n161_b, 32);
    node_conv(n161_in, n161_w, n161_b, n161_out, 1, 32, 40, 40, 32, 3, 1, 1);
    store_ofm(dram2, 36825600, n161_out, 51200);

    static float n162_in[51200], n162_out[51200];
    load_ifm(dram2, 36825600, n162_in, 51200);
    node_sigmoid(n162_in, n162_out, 51200);
    store_ofm(dram2, 36876800, n162_out, 51200);

    static float n163_i1[51200], n163_i2[51200], n163_out[51200];
    load_ifm(dram2, 36825600, n163_i1, 51200);
    load_ifm(dram2, 36876800, n163_i2, 51200);
    node_mul(n163_i1, n163_i2, n163_out, 51200);
    store_ofm(dram2, 36928000, n163_out, 51200);

    // --------------------------------------------------------
    // NODE 164: Add (Shortcut Bottleneck 1)
    // --------------------------------------------------------
    static float n164_i1[51200], n164_i2[51200], n164_out[51200];
    // Shortcut lấy từ Input của m.0.m.0 (chính là SiLU cv1: Node 156)
    load_ifm(dram2, 36569600, n164_i1, 51200);
    load_ifm(dram2, 36928000, n164_i2, 51200);
    node_add(n164_i1, n164_i2, n164_out, 51200);
    store_ofm(dram2, 36979200, n164_out, 51200);

    // --------------------------------------------------------
    // NODE 165, 166, 167: m.0.m.1 phần 1 (Conv 3x3 -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n165_in[51200], n165_w[9216], n165_b[32], n165_out[51200];
    load_ifm(dram2, 36979200, n165_in, 51200); // Input từ Add (Node 164)
    load_weight_bias(dram1, 1433256, n165_w, 9216);
    load_weight_bias(dram1, 1442472, n165_b, 32);
    node_conv(n165_in, n165_w, n165_b, n165_out, 1, 32, 40, 40, 32, 3, 1, 1);
    store_ofm(dram2, 37030400, n165_out, 51200);

    static float n166_in[51200], n166_out[51200];
    load_ifm(dram2, 37030400, n166_in, 51200);
    node_sigmoid(n166_in, n166_out, 51200);
    store_ofm(dram2, 37081600, n166_out, 51200);

    static float n167_i1[51200], n167_i2[51200], n167_out[51200];
    load_ifm(dram2, 37030400, n167_i1, 51200);
    load_ifm(dram2, 37081600, n167_i2, 51200);
    node_mul(n167_i1, n167_i2, n167_out, 51200);
    store_ofm(dram2, 37132800, n167_out, 51200);

    // --------------------------------------------------------
    // NODE 168, 169, 170: m.0.m.1 phần 2 (Conv 3x3 -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n168_in[51200], n168_w[9216], n168_b[32], n168_out[51200];
    load_ifm(dram2, 37132800, n168_in, 51200);
    load_weight_bias(dram1, 1442504, n168_w, 9216);
    load_weight_bias(dram1, 1451720, n168_b, 32);
    node_conv(n168_in, n168_w, n168_b, n168_out, 1, 32, 40, 40, 32, 3, 1, 1);
    store_ofm(dram2, 37184000, n168_out, 51200);

    static float n169_in[51200], n169_out[51200];
    load_ifm(dram2, 37184000, n169_in, 51200);
    node_sigmoid(n169_in, n169_out, 51200);
    store_ofm(dram2, 37235200, n169_out, 51200);

    static float n170_i1[51200], n170_i2[51200], n170_out[51200];
    load_ifm(dram2, 37184000, n170_i1, 51200);
    load_ifm(dram2, 37235200, n170_i2, 51200);
    node_mul(n170_i1, n170_i2, n170_out, 51200);
    store_ofm(dram2, 37286400, n170_out, 51200);

    // --------------------------------------------------------
    // NODE 171: Add (Shortcut Bottleneck 2)
    // --------------------------------------------------------
    static float n171_i1[51200], n171_i2[51200], n171_out[51200];
    // Shortcut lấy từ Input của m.0.m.1 (chính là Add Bottleneck 1: Node 164)
    load_ifm(dram2, 36979200, n171_i1, 51200);
    load_ifm(dram2, 37286400, n171_i2, 51200);
    node_add(n171_i1, n171_i2, n171_out, 51200);
    store_ofm(dram2, 37337600, n171_out, 51200);

    // --------------------------------------------------------
    // NODE 172: Concat (Gộp block 2 + m.0.cv2 đầu ra)
    // --------------------------------------------------------
    static float n172_in1[51200], n172_in2[51200], n172_out[102400];
    load_ifm(dram2, 37337600, n172_in1, 51200); // Output block 2 (Node 171)
    load_ifm(dram2, 36620800, n172_in2, 51200); // Output m.0.cv2 (Node 157)
    
    float *n172_inputs[2] = {n172_in1, n172_in2};
    int n172_sizes[2] = {51200, 51200};
    node_concat(n172_inputs, 2, n172_out, n172_sizes, 102400);
    store_ofm(dram2, 37388800, n172_out, 102400);

    // --------------------------------------------------------
    // NODE 173, 174, 175: cv3 (Conv -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n173_in[102400], n173_w[4096], n173_b[64], n173_out[102400];
    load_ifm(dram2, 37388800, n173_in, 102400);
    load_weight_bias(dram1, 1451752, n173_w, 4096);
    load_weight_bias(dram1, 1455848, n173_b, 64);
    node_conv(n173_in, n173_w, n173_b, n173_out, 1, 64, 40, 40, 64, 1, 1, 0);
    store_ofm(dram2, 37491200, n173_out, 102400);

    static float n174_in[102400], n174_out[102400];
    load_ifm(dram2, 37491200, n174_in, 102400);
    node_sigmoid(n174_in, n174_out, 102400);
    store_ofm(dram2, 37593600, n174_out, 102400);

    static float n175_i1[102400], n175_i2[102400], n175_out[102400];
    load_ifm(dram2, 37491200, n175_i1, 102400);
    load_ifm(dram2, 37593600, n175_i2, 102400);
    node_mul(n175_i1, n175_i2, n175_out, 102400);
    store_ofm(dram2, 37696000, n175_out, 102400);

    // --------------------------------------------------------
    // NODE 176: Concat (Gộp split_0, split_1 và cv3 output)
    // --------------------------------------------------------
    static float n176_in3[102400], n176_out[307200];
    // n151_split0 đã có sẵn (nửa đầu Output Node 150)
    // n151_split1 đã có sẵn (Output Node 151)
    load_ifm(dram2, 37696000, n176_in3, 102400); // cv3 output (Node 175)
    
    // Lưu ý: n151_split0 nằm ở mảng tạm nội bộ của n151_in, ta truyền trực tiếp
    float *n176_inputs[3] = {n151_split0, n151_split1, n176_in3};
    int n176_sizes[3] = {102400, 102400, 102400};
    node_concat(n176_inputs, 3, n176_out, n176_sizes, 307200);
    store_ofm(dram2, 37798400, n176_out, 307200);

    // --------------------------------------------------------
    // NODE 177, 178, 179: cv2 (Conv -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n177_in[307200], n177_w[24576], n177_b[128], n177_out[204800];
    load_ifm(dram2, 37798400, n177_in, 307200);
    load_weight_bias(dram1, 1455912, n177_w, 24576);
    load_weight_bias(dram1, 1480488, n177_b, 128);
    node_conv(n177_in, n177_w, n177_b, n177_out, 1, 192, 40, 40, 128, 1, 1, 0);
    store_ofm(dram2, 38105600, n177_out, 204800);

    static float n178_in[204800], n178_out[204800];
    load_ifm(dram2, 38105600, n178_in, 204800);
    node_sigmoid(n178_in, n178_out, 204800);
    store_ofm(dram2, 38310400, n178_out, 204800);

    static float n179_i1[204800], n179_i2[204800], n179_out[204800];
    load_ifm(dram2, 38105600, n179_i1, 204800);
    load_ifm(dram2, 38310400, n179_i2, 204800);
    node_mul(n179_i1, n179_i2, n179_out, 204800);
    store_ofm(dram2, 38515200, n179_out, 204800);
}

// ============================================================================
// MODULE 14: Upsample
// Input: 1x128x40x40 (Từ Module 13) | Output: 1x128x80x80 
// ============================================================================
void module_14_upsample(const float *dram1, float *dram2) {
    
    // --------------------------------------------------------
    // NODE 180: Resize (Upsample 2x)
    // --------------------------------------------------------
    static float n180_input[204800];   // 1x128x40x40 (Output Node 179)
    static float n180_output[819200];  // 1x128x80x80
    
    // Đọc Output của Node 179 (Kết thúc Module 13)
    load_ifm(dram2, 38515200, n180_input, 204800);
    
    // Resize Nearest Neighbor: 128 kênh, từ 40x40 lên 80x80
    node_resize(n180_input, n180_output, 128, 40, 40, 80, 80);
    
    // Store OFM ra DRAM_2
    store_ofm(dram2, 38720000, n180_output, 819200);   // Địa chỉ: 38720000 -> 39539199
}

// ============================================================================
// MODULE 15: Concat
// Input 1: 1x128x80x80 (Upsampled) | Input 2: 1x128x80x80 (Backbone P3)
// Output: 1x256x80x80
// ============================================================================
void module_15_concat(const float *dram1, float *dram2) {
    
    // --------------------------------------------------------
    // NODE 181: Concat (FPN skip connection P3)
    // --------------------------------------------------------
    static float n181_in1[819200];  // 128 kênh từ Node 180 (Upsample)
    static float n181_in2[819200];  // 128 kênh từ Node 38 (Backbone P3)
    static float n181_output[1638400]; // 256 kênh sau khi nối
    
    // Load Input 1 (Upsample)
    load_ifm(dram2, 38720000, n181_in1, 819200);
    
    // Load Input 2 (Backbone P3 - Output của Node 38)
    load_ifm(dram2, 24678400, n181_in2, 819200);
    
    float *n181_inputs[2] = {n181_in1, n181_in2};
    int n181_sizes[2] = {819200, 819200};
    
    // Gộp 2 tensor lại (mảng phẳng)
    node_concat(n181_inputs, 2, n181_output, n181_sizes, 1638400);
    
    // Store OFM ra DRAM_2
    store_ofm(dram2, 39539200, n181_output, 1638400);   // Địa chỉ: 39539200 -> 41177599
}

// ============================================================================
// MODULE 16: C3k2 -> Detect P3
// Input: 1x256x80x80 (Từ Module 15) | Output: 1x64x80x80 
// ============================================================================
void module_16_c3k2_detect_P3(const float *dram1, float *dram2) {
    
    // --------------------------------------------------------
    // NODE 182, 183, 184: cv1 (Conv -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n182_in[1638400], n182_w[16384], n182_b[64], n182_out[409600];
    
    // Đọc Input từ Output của Module 15 (Node 181)
    load_ifm(dram2, 39539200, n182_in, 1638400);
    load_weight_bias(dram1, 1480620, n182_w, 16384);
    load_weight_bias(dram1, 1497004, n182_b, 64);
    
    node_conv(n182_in, n182_w, n182_b, n182_out, 1, 256, 80, 80, 64, 1, 1, 0);
    store_ofm(dram2, 41177600, n182_out, 409600);

    static float n183_in[409600], n183_out[409600];
    load_ifm(dram2, 41177600, n183_in, 409600);
    node_sigmoid(n183_in, n183_out, 409600);
    store_ofm(dram2, 41587200, n183_out, 409600);

    static float n184_i1[409600], n184_i2[409600], n184_out[409600];
    load_ifm(dram2, 41177600, n184_i1, 409600);
    load_ifm(dram2, 41587200, n184_i2, 409600);
    node_mul(n184_i1, n184_i2, n184_out, 409600);
    store_ofm(dram2, 41996800, n184_out, 409600);

    // --------------------------------------------------------
    // NODE 185: Split (Tách 64 kênh thành 2 luồng 32 kênh)
    // --------------------------------------------------------
    static float n185_in[409600], n185_split0[204800], n185_split1[204800];
    load_ifm(dram2, 41996800, n185_in, 409600);
    
    float *splits_185[2] = {n185_split0, n185_split1};
    node_split(n185_in, splits_185, 2, 204800);
    store_ofm(dram2, 42406400, n185_split1, 204800); // Lưu luồng Bottleneck

    // --------------------------------------------------------
    // NODE 186, 188, 190: m.0.cv1 (Conv -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n186_in[204800], n186_w[512], n186_b[16], n186_out[102400];
    load_ifm(dram2, 42406400, n186_in, 204800);
    load_weight_bias(dram1, 1497070, n186_w, 512);
    load_weight_bias(dram1, 1497582, n186_b, 16);
    node_conv(n186_in, n186_w, n186_b, n186_out, 1, 32, 80, 80, 16, 1, 1, 0);
    store_ofm(dram2, 42611200, n186_out, 102400);

    static float n188_in[102400], n188_out[102400];
    load_ifm(dram2, 42611200, n188_in, 102400);
    node_sigmoid(n188_in, n188_out, 102400);
    store_ofm(dram2, 42816000, n188_out, 102400);

    static float n190_i1[102400], n190_i2[102400], n190_out[102400];
    load_ifm(dram2, 42611200, n190_i1, 102400);
    load_ifm(dram2, 42816000, n190_i2, 102400);
    node_mul(n190_i1, n190_i2, n190_out, 102400);
    store_ofm(dram2, 43020800, n190_out, 102400);

    // --------------------------------------------------------
    // NODE 187, 189, 191: m.0.cv2 (Conv -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n187_w[512], n187_b[16], n187_out[102400];
    // Dùng n186_in (split1) làm input chung
    load_weight_bias(dram1, 1497598, n187_w, 512);
    load_weight_bias(dram1, 1498110, n187_b, 16);
    node_conv(n186_in, n187_w, n187_b, n187_out, 1, 32, 80, 80, 16, 1, 1, 0);
    store_ofm(dram2, 42713600, n187_out, 102400);

    static float n189_in[102400], n189_out[102400];
    load_ifm(dram2, 42713600, n189_in, 102400);
    node_sigmoid(n189_in, n189_out, 102400);
    store_ofm(dram2, 42918400, n189_out, 102400);

    static float n191_i1[102400], n191_i2[102400], n191_out[102400];
    load_ifm(dram2, 42713600, n191_i1, 102400);
    load_ifm(dram2, 42918400, n191_i2, 102400);
    node_mul(n191_i1, n191_i2, n191_out, 102400);
    store_ofm(dram2, 43123200, n191_out, 102400);

    // --------------------------------------------------------
    // NODE 192, 193, 194: m.0.m.0 phần 1 (Conv 3x3 -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n192_in[102400], n192_w[2304], n192_b[16], n192_out[102400];
    load_ifm(dram2, 43020800, n192_in, 102400);
    load_weight_bias(dram1, 1498126, n192_w, 2304);
    load_weight_bias(dram1, 1500430, n192_b, 16);
    node_conv(n192_in, n192_w, n192_b, n192_out, 1, 16, 80, 80, 16, 3, 1, 1);
    store_ofm(dram2, 43225600, n192_out, 102400);

    static float n193_in[102400], n193_out[102400];
    load_ifm(dram2, 43225600, n193_in, 102400);
    node_sigmoid(n193_in, n193_out, 102400);
    store_ofm(dram2, 43328000, n193_out, 102400);

    static float n194_i1[102400], n194_i2[102400], n194_out[102400];
    load_ifm(dram2, 43225600, n194_i1, 102400);
    load_ifm(dram2, 43328000, n194_i2, 102400);
    node_mul(n194_i1, n194_i2, n194_out, 102400);
    store_ofm(dram2, 43430400, n194_out, 102400);

    // --------------------------------------------------------
    // NODE 195, 196, 197: m.0.m.0 phần 2 (Conv 3x3 -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n195_in[102400], n195_w[2304], n195_b[16], n195_out[102400];
    load_ifm(dram2, 43430400, n195_in, 102400);
    load_weight_bias(dram1, 1500446, n195_w, 2304);
    load_weight_bias(dram1, 1502750, n195_b, 16);
    node_conv(n195_in, n195_w, n195_b, n195_out, 1, 16, 80, 80, 16, 3, 1, 1);
    store_ofm(dram2, 43532800, n195_out, 102400);

    static float n196_in[102400], n196_out[102400];
    load_ifm(dram2, 43532800, n196_in, 102400);
    node_sigmoid(n196_in, n196_out, 102400);
    store_ofm(dram2, 43635200, n196_out, 102400);

    static float n197_i1[102400], n197_i2[102400], n197_out[102400];
    load_ifm(dram2, 43532800, n197_i1, 102400);
    load_ifm(dram2, 43635200, n197_i2, 102400);
    node_mul(n197_i1, n197_i2, n197_out, 102400);
    store_ofm(dram2, 43737600, n197_out, 102400);

    // --------------------------------------------------------
    // NODE 198: Add (Shortcut Bottleneck 1)
    // --------------------------------------------------------
    static float n198_i1[102400], n198_i2[102400], n198_out[102400];
    // Lấy Input của m.0.m.0 (Node 190)
    load_ifm(dram2, 43020800, n198_i1, 102400);
    load_ifm(dram2, 43737600, n198_i2, 102400);
    node_add(n198_i1, n198_i2, n198_out, 102400);
    store_ofm(dram2, 43840000, n198_out, 102400);

    // --------------------------------------------------------
    // NODE 199, 200, 201: m.0.m.1 phần 1 (Conv 3x3 -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n199_in[102400], n199_w[2304], n199_b[16], n199_out[102400];
    load_ifm(dram2, 43840000, n199_in, 102400);
    load_weight_bias(dram1, 1502766, n199_w, 2304);
    load_weight_bias(dram1, 1505070, n199_b, 16);
    node_conv(n199_in, n199_w, n199_b, n199_out, 1, 16, 80, 80, 16, 3, 1, 1);
    store_ofm(dram2, 43942400, n199_out, 102400);

    static float n200_in[102400], n200_out[102400];
    load_ifm(dram2, 43942400, n200_in, 102400);
    node_sigmoid(n200_in, n200_out, 102400);
    store_ofm(dram2, 44044800, n200_out, 102400);

    static float n201_i1[102400], n201_i2[102400], n201_out[102400];
    load_ifm(dram2, 43942400, n201_i1, 102400);
    load_ifm(dram2, 44044800, n201_i2, 102400);
    node_mul(n201_i1, n201_i2, n201_out, 102400);
    store_ofm(dram2, 44147200, n201_out, 102400);

    // --------------------------------------------------------
    // NODE 202, 203, 204: m.0.m.1 phần 2 (Conv 3x3 -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n202_in[102400], n202_w[2304], n202_b[16], n202_out[102400];
    load_ifm(dram2, 44147200, n202_in, 102400);
    load_weight_bias(dram1, 1505086, n202_w, 2304);
    load_weight_bias(dram1, 1507390, n202_b, 16);
    node_conv(n202_in, n202_w, n202_b, n202_out, 1, 16, 80, 80, 16, 3, 1, 1);
    store_ofm(dram2, 44249600, n202_out, 102400);

    static float n203_in[102400], n203_out[102400];
    load_ifm(dram2, 44249600, n203_in, 102400);
    node_sigmoid(n203_in, n203_out, 102400);
    store_ofm(dram2, 44352000, n203_out, 102400);

    static float n204_i1[102400], n204_i2[102400], n204_out[102400];
    load_ifm(dram2, 44249600, n204_i1, 102400);
    load_ifm(dram2, 44352000, n204_i2, 102400);
    node_mul(n204_i1, n204_i2, n204_out, 102400);
    store_ofm(dram2, 44454400, n204_out, 102400);

    // --------------------------------------------------------
    // NODE 205: Add (Shortcut Bottleneck 2)
    // --------------------------------------------------------
    static float n205_i1[102400], n205_i2[102400], n205_out[102400];
    // Lấy Input của m.0.m.1 (Node 198)
    load_ifm(dram2, 43840000, n205_i1, 102400);
    load_ifm(dram2, 44454400, n205_i2, 102400);
    node_add(n205_i1, n205_i2, n205_out, 102400);
    store_ofm(dram2, 44556800, n205_out, 102400);

    // --------------------------------------------------------
    // NODE 206: Concat (Gộp block 2 + m.0.cv2 đầu ra)
    // --------------------------------------------------------
    static float n206_in1[102400], n206_in2[102400], n206_out[204800];
    load_ifm(dram2, 44556800, n206_in1, 102400); // Output block 2 (Node 205)
    load_ifm(dram2, 43123200, n206_in2, 102400); // Output m.0.cv2 (Node 191)
    
    float *n206_inputs[2] = {n206_in1, n206_in2};
    int n206_sizes[2] = {102400, 102400};
    node_concat(n206_inputs, 2, n206_out, n206_sizes, 204800);
    store_ofm(dram2, 44659200, n206_out, 204800);

    // --------------------------------------------------------
    // NODE 207, 208, 209: cv3 (Conv -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n207_in[204800], n207_w[1024], n207_b[32], n207_out[204800];
    load_ifm(dram2, 44659200, n207_in, 204800);
    load_weight_bias(dram1, 1507406, n207_w, 1024);
    load_weight_bias(dram1, 1508430, n207_b, 32);
    node_conv(n207_in, n207_w, n207_b, n207_out, 1, 32, 80, 80, 32, 1, 1, 0);
    store_ofm(dram2, 44864000, n207_out, 204800);

    static float n208_in[204800], n208_out[204800];
    load_ifm(dram2, 44864000, n208_in, 204800);
    node_sigmoid(n208_in, n208_out, 204800);
    store_ofm(dram2, 45068800, n208_out, 204800);

    static float n209_i1[204800], n209_i2[204800], n209_out[204800];
    load_ifm(dram2, 44864000, n209_i1, 204800);
    load_ifm(dram2, 45068800, n209_i2, 204800);
    node_mul(n209_i1, n209_i2, n209_out, 204800);
    store_ofm(dram2, 45273600, n209_out, 204800);

    // --------------------------------------------------------
    // NODE 210: Concat (Gộp split_0, split_1 và cv3 output)
    // --------------------------------------------------------
    static float n210_in3[204800], n210_out[614400];
    load_ifm(dram2, 45273600, n210_in3, 204800);
    
    // n185_split0 và n185_split1 có sẵn
    float *n210_inputs[3] = {n185_split0, n185_split1, n210_in3};
    int n210_sizes[3] = {204800, 204800, 204800};
    node_concat(n210_inputs, 3, n210_out, n210_sizes, 614400);
    store_ofm(dram2, 45478400, n210_out, 614400);

    // --------------------------------------------------------
    // NODE 211, 212, 213: cv2 (Conv -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n211_in[614400], n211_w[6144], n211_b[64], n211_out[409600];
    load_ifm(dram2, 45478400, n211_in, 614400);
    load_weight_bias(dram1, 1508462, n211_w, 6144);
    load_weight_bias(dram1, 1514606, n211_b, 64);
    node_conv(n211_in, n211_w, n211_b, n211_out, 1, 96, 80, 80, 64, 1, 1, 0);
    store_ofm(dram2, 46092800, n211_out, 409600);

    static float n212_in[409600], n212_out[409600];
    load_ifm(dram2, 46092800, n212_in, 409600);
    node_sigmoid(n212_in, n212_out, 409600);
    store_ofm(dram2, 46502400, n212_out, 409600);

    static float n213_i1[409600], n213_i2[409600], n213_out[409600];
    load_ifm(dram2, 46092800, n213_i1, 409600);
    load_ifm(dram2, 46502400, n213_i2, 409600);
    node_mul(n213_i1, n213_i2, n213_out, 409600);
    store_ofm(dram2, 46912000, n213_out, 409600); 
}

// ============================================================================
// MODULE 17: Conv Downsample (P3 -> P4)
// Input: 1x64x80x80 (Từ Module 16) | Output: 1x64x40x40 
// ============================================================================
void module_17_conv_downsample(const float *dram1, float *dram2) {
    
    // --------------------------------------------------------
    // NODE 214: Conv (k=3, s=2, p=1 để giảm kích thước ảnh 80->40)
    // --------------------------------------------------------
    static float n214_in[409600];   // 1x64x80x80 (Output của Node 213)
    static float n214_w[36864];     // 64x64x3x3
    static float n214_b[64];        // 64
    static float n214_out[102400];  // 1x64x40x40
    
    // Đọc Input từ Output của Module 16 (Node 213)
    load_ifm(dram2, 46912000, n214_in, 409600);
    
    load_weight_bias(dram1, 1514670, n214_w, 36864);
    load_weight_bias(dram1, 1551534, n214_b, 64);
    
    // Thực thi Compute (Batch=1, InC=64, InH=80, InW=80, OutC=64, K=3, S=2, P=1)
    node_conv(n214_in, n214_w, n214_b, n214_out, 1, 64, 80, 80, 64, 3, 2, 1);
    
    // Store OFM ra DRAM_2
    store_ofm(dram2, 47321600, n214_out, 102400);

    // --------------------------------------------------------
    // NODE 217: Sigmoid
    // --------------------------------------------------------
    static float n217_in[102400];
    static float n217_out[102400];
    
    load_ifm(dram2, 47321600, n217_in, 102400);
    node_sigmoid(n217_in, n217_out, 102400);
    store_ofm(dram2, 47936000, n217_out, 102400);

    // --------------------------------------------------------
    // NODE 220: Mul (SiLU = Conv_Output * Sigmoid_Output)
    // --------------------------------------------------------
    static float n220_i1[102400];
    static float n220_i2[102400];
    static float n220_out[102400];
    
    load_ifm(dram2, 47321600, n220_i1, 102400);
    load_ifm(dram2, 47936000, n220_i2, 102400);
    node_mul(n220_i1, n220_i2, n220_out, 102400);
    
    store_ofm(dram2, 48550400, n220_out, 102400);
}

// ============================================================================
// MODULE 18: Concat
// Input 1: 1x64x40x40 (Từ Module 17 - Downsampled P3)
// Input 2: 1x128x40x40 (Từ Module 13 - Nhánh P4)
// Output: 1x192x40x40
// ============================================================================
void module_18_concat(const float *dram1, float *dram2) {
    
    // --------------------------------------------------------
    // NODE 223: Concat (PANet skip connection P3 -> P4)
    // --------------------------------------------------------
    static float n223_in1[102400]; // 64 kênh từ Node 220 (Module 17)
    static float n223_in2[204800]; // 128 kênh từ Node 179 (Module 13)
    static float n223_output[307200]; // 192 kênh sau khi nối (192 * 40 * 40 = 307200)
    
    // Load Input 1 (Downsampled P3 - Địa chỉ Output của Node 220)
    load_ifm(dram2, 48550400, n223_in1, 102400);
    
    // Load Input 2 (Nhánh P4 hiện tại - Địa chỉ Output của Node 179)
    load_ifm(dram2, 38515200, n223_in2, 204800);
    
    float *n223_inputs[2] = {n223_in1, n223_in2};
    int n223_sizes[2] = {102400, 204800}; 
    
    // Nối tensor
    node_concat(n223_inputs, 2, n223_output, n223_sizes, 307200);
    
    // Store OFM ra DRAM_2
    store_ofm(dram2, 49164800, n223_output, 307200);   // Địa chỉ: 49164800 -> 49471999
}

// ============================================================================
// MODULE 19: C3k2 -> Detect P4
// Input: 1x192x40x40 (Từ Module 18) | Output: 1x128x40x40 
// ============================================================================
void module_19_c3k2_detect_P4(const float *dram1, float *dram2) {
    
    // --------------------------------------------------------
    // NODE 226, 229, 232: cv1 (Conv -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n226_in[307200], n226_w[24576], n226_b[128], n226_out[204800];
    
    // Lấy Input từ Output của Module 18 (Node 223)
    load_ifm(dram2, 49164800, n226_in, 307200);
    load_weight_bias(dram1, 1567950, n226_w, 24576);
    load_weight_bias(dram1, 1592526, n226_b, 128);
    
    node_conv(n226_in, n226_w, n226_b, n226_out, 1, 192, 40, 40, 128, 1, 1, 0);
    store_ofm(dram2, 49984000, n226_out, 204800);

    static float n229_in[204800], n229_out[204800];
    load_ifm(dram2, 49984000, n229_in, 204800);
    node_sigmoid(n229_in, n229_out, 204800);
    store_ofm(dram2, 50700800, n229_out, 204800);

    static float n232_i1[204800], n232_i2[204800], n232_out[204800];
    load_ifm(dram2, 49984000, n232_i1, 204800);
    load_ifm(dram2, 50700800, n232_i2, 204800);
    node_mul(n232_i1, n232_i2, n232_out, 204800);
    store_ofm(dram2, 51417600, n232_out, 204800);

    // --------------------------------------------------------
    // NODE 235: Split (Tách 128 kênh thành 2 luồng 64 kênh)
    // --------------------------------------------------------
    static float n235_in[204800], n235_split0[102400], n235_split1[102400];
    load_ifm(dram2, 51417600, n235_in, 204800);
    
    float *splits_235[2] = {n235_split0, n235_split1};
    node_split(n235_in, splits_235, 2, 102400);
    store_ofm(dram2, 52057600, n235_split1, 102400); // Lưu luồng Bottleneck

    // --------------------------------------------------------
    // NODE 238, 241, 244: m.0.cv1 (Conv -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n238_in[102400], n238_w[2048], n238_b[32], n238_out[51200];
    load_ifm(dram2, 52057600, n238_in, 102400);
    load_weight_bias(dram1, 1593367, n238_w, 2048);
    load_weight_bias(dram1, 1595415, n238_b, 32);
    node_conv(n238_in, n238_w, n238_b, n238_out, 1, 64, 40, 40, 32, 1, 1, 0);
    store_ofm(dram2, 52595200, n238_out, 51200);

    static float n241_in[51200], n241_out[51200];
    load_ifm(dram2, 52595200, n241_in, 51200);
    node_sigmoid(n241_in, n241_out, 51200);
    store_ofm(dram2, 53107200, n241_out, 51200);

    static float n244_i1[51200], n244_i2[51200], n244_out[51200];
    load_ifm(dram2, 52595200, n244_i1, 51200);
    load_ifm(dram2, 53107200, n244_i2, 51200);
    node_mul(n244_i1, n244_i2, n244_out, 51200);
    store_ofm(dram2, 53619200, n244_out, 51200);

    // --------------------------------------------------------
    // NODE 239, 242, 245: m.0.cv2 (Conv -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n239_w[2048], n239_b[32], n239_out[51200];
    // Dùng n238_in (split1) làm input chung
    load_weight_bias(dram1, 1595447, n239_w, 2048);
    load_weight_bias(dram1, 1597495, n239_b, 32);
    node_conv(n238_in, n239_w, n239_b, n239_out, 1, 64, 40, 40, 32, 1, 1, 0);
    store_ofm(dram2, 52646400, n239_out, 51200);

    static float n242_in[51200], n242_out[51200];
    load_ifm(dram2, 52646400, n242_in, 51200);
    node_sigmoid(n242_in, n242_out, 51200);
    store_ofm(dram2, 53158400, n242_out, 51200);

    static float n245_i1[51200], n245_i2[51200], n245_out[51200];
    load_ifm(dram2, 52646400, n245_i1, 51200);
    load_ifm(dram2, 53158400, n245_i2, 51200);
    node_mul(n245_i1, n245_i2, n245_out, 51200);
    store_ofm(dram2, 53670400, n245_out, 51200);

    // --------------------------------------------------------
    // NODE 247, 249, 251: m.0.m.0 phần 1 (Conv 3x3 -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n247_in[51200], n247_w[9216], n247_b[32], n247_out[51200];
    load_ifm(dram2, 53619200, n247_in, 51200);
    load_weight_bias(dram1, 1601687, n247_w, 9216);
    load_weight_bias(dram1, 1610903, n247_b, 32);
    node_conv(n247_in, n247_w, n247_b, n247_out, 1, 32, 40, 40, 32, 3, 1, 1);
    store_ofm(dram2, 54131200, n247_out, 51200);

    static float n249_in[51200], n249_out[51200];
    load_ifm(dram2, 54131200, n249_in, 51200);
    node_sigmoid(n249_in, n249_out, 51200);
    store_ofm(dram2, 54592000, n249_out, 51200);

    static float n251_i1[51200], n251_i2[51200], n251_out[51200];
    load_ifm(dram2, 54131200, n251_i1, 51200);
    load_ifm(dram2, 54592000, n251_i2, 51200);
    node_mul(n251_i1, n251_i2, n251_out, 51200);
    store_ofm(dram2, 54649600, n251_out, 51200);

    // --------------------------------------------------------
    // NODE 253, 254, 255: m.0.m.0 phần 2 (Conv 3x3 -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n253_in[51200], n253_w[9216], n253_b[32], n253_out[51200];
    load_ifm(dram2, 54649600, n253_in, 51200);
    load_weight_bias(dram1, 1611003, n253_w, 9216);
    load_weight_bias(dram1, 1620219, n253_b, 32);
    node_conv(n253_in, n253_w, n253_b, n253_out, 1, 32, 40, 40, 32, 3, 1, 1);
    store_ofm(dram2, 54707200, n253_out, 51200);

    static float n254_in[51200], n254_out[51200];
    load_ifm(dram2, 54707200, n254_in, 51200);
    node_sigmoid(n254_in, n254_out, 51200);
    store_ofm(dram2, 54758400, n254_out, 51200);

    static float n255_i1[51200], n255_i2[51200], n255_out[51200];
    load_ifm(dram2, 54707200, n255_i1, 51200);
    load_ifm(dram2, 54758400, n255_i2, 51200);
    node_mul(n255_i1, n255_i2, n255_out, 51200);
    store_ofm(dram2, 54809600, n255_out, 51200);

    // --------------------------------------------------------
    // NODE 256: Add (Shortcut Bottleneck 1)
    // --------------------------------------------------------
    static float n256_i1[51200], n256_i2[51200], n256_out[51200];
    // Lấy Input của m.0.m.0 (Node 244)
    load_ifm(dram2, 53619200, n256_i1, 51200);
    load_ifm(dram2, 54809600, n256_i2, 51200);
    node_add(n256_i1, n256_i2, n256_out, 51200);
    store_ofm(dram2, 54860800, n256_out, 51200);

    // --------------------------------------------------------
    // NODE 257, 258, 259: m.0.m.1 phần 1 (Conv 3x3 -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n257_in[51200], n257_w[9216], n257_b[32], n257_out[51200];
    load_ifm(dram2, 54860800, n257_in, 51200);
    load_weight_bias(dram1, 1620251, n257_w, 9216);
    load_weight_bias(dram1, 1629467, n257_b, 32);
    node_conv(n257_in, n257_w, n257_b, n257_out, 1, 32, 40, 40, 32, 3, 1, 1);
    store_ofm(dram2, 54912000, n257_out, 51200);

    static float n258_in[51200], n258_out[51200];
    load_ifm(dram2, 54912000, n258_in, 51200);
    node_sigmoid(n258_in, n258_out, 51200);
    store_ofm(dram2, 54963200, n258_out, 51200);

    static float n259_i1[51200], n259_i2[51200], n259_out[51200];
    load_ifm(dram2, 54912000, n259_i1, 51200);
    load_ifm(dram2, 54963200, n259_i2, 51200);
    node_mul(n259_i1, n259_i2, n259_out, 51200);
    store_ofm(dram2, 55014400, n259_out, 51200);

    // --------------------------------------------------------
    // NODE 260, 261, 262: m.0.m.1 phần 2 (Conv 3x3 -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n260_in[51200], n260_w[9216], n260_b[32], n260_out[51200];
    load_ifm(dram2, 55014400, n260_in, 51200);
    load_weight_bias(dram1, 1629499, n260_w, 9216);
    load_weight_bias(dram1, 1638715, n260_b, 32);
    node_conv(n260_in, n260_w, n260_b, n260_out, 1, 32, 40, 40, 32, 3, 1, 1);
    store_ofm(dram2, 55065600, n260_out, 51200);

    static float n261_in[51200], n261_out[51200];
    load_ifm(dram2, 55065600, n261_in, 51200);
    node_sigmoid(n261_in, n261_out, 51200);
    store_ofm(dram2, 55116800, n261_out, 51200);

    static float n262_i1[51200], n262_i2[51200], n262_out[51200];
    load_ifm(dram2, 55065600, n262_i1, 51200);
    load_ifm(dram2, 55116800, n262_i2, 51200);
    node_mul(n262_i1, n262_i2, n262_out, 51200);
    store_ofm(dram2, 55168000, n262_out, 51200);

    // --------------------------------------------------------
    // NODE 263: Add (Shortcut Bottleneck 2)
    // --------------------------------------------------------
    static float n263_i1[51200], n263_i2[51200], n263_out[51200];
    // Lấy Input của m.0.m.1 (Node 256)
    load_ifm(dram2, 54860800, n263_i1, 51200);
    load_ifm(dram2, 55168000, n263_i2, 51200);
    node_add(n263_i1, n263_i2, n263_out, 51200);
    store_ofm(dram2, 55219200, n263_out, 51200);

    // --------------------------------------------------------
    // NODE 264: Concat (Gộp block 2 + m.0.cv2 đầu ra)
    // --------------------------------------------------------
    static float n264_in1[51200], n264_in2[51200], n264_out[102400];
    load_ifm(dram2, 55219200, n264_in1, 51200); // Output block 2 (Node 263)
    load_ifm(dram2, 53670400, n264_in2, 51200); // Output m.0.cv2 (Node 245)
    
    float *n264_inputs[2] = {n264_in1, n264_in2};
    int n264_sizes[2] = {51200, 51200};
    node_concat(n264_inputs, 2, n264_out, n264_sizes, 102400);
    store_ofm(dram2, 55270400, n264_out, 102400);

    // --------------------------------------------------------
    // NODE 265, 266, 267: cv3 (Conv -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n265_in[102400], n265_w[4096], n265_b[64], n265_out[102400];
    load_ifm(dram2, 55270400, n265_in, 102400);
    load_weight_bias(dram1, 1638747, n265_w, 4096);
    load_weight_bias(dram1, 1642843, n265_b, 64);
    node_conv(n265_in, n265_w, n265_b, n265_out, 1, 64, 40, 40, 64, 1, 1, 0);
    store_ofm(dram2, 55372800, n265_out, 102400);

    static float n266_in[102400], n266_out[102400];
    load_ifm(dram2, 55372800, n266_in, 102400);
    node_sigmoid(n266_in, n266_out, 102400);
    store_ofm(dram2, 55475200, n266_out, 102400);

    static float n267_i1[102400], n267_i2[102400], n267_out[102400];
    load_ifm(dram2, 55372800, n267_i1, 102400);
    load_ifm(dram2, 55475200, n267_i2, 102400);
    node_mul(n267_i1, n267_i2, n267_out, 102400);
    store_ofm(dram2, 55577600, n267_out, 102400);

    // --------------------------------------------------------
    // NODE 268: Concat (Gộp split_0, split_1 và cv3 output)
    // --------------------------------------------------------
    static float n268_in3[102400], n268_out[307200];
    load_ifm(dram2, 55577600, n268_in3, 102400);
    
    float *n268_inputs[3] = {n235_split0, n235_split1, n268_in3};
    int n268_sizes[3] = {102400, 102400, 102400};
    node_concat(n268_inputs, 3, n268_out, n268_sizes, 307200);
    store_ofm(dram2, 55680000, n268_out, 307200);

    // --------------------------------------------------------
    // NODE 269, 270, 271: cv2 (Conv -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n269_in[307200], n269_w[24576], n269_b[128], n269_out[204800];
    load_ifm(dram2, 55680000, n269_in, 307200);
    load_weight_bias(dram1, 1642907, n269_w, 24576);
    load_weight_bias(dram1, 1667483, n269_b, 128);
    node_conv(n269_in, n269_w, n269_b, n269_out, 1, 192, 40, 40, 128, 1, 1, 0);
    store_ofm(dram2, 55987200, n269_out, 204800);

    static float n270_in[204800], n270_out[204800];
    load_ifm(dram2, 55987200, n270_in, 204800);
    node_sigmoid(n270_in, n270_out, 204800);
    store_ofm(dram2, 56192000, n270_out, 204800);

    static float n271_i1[204800], n271_i2[204800], n271_out[204800];
    load_ifm(dram2, 55987200, n271_i1, 204800);
    load_ifm(dram2, 56192000, n271_i2, 204800);
    node_mul(n271_i1, n271_i2, n271_out, 204800);
    store_ofm(dram2, 56396800, n271_out, 204800);
}

// ============================================================================
// MODULE 20: Conv Downsample (P4 -> P5)
// Input: 1x128x40x40 (Từ Module 19) | Output: 1x128x20x20 
// ============================================================================
void module_20_conv_downsample(const float *dram1, float *dram2) {
    
    // --------------------------------------------------------
    // NODE 272: Conv (k=3, s=2, p=1 để giảm kích thước ảnh 40->20)
    // --------------------------------------------------------
    static float n272_in[204800];   // 1x128x40x40 (Output của Node 271)
    static float n272_w[147456];    // 128x128x3x3
    static float n272_b[128];       // 128
    static float n272_out[51200];   // 1x128x20x20
    
    // Đọc Input từ Output của Module 19 (Node 271)
    load_ifm(dram2, 56396800, n272_in, 204800);
    
    load_weight_bias(dram1, 1667611, n272_w, 147456);
    load_weight_bias(dram1, 1815067, n272_b, 128);
    
    // Thực thi Compute (Batch=1, InC=128, InH=40, InW=40, OutC=128, K=3, S=2, P=1)
    node_conv(n272_in, n272_w, n272_b, n272_out, 1, 128, 40, 40, 128, 3, 2, 1);
    
    // Store OFM ra DRAM_2
    store_ofm(dram2, 56601600, n272_out, 51200);

    // --------------------------------------------------------
    // NODE 275: Sigmoid
    // --------------------------------------------------------
    static float n275_in[51200];
    static float n275_out[51200];
    
    load_ifm(dram2, 56601600, n275_in, 51200);
    node_sigmoid(n275_in, n275_out, 51200);
    store_ofm(dram2, 56883200, n275_out, 51200);

    // --------------------------------------------------------
    // NODE 278: Mul (SiLU = Conv_Output * Sigmoid_Output)
    // --------------------------------------------------------
    static float n278_i1[51200];
    static float n278_i2[51200];
    static float n278_out[51200];
    
    load_ifm(dram2, 56601600, n278_i1, 51200);
    load_ifm(dram2, 56883200, n278_i2, 51200);
    node_mul(n278_i1, n278_i2, n278_out, 51200);
    
    store_ofm(dram2, 57164800, n278_out, 51200);
}

// ============================================================================
// MODULE 21: Concat (P5 Head)
// Input 1: 1x128x20x20 (Từ Module 20 - Downsampled P4)
// Input 2: 1x256x20x20 (Từ Module 10 - C2PSA P5)
// Output: 1x384x20x20
// ============================================================================
void module_21_concat(const float *dram1, float *dram2) {
    
    // --------------------------------------------------------
    // NODE 281: Concat (PANet skip connection)
    // --------------------------------------------------------
    static float n281_in1[51200];  // 128 kênh từ Node 278 (Module 20)
    static float n281_in2[102400]; // 256 kênh từ Node 145 (Module 10)
    static float n281_output[153600]; // 384 kênh sau khi nối (384 * 20 * 20 = 153600)
    
    // Load Input 1 (Downsampled P4 - Địa chỉ Output của Node 278)
    load_ifm(dram2, 57164800, n281_in1, 51200);
    
    // Load Input 2 (C2PSA P5 - Địa chỉ Output của Node 145)
    load_ifm(dram2, 34521600, n281_in2, 102400);
    
    float *n281_inputs[2] = {n281_in1, n281_in2};
    int n281_sizes[2] = {51200, 102400}; 
    
    // Nối tensor dọc theo channel
    node_concat(n281_inputs, 2, n281_output, n281_sizes, 153600);
    
    // Store OFM ra DRAM_2
    store_ofm(dram2, 57446400, n281_output, 153600);   // Địa chỉ: 57446400 -> 57599999
}

// ============================================================================
// MODULE 22: C3k2 -> Detect P5 (Bottleneck + PSABlock)
// Input: 1x384x20x20 (Từ Module 21) | Output: 1x256x20x20 
// ============================================================================
void module_22_c3k2_detect_P5(const float *dram1, float *dram2) {
    
    // --------------------------------------------------------
    // NODE 284, 287, 290: cv1 (Conv -> Sigmoid -> Mul)
    // --------------------------------------------------------
    static float n284_in[153600], n284_w[98304], n284_b[256], n284_out[102400];
    
    load_ifm(dram2, 57446400, n284_in, 153600); // Output của Module 21
    load_weight_bias(dram1, 1845499, n284_w, 98304);
    load_weight_bias(dram1, 1943803, n284_b, 256);
    
    node_conv(n284_in, n284_w, n284_b, n284_out, 1, 384, 20, 20, 256, 1, 1, 0);
    store_ofm(dram2, 57728000, n284_out, 102400);

    static float n287_in[102400], n287_out[102400];
    load_ifm(dram2, 57728000, n287_in, 102400);
    node_sigmoid(n287_in, n287_out, 102400);
    store_ofm(dram2, 57958400, n287_out, 102400);

    static float n290_i1[102400], n290_i2[102400], n290_out[102400];
    load_ifm(dram2, 57728000, n290_i1, 102400);
    load_ifm(dram2, 57958400, n290_i2, 102400);
    node_mul(n290_i1, n290_i2, n290_out, 102400);
    store_ofm(dram2, 58188800, n290_out, 102400);

    // --------------------------------------------------------
    // NODE 293: Split (Tách 256 kênh thành 2 luồng 128 kênh)
    // --------------------------------------------------------
    static float n293_in[102400], n293_split0[51200], n293_split1[51200];
    load_ifm(dram2, 58188800, n293_in, 102400);
    
    float *splits_293[2] = {n293_split0, n293_split1};
    node_split(n293_in, splits_293, 2, 51200);
    
    // Đã sửa: Lưu split0 ra DRAM để Verify khớp với file Golden
    store_ofm(dram2, 58400000, n293_split0, 51200); 

    // ========================================================
    // BLOCK 1: STANDARD BOTTLENECK (m.0)
    // ========================================================
    
    // NODE 296, 298, 300: m.0.cv1 (Conv -> Sigmoid -> Mul)
    static float n296_w[73728], n296_b[64], n296_out[25600];
    
    // Đã sửa: Không đọc DRAM, truyền thẳng n293_split1 vào hàm conv
    load_weight_bias(dram1, 1944772, n296_w, 73728);
    load_weight_bias(dram1, 2018500, n296_b, 64);
    node_conv(n293_split1, n296_w, n296_b, n296_out, 1, 128, 20, 20, 64, 3, 1, 1);
    store_ofm(dram2, 58560000, n296_out, 25600);

    static float n298_in[25600], n298_out[25600];
    load_ifm(dram2, 58560000, n298_in, 25600);
    node_sigmoid(n298_in, n298_out, 25600);
    store_ofm(dram2, 58688000, n298_out, 25600);

    static float n300_i1[25600], n300_i2[25600], n300_out[25600];
    load_ifm(dram2, 58560000, n300_i1, 25600);
    load_ifm(dram2, 58688000, n300_i2, 25600);
    node_mul(n300_i1, n300_i2, n300_out, 25600);
    store_ofm(dram2, 58816000, n300_out, 25600);

    // NODE 302, 304, 306: m.0.cv2 (Conv -> Sigmoid -> Mul)
    static float n302_in[25600], n302_w[73728], n302_b[128], n302_out[51200];
    load_ifm(dram2, 58816000, n302_in, 25600);
    load_weight_bias(dram1, 2022724, n302_w, 73728);
    load_weight_bias(dram1, 2096452, n302_b, 128);
    node_conv(n302_in, n302_w, n302_b, n302_out, 1, 64, 20, 20, 128, 3, 1, 1);
    store_ofm(dram2, 58944000, n302_out, 51200);

    static float n304_in[51200], n304_out[51200];
    load_ifm(dram2, 58944000, n304_in, 51200);
    node_sigmoid(n304_in, n304_out, 51200);
    store_ofm(dram2, 59097600, n304_out, 51200);

    static float n306_i1[51200], n306_i2[51200], n306_out[51200];
    load_ifm(dram2, 58944000, n306_i1, 51200);
    load_ifm(dram2, 59097600, n306_i2, 51200);
    node_mul(n306_i1, n306_i2, n306_out, 51200);
    store_ofm(dram2, 59150400, n306_out, 51200);

    // NODE 308: Add (Shortcut Bottleneck 1)
    static float n308_i2[51200], n308_out[51200];
    load_ifm(dram2, 59150400, n308_i2, 51200); 
    
    // Đã sửa: Truyền thẳng n293_split1 vào hàm Add thay vì đọc DRAM
    node_add(n293_split1, n308_i2, n308_out, 51200);
    store_ofm(dram2, 59203200, n308_out, 51200);
    // ========================================================
    // BLOCK 2: ATTENTION PSA BLOCK (m.1)
    // ========================================================

    // NODE 309, 310: Conv (QKV Gen) -> Reshape
    static float n309_in[51200], n309_w[32768], n309_b[256], n309_out[102400];
    load_ifm(dram2, 59203200, n309_in, 51200);
    load_weight_bias(dram1, 2096648, n309_w, 32768);
    load_weight_bias(dram1, 2129416, n309_b, 256);
    node_conv(n309_in, n309_w, n309_b, n309_out, 1, 128, 20, 20, 256, 1, 1, 0);
    store_ofm(dram2, 59254400, n309_out, 102400);

    static float n310_out[102400];
    node_reshape(n309_out, n310_out, 102400); 
    store_ofm(dram2, 59356800, n310_out, 102400);

    // NODE 311: Split Q, K, V (Manual Loop)
    static float n311_q[25600], n311_k[25600], n311_v[51200];
    int offset_q = 0, offset_k = 0, offset_v = 0;
    for (int h = 0; h < 2; h++) { 
        int head_offset = h * 51200; 
        for (int i = 0; i < 12800; i++) n311_q[offset_q++] = n310_out[head_offset + i];
        for (int i = 0; i < 12800; i++) n311_k[offset_k++] = n310_out[head_offset + 12800 + i];
        for (int i = 0; i < 25600; i++) n311_v[offset_v++] = n310_out[head_offset + 25600 + i];
    }
    store_ofm(dram2, 59459200, n311_q, 25600);

    // NODE 312: Mul (Scale Q)
    static float n312_out[25600];
    float scale = 0.176776695f; 
    node_mul_scalar(n311_q, scale, n312_out, 25600);
    store_ofm(dram2, 59484800, n312_out, 25600);

    // NODE 313: Reshape V (Lưu lại V vào DRAM)
    store_ofm(dram2, 59510400, n311_v, 51200);

    // NODE 314: Transpose Q_scaled (Manual Loop)
    static float n314_out[25600];
    int idx314 = 0;
    for (int h = 0; h < 2; h++) {
        for (int w = 0; w < 400; w++) {
            for (int c = 0; c < 32; c++) {
                n314_out[idx314++] = n312_out[h * 12800 + c * 400 + w];
            }
        }
    }
    store_ofm(dram2, 59561600, n314_out, 25600);

    // NODE 315: Depthwise Conv PE
    static float n315_w[1152], n315_b[128], n315_out[51200];
    load_weight_bias(dram1, 2129684, n315_w, 1152);
    load_weight_bias(dram1, 2130836, n315_b, 128);
    node_conv_depthwise(n311_v, n315_w, n315_b, n315_out, 1, 128, 20, 20, 128, 3, 1, 1);
    store_ofm(dram2, 59587200, n315_out, 51200);

    // NODE 316: MatMul Q_T * K (Manual Loop)
    static float n316_out[320000];
    for (int h = 0; h < 2; h++) {
        int offset_q_t = h * 12800;
        int offset_k   = h * 12800;
        int offset_out = h * 160000;
        for (int i = 0; i < 400; i++) {
            for (int j = 0; j < 400; j++) {
                float sum = 0.0f;
                for (int p = 0; p < 32; p++) {
                    sum += n314_out[offset_q_t + i * 32 + p] * n311_k[offset_k + p * 400 + j];
                }
                n316_out[offset_out + i * 400 + j] = sum;
            }
        }
    }
    store_ofm(dram2, 59638400, n316_out, 320000);

    // NODE 317: Softmax
    static float n317_out[320000];
    node_softmax(n316_out, n317_out, 800, 400); 
    store_ofm(dram2, 59958400, n317_out, 320000);

    // NODE 318: Transpose Attn (Manual Loop)
    static float n318_out[320000];
    int idx318 = 0;
    for (int h = 0; h < 2; h++) {
        for (int w = 0; w < 400; w++) {
            for (int h2 = 0; h2 < 400; h2++) {
                n318_out[idx318++] = n317_out[h * 160000 + h2 * 400 + w];
            }
        }
    }
    store_ofm(dram2, 60278400, n318_out, 320000);

    // NODE 319: MatMul V * Attn_T (Manual Loop)
    static float n319_out[51200];
    for (int h = 0; h < 2; h++) {
        int offset_v = h * 25600; 
        int offset_attn_t = h * 160000; 
        int offset_out = h * 25600; 
        for (int i = 0; i < 64; i++) {
            for (int j = 0; j < 400; j++) {
                float sum = 0.0f;
                for (int p = 0; p < 400; p++) {
                    sum += n311_v[offset_v + i * 400 + p] * n318_out[offset_attn_t + p * 400 + j];
                }
                n319_out[offset_out + i * 400 + j] = sum;
            }
        }
    }
    store_ofm(dram2, 60598400, n319_out, 51200);

    // NODE 320: Reshape
    static float n320_out[51200];
    node_reshape(n319_out, n320_out, 51200);
    store_ofm(dram2, 60649600, n320_out, 51200);

    // NODE 321: Add (PE + Attn_out)
    static float n321_out[51200];
    node_add(n320_out, n315_out, n321_out, 51200);
    store_ofm(dram2, 60700800, n321_out, 51200);

    // NODE 322: Conv (Proj)
    static float n322_w[16384], n322_b[128], n322_out[51200];
    load_weight_bias(dram1, 2130968, n322_w, 16384);
    load_weight_bias(dram1, 2147352, n322_b, 128);
    node_conv(n321_out, n322_w, n322_b, n322_out, 1, 128, 20, 20, 128, 1, 1, 0);
    store_ofm(dram2, 60752000, n322_out, 51200);

    // NODE 323: Add (Shortcut Attention)
    static float n323_out[51200];
    node_add(n308_out, n322_out, n323_out, 51200);
    store_ofm(dram2, 60803200, n323_out, 51200);

    // NODE 324, 325, 326: FFN cv1
    static float n324_w[32768], n324_b[256], n324_out[102400];
    load_weight_bias(dram1, 2147480, n324_w, 32768);
    load_weight_bias(dram1, 2180248, n324_b, 256);
    node_conv(n323_out, n324_w, n324_b, n324_out, 1, 128, 20, 20, 256, 1, 1, 0);
    store_ofm(dram2, 60854400, n324_out, 102400);

    static float n325_out[102400];
    node_sigmoid(n324_out, n325_out, 102400);
    store_ofm(dram2, 60956800, n325_out, 102400);

    static float n326_out[102400];
    node_mul(n324_out, n325_out, n326_out, 102400);
    store_ofm(dram2, 61059200, n326_out, 102400);

    // NODE 327: FFN cv2
    static float n327_w[32768], n327_b[128], n327_out[51200];
    load_weight_bias(dram1, 2180504, n327_w, 32768);
    load_weight_bias(dram1, 2213272, n327_b, 128);
    node_conv(n326_out, n327_w, n327_b, n327_out, 1, 256, 20, 20, 128, 1, 1, 0);
    store_ofm(dram2, 61161600, n327_out, 51200);

    // NODE 328: Add (Shortcut FFN)
    static float n328_out[51200];
    node_add(n323_out, n327_out, n328_out, 51200);
    store_ofm(dram2, 61212800, n328_out, 51200);

    // ========================================================
    // CONCAT & OUTPUT
    // ========================================================

    // NODE 329: Concat (Gộp split0, split1 và đầu ra của khối Attention)
    static float n329_out[153600];
    float *n329_inputs[3] = {n293_split0, n293_split1, n328_out};
    int n329_sizes[3] = {51200, 51200, 51200};
    node_concat(n329_inputs, 3, n329_out, n329_sizes, 153600);
    store_ofm(dram2, 61264000, n329_out, 153600);

    // NODE 330, 331, 332: cv3 (Conv -> Sigmoid -> Mul)
    static float n330_w[98304], n330_b[256], n330_out[102400];
    load_weight_bias(dram1, 2213400, n330_w, 98304);
    load_weight_bias(dram1, 2311704, n330_b, 256);
    node_conv(n329_out, n330_w, n330_b, n330_out, 1, 384, 20, 20, 256, 1, 1, 0);
    store_ofm(dram2, 61417600, n330_out, 102400);

    static float n331_out[102400];
    node_sigmoid(n330_out, n331_out, 102400);
    store_ofm(dram2, 61520000, n331_out, 102400);

    static float n332_out[102400];
    node_mul(n330_out, n331_out, n332_out, 102400);
    store_ofm(dram2, 61622400, n332_out, 102400); 
}

// ============================================================================
// MODULE 23: DETECT HEAD (YOLOv8 / YOLO26n)
// Xử lý P3 (80x80), P4 (40x40), P5 (20x20) -> Decode Box -> Dense Array 8400
// ============================================================================
void module_23_detect_head(const float *dram1, float *dram2) {
    
    // ========================================================================
    // 1. NHÁNH P3 (80x80) - Input từ Node 213 (Offset: 46912000)
    // ========================================================================
    
    // --- P3 BOX BRANCH ---
    static float w215[9216], b215[16]; load_weight_bias(dram1, 1551598, w215, 9216); load_weight_bias(dram1, 1560814, b215, 16);
    node_conv(&dram2[46912000], w215, b215, &dram2[47424000], 1, 64, 80, 80, 16, 3, 1, 1);
    node_sigmoid(&dram2[47424000], &dram2[48038400], 102400);
    node_mul(&dram2[47424000], &dram2[48038400], &dram2[48652800], 102400);

    static float w224[2304], b224[16]; load_weight_bias(dram1, 1561470, w224, 2304); load_weight_bias(dram1, 1563774, b224, 16);
    node_conv(&dram2[48652800], w224, b224, &dram2[49472000], 1, 16, 80, 80, 16, 3, 1, 1);
    node_sigmoid(&dram2[49472000], &dram2[50188800], 102400);
    node_mul(&dram2[49472000], &dram2[50188800], &dram2[50905600], 102400);

    static float w233[64], b233[4]; load_weight_bias(dram1, 1592654, w233, 64); load_weight_bias(dram1, 1592718, b233, 4);
    node_conv(&dram2[50905600], w233, b233, &dram2[51622400], 1, 16, 80, 80, 4, 1, 1, 0);
    node_reshape(&dram2[51622400], &dram2[52160000], 25600); // N236: 1x4x6400

    // --- P3 CLS BRANCH ---
    static float w216[576], b216[64]; load_weight_bias(dram1, 1560830, w216, 576); load_weight_bias(dram1, 1561406, b216, 64);
    node_conv_depthwise(&dram2[46912000], w216, b216, &dram2[47526400], 1, 64, 80, 80, 64, 3, 1, 1);
    node_sigmoid(&dram2[47526400], &dram2[48140800], 409600);
    node_mul(&dram2[47526400], &dram2[48140800], &dram2[48755200], 409600);

    static float w225[4096], b225[64]; load_weight_bias(dram1, 1563790, w225, 4096); load_weight_bias(dram1, 1567886, b225, 64);
    node_conv(&dram2[48755200], w225, b225, &dram2[49574400], 1, 64, 80, 80, 64, 1, 1, 0);
    node_sigmoid(&dram2[49574400], &dram2[50291200], 409600);
    node_mul(&dram2[49574400], &dram2[50291200], &dram2[51008000], 409600);

    static float w234[576], b234[64]; load_weight_bias(dram1, 1592722, w234, 576); load_weight_bias(dram1, 1593298, b234, 64);
    node_conv_depthwise(&dram2[51008000], w234, b234, &dram2[51648000], 1, 64, 80, 80, 64, 3, 1, 1);
    node_sigmoid(&dram2[51648000], &dram2[52185600], 409600);
    node_mul(&dram2[51648000], &dram2[52185600], &dram2[52697600], 409600);

    static float w243[4096], b243[64]; load_weight_bias(dram1, 1597527, w243, 4096); load_weight_bias(dram1, 1601623, b243, 64);
    node_conv(&dram2[52697600], w243, b243, &dram2[53209600], 1, 64, 80, 80, 64, 1, 1, 0);
    node_sigmoid(&dram2[53209600], &dram2[53721600], 409600);
    node_mul(&dram2[53209600], &dram2[53721600], &dram2[54182400], 409600);

    static float w250[64], b250[1]; load_weight_bias(dram1, 1610935, w250, 64); load_weight_bias(dram1, 1610999, b250, 1);
    node_conv(&dram2[54182400], w250, b250, &dram2[54643200], 1, 64, 80, 80, 1, 1, 1, 0);
    node_reshape(&dram2[54643200], &dram2[54700800], 6400); // N252: 1x1x6400

    // ========================================================================
    // 2. NHÁNH P4 (40x40) - Input từ Node 271 (Offset: 56396800)
    // ========================================================================

    // --- P4 BOX BRANCH ---
    static float w273[18432], b273[16]; load_weight_bias(dram1, 1815195, w273, 18432); load_weight_bias(dram1, 1833627, b273, 16);
    node_conv(&dram2[56396800], w273, b273, &dram2[56652800], 1, 128, 40, 40, 16, 3, 1, 1);
    node_sigmoid(&dram2[56652800], &dram2[56934400], 25600);
    node_mul(&dram2[56652800], &dram2[56934400], &dram2[57216000], 25600);

    static float w282[2304], b282[16]; load_weight_bias(dram1, 1834923, w282, 2304); load_weight_bias(dram1, 1837227, b282, 16);
    node_conv(&dram2[57216000], w282, b282, &dram2[57600000], 1, 16, 40, 40, 16, 3, 1, 1);
    node_sigmoid(&dram2[57600000], &dram2[57830400], 25600);
    node_mul(&dram2[57600000], &dram2[57830400], &dram2[58060800], 25600);

    static float w291[64], b291[4]; load_weight_bias(dram1, 1944059, w291, 64); load_weight_bias(dram1, 1944123, b291, 4);
    node_conv(&dram2[58060800], w291, b291, &dram2[58291200], 1, 16, 40, 40, 4, 1, 1, 0);
    node_reshape(&dram2[58291200], &dram2[58451200], 6400); // N294: 1x4x1600

    // --- P4 CLS BRANCH ---
    static float w274[1152], b274[128]; load_weight_bias(dram1, 1833643, w274, 1152); load_weight_bias(dram1, 1834795, b274, 128);
    node_conv_depthwise(&dram2[56396800], w274, b274, &dram2[56678400], 1, 128, 40, 40, 128, 3, 1, 1);
    node_sigmoid(&dram2[56678400], &dram2[56960000], 204800);
    node_mul(&dram2[56678400], &dram2[56960000], &dram2[57241600], 204800);

    static float w283[8192], b283[64]; load_weight_bias(dram1, 1837243, w283, 8192); load_weight_bias(dram1, 1845435, b283, 64);
    node_conv(&dram2[57241600], w283, b283, &dram2[57625600], 1, 128, 40, 40, 64, 1, 1, 0);
    node_sigmoid(&dram2[57625600], &dram2[57856000], 102400);
    node_mul(&dram2[57625600], &dram2[57856000], &dram2[58086400], 102400);

    static float w292[576], b292[64]; load_weight_bias(dram1, 1944127, w292, 576); load_weight_bias(dram1, 1944703, b292, 64);
    node_conv_depthwise(&dram2[58086400], w292, b292, &dram2[58297600], 1, 64, 40, 40, 64, 3, 1, 1);
    node_sigmoid(&dram2[58297600], &dram2[58457600], 102400);
    node_mul(&dram2[58297600], &dram2[58457600], &dram2[58585600], 102400);

    static float w299[4096], b299[64]; load_weight_bias(dram1, 2018564, w299, 4096); load_weight_bias(dram1, 2022660, b299, 64);
    node_conv(&dram2[58585600], w299, b299, &dram2[58713600], 1, 64, 40, 40, 64, 1, 1, 0);
    node_sigmoid(&dram2[58713600], &dram2[58841600], 102400);
    node_mul(&dram2[58713600], &dram2[58841600], &dram2[58995200], 102400);

    static float w305[64], b305[1]; load_weight_bias(dram1, 2096580, w305, 64); load_weight_bias(dram1, 2096644, b305, 1);
    node_conv(&dram2[58995200], w305, b305, &dram2[59148800], 1, 64, 40, 40, 1, 1, 1, 0);
    node_reshape(&dram2[59148800], &dram2[59201600], 1600); // N307: 1x1x1600

    // ========================================================================
    // 3. NHÁNH P5 (20x20) - Input từ Node 332 (Offset: 61622400)
    // ========================================================================

    // --- P5 BOX BRANCH ---
    static float w333[36864], b333[16]; load_weight_bias(dram1, 2311960, w333, 36864); load_weight_bias(dram1, 2348824, b333, 16);
    node_conv(&dram2[61622400], w333, b333, &dram2[61724800], 1, 256, 20, 20, 16, 3, 1, 1);
    node_sigmoid(&dram2[61724800], &dram2[61833600], 6400);
    node_mul(&dram2[61724800], &dram2[61833600], &dram2[61942400], 6400);

    static float w339[2304], b339[16]; load_weight_bias(dram1, 2351400, w339, 2304); load_weight_bias(dram1, 2353704, b339, 16);
    node_conv(&dram2[61942400], w339, b339, &dram2[62051200], 1, 16, 20, 20, 16, 3, 1, 1);
    node_sigmoid(&dram2[62051200], &dram2[62083200], 6400);
    node_mul(&dram2[62051200], &dram2[62083200], &dram2[62115200], 6400);

    static float w345[64], b345[4]; load_weight_bias(dram1, 2370168, w345, 64); load_weight_bias(dram1, 2370232, b345, 4);
    node_conv(&dram2[62115200], w345, b345, &dram2[62147200], 1, 16, 20, 20, 4, 1, 1, 0);
    node_reshape(&dram2[62147200], &dram2[62174400], 1600); // N347: 1x4x400

    // --- P5 CLS BRANCH ---
    static float w334[2304], b334[256]; load_weight_bias(dram1, 2348840, w334, 2304); load_weight_bias(dram1, 2351144, b334, 256);
    node_conv_depthwise(&dram2[61622400], w334, b334, &dram2[61731200], 1, 256, 20, 20, 256, 3, 1, 1);
    node_sigmoid(&dram2[61731200], &dram2[61840000], 102400);
    node_mul(&dram2[61731200], &dram2[61840000], &dram2[61948800], 102400);

    static float w340[16384], b340[64]; load_weight_bias(dram1, 2353720, w340, 16384); load_weight_bias(dram1, 2370104, b340, 64);
    node_conv(&dram2[61948800], w340, b340, &dram2[62057600], 1, 256, 20, 20, 64, 1, 1, 0);
    node_sigmoid(&dram2[62057600], &dram2[62089600], 25600);
    node_mul(&dram2[62057600], &dram2[62089600], &dram2[62121600], 25600);

    static float w346[576], b346[64]; load_weight_bias(dram1, 2370236, w346, 576); load_weight_bias(dram1, 2370812, b346, 64);
    node_conv_depthwise(&dram2[62121600], w346, b346, &dram2[62148800], 1, 64, 20, 20, 64, 3, 1, 1);
    node_sigmoid(&dram2[62148800], &dram2[62176000], 25600);
    node_mul(&dram2[62148800], &dram2[62176000], &dram2[62235200], 25600);

    static float w351[4096], b351[64]; load_weight_bias(dram1, 2370879, w351, 4096); load_weight_bias(dram1, 2374975, b351, 64);
    node_conv(&dram2[62235200], w351, b351, &dram2[62260800], 1, 64, 20, 20, 64, 1, 1, 0);
    node_sigmoid(&dram2[62260800], &dram2[62320000], 25600);
    node_mul(&dram2[62260800], &dram2[62320000], &dram2[62379200], 25600);

    static float w359[64], b359[1]; load_weight_bias(dram1, 2408645, w359, 64); load_weight_bias(dram1, 2408709, b359, 1);
    node_conv(&dram2[62379200], w359, b359, &dram2[62438400], 1, 64, 20, 20, 1, 1, 1, 0);
    node_reshape(&dram2[62438400], &dram2[62472400], 400); // N361: 1x1x400

    // ========================================================================
    // 4. POST-PROCESSING (DFL BOX DECODING & CONCAT SCORE)
    // ========================================================================

    // ========================================================================
    // 4. POST-PROCESSING (DFL BOX DECODING & CONCAT SCORE)
    // ========================================================================

    // N349: Concat Boxes (N236, N294, N347) -> Output: 1x4x8400
    // SỬA LỖI: Gộp theo chiều không gian (axis=2) đan xen giữa P3, P4, P5
    int idx349 = 0;
    for (int c = 0; c < 4; c++) {
        // Copy Kênh c của P3 (6400 phần tử)
        for (int i = 0; i < 6400; i++) {
            dram2[62201600 + idx349++] = dram2[52160000 + c * 6400 + i];
        }
        // Copy Kênh c của P4 (1600 phần tử)
        for (int i = 0; i < 1600; i++) {
            dram2[62201600 + idx349++] = dram2[58451200 + c * 1600 + i];
        }
        // Copy Kênh c của P5 (400 phần tử)
        for (int i = 0; i < 400; i++) {
            dram2[62201600 + idx349++] = dram2[62174400 + c * 400 + i];
        }
    }

    // N352, N353: Slice (Tách mảng 1x4x8400 thành 2 mảng 1x2x8400 cho dFL)
    for(int i = 0; i < 16800; i++) {
        dram2[62286400 + i] = dram2[62201600 + i];          // N352: ch 0,1 (Top-Left)
        dram2[62303200 + i] = dram2[62201600 + 16800 + i];  // N353: ch 2,3 (Bottom-Right)
    }

    // N355: Sub (Anchors_cxcy - TopLeft) -> Tính x1, y1
    static float w355[16800]; load_weight_bias(dram1, 2375045, w355, 16800);
    for(int i = 0; i < 16800; i++) dram2[62345600 + i] = w355[i] - dram2[62286400 + i];

    // N356: Add (Anchors_cxcy + BottomRight) -> Tính x2, y2
    static float w356[16800]; load_weight_bias(dram1, 2391845, w356, 16800);
    for(int i = 0; i < 16800; i++) dram2[62362400 + i] = w356[i] + dram2[62303200 + i];

    // N358: Concat (Sub, Add) -> Ghép x1, y1, x2, y2 lại thành 1x4x8400
    for(int i = 0; i < 16800; i++) {
        dram2[62404800 + i] = dram2[62345600 + i];
        dram2[62404800 + 16800 + i] = dram2[62362400 + i];
    }

    // N360: Mul (Box * Strides) -> Đưa toạ độ về kích thước ảnh gốc 640x640
    static float w360[8400]; load_weight_bias(dram1, 2408710, w360, 8400);
    for(int c = 0; c < 4; c++) {
        for(int i = 0; i < 8400; i++) {
            dram2[62438800 + c*8400 + i] = dram2[62404800 + c*8400 + i] * w360[i];
        }
    }

    // N362: Concat Cls (N252, N307, N361) -> Output: 1x1x8400 (8400)
    float *n362_inputs[3] = {&dram2[54700800], &dram2[59201600], &dram2[62472400]};
    int n362_sizes[3] = {6400, 1600, 400};
    node_concat(n362_inputs, 3, &dram2[62472800], n362_sizes, 8400);

    // N363: Sigmoid (Tính xác suất cho 8400 box)
    node_sigmoid(&dram2[62472800], &dram2[62481200], 8400);

    // N364: Concat Box (N360) và Score (N363) -> 1x5x8400
    for(int i = 0; i < 33600; i++) dram2[62489600 + i] = dram2[62438800 + i];
    for(int i = 0; i < 8400; i++) dram2[62489600 + 33600 + i] = dram2[62481200 + i];

    // N365: Transpose 1x5x8400 -> 1x8400x5 (Để chuẩn bị NMS)
    for(int c = 0; c < 5; c++) {
        for(int i = 0; i < 8400; i++) dram2[62531600 + i*5 + c] = dram2[62489600 + c*8400 + i];
    }

    // N366: Split (Tách mảng Transpose thành Boxes và Scores)
    for(int i = 0; i < 8400; i++) {
        for(int c = 0; c < 4; c++) dram2[62573600 + i*4 + c] = dram2[62531600 + i*5 + c];
    }

    // N367: ReduceMax (Lấy Score cao nhất của mỗi Box - Ở đây chỉ có 1 class nên lấy chính nó)
    for(int i = 0; i < 8400; i++) dram2[62607200 + i] = dram2[62531600 + i*5 + 4];
}