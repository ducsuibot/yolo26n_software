#ifndef NODE_H
#define NODE_H

#include <stdint.h>
#include <stddef.h>

// --------------------------------------------------------
// YOLO26n - Node Operations Header
// --------------------------------------------------------

// 1. Convolution (kết hợp Weight và Bias)
void node_conv(float *input, float *weight, float *bias, float *output, 
               int batch, int in_c, int in_h, int in_w, 
               int out_c, int k_size, int stride, int pad);

// 2. Kích hoạt Sigmoid
void node_sigmoid(float *input, float *output, int total_elements);

// 3. Nhân Element-wise (thường dùng trong hàm kích hoạt Swish/SiLU)
void node_mul(float *input1, float *input2, float *output, int total_elements);

// 4. Chia Tensor (Split)
void node_split(float *input, float **outputs, int num_splits, int split_size);

// 5. Cộng Element-wise (Skip connections)
void node_add(float *input1, float *input2, float *output, int total_elements);

// 6. Nối Tensor (Concat) - Nhận mảng các con trỏ input
void node_concat(float **inputs, int num_inputs, float *output, 
                 int *elements_per_input, int total_elements);

// 7. Max Pooling
void node_maxpool(float *input, float *output, 
                  int c, int in_h, int in_w, int k_size, int stride, int pad);

// 8. Reshape (Thường chỉ là copy/đổi góc nhìn pointer)
void node_reshape(float *input, float *output, int total_elements);

// 9. Transpose (Hoán vị các chiều)
void node_transpose(float *input, float *output, 
                    int *in_shape, int *axes, int ndim);

// 10. Nhân ma trận (Self-Attention / QKV)
void node_matmul(float *input1, float *input2, float *output, 
                 int m, int k, int n);

// 11. Softmax (dùng trong Attention)
void node_softmax(float *input, float *output, int num_rows, int cols_per_row);

// 12. Resize / Upsample (Nearest neighbor hoặc Bilinear)
void node_resize(float *input, float *output, 
                 int c, int in_h, int in_w, int out_h, int out_w);

// 13. Cắt Tensor (Slice - dùng trong Detect Head)
void node_slice(float *input, float *output, 
                int total_elements, int start_idx, int end_idx);

// 14. Trừ Element-wise
void node_sub(float *input1, float *input2, float *output, int total_elements);

// 15. Tìm Max theo trục (ReduceMax)
void node_reducemax(float *input, float *output, int elements_per_reduce);

// 16. Lấy top K phần tử lớn nhất
void node_topk(float *input, float *out_values, int *out_indices, 
               int total_elements, int k);

// 17. Thêm chiều (Unsqueeze)
void node_unsqueeze(float *input, float *output, int total_elements);

// 18. Lấy phần tử theo index (GatherElements)
void node_gatherelements(float *input, int *indices, float *output, int total_elements);

// 19. Duỗi mảng (Flatten)
void node_flatten(float *input, float *output, int total_elements);

// 20. Chia lấy dư (Modulo)
void node_mod(float *input, float mod_val, float *output, int total_elements);

// 21. Ép kiểu (Cast)
void node_cast(float *input, float *output, int total_elements);

// 22. Mở rộng kích thước Tensor (Expand)
void node_expand(float *input, float *output, int in_elements, int out_elements);

// thêm : CHỈ DÙNG KHI CÓ ATTENTION

// 23. Tách mảng không đều (Multi-Split)
void node_split_multi(float *input, float **outputs, int num_splits, int *split_sizes);

// 24. Nhân vô hướng (Scalar Mul)
void node_mul_scalar(float *input, float scalar, float *output, int total_elements);

// 25. Tích chập theo chiều sâu (Depthwise Conv)
void node_conv_depthwise(float *input, float *weight, float *bias, float *output, 
                         int batch, int in_c, int in_h, int in_w, int out_c, int k_size, int stride, int pad);

// 26. Batched MatMul (Dành riêng cho Attention: Q*K và V*Attn)
void node_matmul_batched(float *input1, float *input2, float *output, 
                         int batch, int heads, int m, int k, int n);

#endif // NODE_H