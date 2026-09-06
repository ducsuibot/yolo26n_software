#include "node.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

// --------------------------------------------------------
// YOLO26n - Node Operations Implementation (Baseline)
// --------------------------------------------------------

// 1. Convolution 2D (Direct implementation)
void node_conv(float *input, float *weight, float *bias, float *output, 
               int batch, int in_c, int in_h, int in_w, 
               int out_c, int k_size, int stride, int pad) {
    int out_h = (in_h - k_size + 2 * pad) / stride + 1;
    int out_w = (in_w - k_size + 2 * pad) / stride + 1;

    for (int b = 0; b < batch; b++) {
        for (int oc = 0; oc < out_c; oc++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    float sum = bias ? bias[oc] : 0.0f;
                    
                    for (int ic = 0; ic < in_c; ic++) {
                        for (int kh = 0; kh < k_size; kh++) {
                            for (int kw = 0; kw < k_size; kw++) {
                                int ih = oh * stride - pad + kh;
                                int iw = ow * stride - pad + kw;
                                
                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    int in_idx = ((b * in_c + ic) * in_h + ih) * in_w + iw;
                                    int w_idx = ((oc * in_c + ic) * k_size + kh) * k_size + kw;
                                    sum += input[in_idx] * weight[w_idx];
                                }
                            }
                        }
                    }
                    int out_idx = ((b * out_c + oc) * out_h + oh) * out_w + ow;
                    output[out_idx] = sum;
                }
            }
        }
    }
}

// 2. Kích hoạt Sigmoid
void node_sigmoid(float *input, float *output, int total_elements) {
    for (int i = 0; i < total_elements; i++) {
        output[i] = 1.0f / (1.0f + expf(-input[i]));
    }
}

// 3. Nhân Element-wise
void node_mul(float *input1, float *input2, float *output, int total_elements) {
    for (int i = 0; i < total_elements; i++) {
        output[i] = input1[i] * input2[i];
    }
}

// 4. Chia Tensor (Split) - Cắt mảng liên tục
void node_split(float *input, float **outputs, int num_splits, int split_size) {
    for (int i = 0; i < num_splits; i++) {
        if (input != outputs[i]) {
            memcpy(outputs[i], input + i * split_size, split_size * sizeof(float));
        }
    }
}

// 5. Cộng Element-wise (Skip connections)
void node_add(float *input1, float *input2, float *output, int total_elements) {
    for (int i = 0; i < total_elements; i++) {
        output[i] = input1[i] + input2[i];
    }
}

// 6. Nối Tensor (Concat) - Theo chiều phẳng
void node_concat(float **inputs, int num_inputs, float *output, 
                 int *elements_per_input, int total_elements) {
    int offset = 0;
    for (int i = 0; i < num_inputs; i++) {
        memcpy(output + offset, inputs[i], elements_per_input[i] * sizeof(float));
        offset += elements_per_input[i];
    }
}

// 7. Max Pooling 2D
void node_maxpool(float *input, float *output, 
                  int c, int in_h, int in_w, int k_size, int stride, int pad) {
    int out_h = (in_h - k_size + 2 * pad) / stride + 1;
    int out_w = (in_w - k_size + 2 * pad) / stride + 1;

    for (int ch = 0; ch < c; ch++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float max_val = -INFINITY;
                for (int kh = 0; kh < k_size; kh++) {
                    for (int kw = 0; kw < k_size; kw++) {
                        int ih = oh * stride - pad + kh;
                        int iw = ow * stride - pad + kw;
                        
                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            int in_idx = (ch * in_h + ih) * in_w + iw;
                            if (input[in_idx] > max_val) {
                                max_val = input[in_idx];
                            }
                        }
                    }
                }
                int out_idx = (ch * out_h + oh) * out_w + ow;
                output[out_idx] = max_val;
            }
        }
    }
}

// 8. Reshape
void node_reshape(float *input, float *output, int total_elements) {
    if (input != output) {
        memcpy(output, input, total_elements * sizeof(float));
    }
}

// 9. Transpose (Tổng quát cho 4D - Phổ biến trong YOLO)
void node_transpose(float *input, float *output, 
                    int *in_shape, int *axes, int ndim) {
    // Chỉ triển khai mẫu cho 4D (Batch, Channel, Height, Width)
    if (ndim == 4) {
        int out_shape[4];
        for (int i = 0; i < 4; i++) out_shape[i] = in_shape[axes[i]];
        
        for (int i0 = 0; i0 < out_shape[0]; i0++) {
            for (int i1 = 0; i1 < out_shape[1]; i1++) {
                for (int i2 = 0; i2 < out_shape[2]; i2++) {
                    for (int i3 = 0; i3 < out_shape[3]; i3++) {
                        int out_idx = ((i0 * out_shape[1] + i1) * out_shape[2] + i2) * out_shape[3] + i3;
                        
                        // Map lại index cũ
                        int in_idx_arr[4];
                        int current_out[4] = {i0, i1, i2, i3};
                        for (int k = 0; k < 4; k++) in_idx_arr[axes[k]] = current_out[k];
                        
                        int in_idx = ((in_idx_arr[0] * in_shape[1] + in_idx_arr[1]) * in_shape[2] + in_idx_arr[2]) * in_shape[3] + in_idx_arr[3];
                        output[out_idx] = input[in_idx];
                    }
                }
            }
        }
    }
}

// 10. Nhân ma trận (O = I1 * I2)
void node_matmul(float *input1, float *input2, float *output, 
                 int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += input1[i * k + p] * input2[p * n + j];
            }
            output[i * n + j] = sum;
        }
    }
}

// 11. Softmax (Tính theo từng hàng)
void node_softmax(float *input, float *output, int num_rows, int cols_per_row) {
    for (int r = 0; r < num_rows; r++) {
        float max_val = -INFINITY;
        int row_offset = r * cols_per_row;
        
        for (int c = 0; c < cols_per_row; c++) {
            if (input[row_offset + c] > max_val) max_val = input[row_offset + c];
        }
        
        float sum = 0.0f;
        for (int c = 0; c < cols_per_row; c++) {
            output[row_offset + c] = expf(input[row_offset + c] - max_val);
            sum += output[row_offset + c];
        }
        
        for (int c = 0; c < cols_per_row; c++) {
            output[row_offset + c] /= sum;
        }
    }
}

// 12. Resize / Upsample (Nearest neighbor)
void node_resize(float *input, float *output, 
                 int c, int in_h, int in_w, int out_h, int out_w) {
    float h_scale = (float)in_h / out_h;
    float w_scale = (float)in_w / out_w;

    for (int ch = 0; ch < c; ch++) {
        for (int oh = 0; oh < out_h; oh++) {
            int ih = (int)(oh * h_scale);
            for (int ow = 0; ow < out_w; ow++) {
                int iw = (int)(ow * w_scale);
                int in_idx = (ch * in_h + ih) * in_w + iw;
                int out_idx = (ch * out_h + oh) * out_w + ow;
                output[out_idx] = input[in_idx];
            }
        }
    }
}

// 13. Cắt Tensor (Slice phẳng)
void node_slice(float *input, float *output, 
                int total_elements, int start_idx, int end_idx) {
    int length = end_idx - start_idx;
    if (length > 0) {
        memcpy(output, input + start_idx, length * sizeof(float));
    }
}

// 14. Trừ Element-wise
void node_sub(float *input1, float *input2, float *output, int total_elements) {
    for (int i = 0; i < total_elements; i++) {
        output[i] = input1[i] - input2[i];
    }
}

// 15. Tìm Max dọc theo trục (ReduceMax phẳng)
void node_reducemax(float *input, float *output, int elements_per_reduce) {
    float max_val = -INFINITY;
    for (int i = 0; i < elements_per_reduce; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    output[0] = max_val;
}

// Cấu trúc hỗ trợ TopK
typedef struct {
    float value;
    int index;
} KVPair;

int compare_desc(const void *a, const void *b) {
    float diff = ((KVPair *)b)->value - ((KVPair *)a)->value;
    return (diff > 0) ? 1 : ((diff < 0) ? -1 : 0);
}

// 16. Lấy top K
void node_topk(float *input, float *out_values, int *out_indices, 
               int total_elements, int k) {
    KVPair *pairs = (KVPair *)malloc(total_elements * sizeof(KVPair));
    for (int i = 0; i < total_elements; i++) {
        pairs[i].value = input[i];
        pairs[i].index = i;
    }
    
    qsort(pairs, total_elements, sizeof(KVPair), compare_desc);
    
    for (int i = 0; i < k; i++) {
        out_values[i] = pairs[i].value;
        out_indices[i] = pairs[i].index;
    }
    free(pairs);
}

// 17. Thêm chiều (Unsqueeze)
void node_unsqueeze(float *input, float *output, int total_elements) {
    if (input != output) {
        memcpy(output, input, total_elements * sizeof(float));
    }
}

// 18. Lấy phần tử theo index
void node_gatherelements(float *input, int *indices, float *output, int total_elements) {
    for (int i = 0; i < total_elements; i++) {
        output[i] = input[indices[i]];
    }
}

// 19. Duỗi mảng (Flatten)
void node_flatten(float *input, float *output, int total_elements) {
    if (input != output) {
        memcpy(output, input, total_elements * sizeof(float));
    }
}

// 20. Chia lấy dư (Modulo)
void node_mod(float *input, float mod_val, float *output, int total_elements) {
    for (int i = 0; i < total_elements; i++) {
        output[i] = fmodf(input[i], mod_val);
    }
}

// 21. Ép kiểu (Cast - Float sang Float do kiến trúc chung)
void node_cast(float *input, float *output, int total_elements) {
    if (input != output) {
        memcpy(output, input, total_elements * sizeof(float));
    }
}

// 22. Mở rộng Tensor (Expand - Lặp lại mảng nhỏ)
void node_expand(float *input, float *output, int in_elements, int out_elements) {
    int repeats = out_elements / in_elements;
    for (int i = 0; i < repeats; i++) {
        memcpy(output + i * in_elements, input, in_elements * sizeof(float));
    }
}

// thêm : CHỈ DÙNG KHI CÓ ATTENTION

// 23. Tách mảng không đều (Multi-Split)
void node_split_multi(float *input, float **outputs, int num_splits, int *split_sizes) {
    int offset = 0;
    for (int i = 0; i < num_splits; i++) {
        memcpy(outputs[i], input + offset, split_sizes[i] * sizeof(float));
        offset += split_sizes[i];
    }
}

// 24. Nhân vô hướng (Scalar Mul)
void node_mul_scalar(float *input, float scalar, float *output, int total_elements) {
    for (int i = 0; i < total_elements; i++) {
        output[i] = input[i] * scalar;
    }
}

// 25. Tích chập theo chiều sâu (Depthwise Conv)
void node_conv_depthwise(float *input, float *weight, float *bias, float *output, 
                         int batch, int in_c, int in_h, int in_w, int out_c, int k_size, int stride, int pad) {
    int out_h = (in_h - k_size + 2 * pad) / stride + 1;
    int out_w = (in_w - k_size + 2 * pad) / stride + 1;
    
    // Đối với Depthwise, in_c == out_c == groups
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < in_c; c++) { 
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    float sum = bias ? bias[c] : 0.0f;
                    for (int kh = 0; kh < k_size; kh++) {
                        for (int kw = 0; kw < k_size; kw++) {
                            int ih = oh * stride - pad + kh;
                            int iw = ow * stride - pad + kw;
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                int in_idx = ((b * in_c + c) * in_h + ih) * in_w + iw;
                                int w_idx = (c * k_size + kh) * k_size + kw;
                                sum += input[in_idx] * weight[w_idx];
                            }
                        }
                    }
                    int out_idx = ((b * in_c + c) * out_h + oh) * out_w + ow;
                    output[out_idx] = sum;
                }
            }
        }
    }
}

// 26. Batched MatMul (Dành riêng cho Attention: Q*K và V*Attn)
void node_matmul_batched(float *input1, float *input2, float *output, 
                         int batch, int heads, int m, int k, int n) {
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < heads; h++) {
            int offset1 = (b * heads + h) * m * k;
            int offset2 = (b * heads + h) * k * n;
            int offset_out = (b * heads + h) * m * n;
            
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    float sum = 0.0f;
                    for (int p = 0; p < k; p++) {
                        sum += input1[offset1 + i * k + p] * input2[offset2 + p * n + j];
                    }
                    output[offset_out + i * n + j] = sum;
                }
            }
        }
    }
}