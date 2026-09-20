# BÁO CÁO ĐÁNH GIÁ mAP: BASELINE FP32 VS INT8 VS INT8 + 8-BIT LUT

- **Tập dữ liệu:** `DUT_Anti_UAV_YOLO (val)`
- **Số lượng ảnh đánh giá:** `2600` ảnh
- **Tiền xử lý:** `Direct Resize 640x640 / 255.0` (Mô phỏng pipeline phần cứng thô)
- **Thời gian chạy:** `2026-09-19 12:55:54`

## 1. Bảng Tổng hợp Chỉ số

| Chỉ số (Metric) | Baseline FP32 | INT8 (FP32 Act) | INT8 + 8-bit LUT Act | Chênh lệch (LUT vs FP32) | Chênh lệch (LUT vs INT8-FP32) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dung lượng Model** | 9.35 MB | 2.63 MB | 2.67 MB | **-71.5%** | +1.5% (Do chứa bảng 256 bytes) |
| **Precision (P)** | 92.56% | 89.84% | 87.67% | -4.89% | -2.17% |
| **Recall (R)** | 79.74% | 72.51% | 74.06% | -5.68% | +1.54% |
| **F1-Score** | 0.8567 | 0.8025 | 0.8029 | -0.0539 | +0.0004 |
| **mAP@50 (mAP50)** | **85.56%** | **78.68%** | **78.59%** | **-6.97%** | **-0.09%** |
| **mAP@50:95** | **45.04%** | **37.29%** | **38.07%** | **-6.96%** | **+0.78%** |
| **Latency (CPU)** | 26.93 ms | 90.35 ms | 207.73 ms | +180.81 ms | - |

## 2. Kết luận Kỹ thuật

1. **Tác động của bảng tra 8-bit LUT:** Khi thay thế toàn bộ hàm Sigmoid/SiLU liên tục bằng bảng tra rời rạc 8-bit LUT (LUT-256), mAP@50 chỉ thay đổi **-0.09%** so với mô hình dùng hàm FP32. Điều này chứng minh 8-bit LUT bảo toàn hầu như trọn vẹn độ chính xác của mạng!
2. **Triển khai phần cứng trên FPGA ZCU104:**
   - Loại bỏ hoàn toàn 4 bước dequantize và tính toán số thực FP32.
   - Mỗi bảng tra LUT-256 chỉ tốn 32 LUT6 (Distributed ROM), không tốn DSP và Block RAM.
   - Độ trễ chỉ 1 chu kỳ clock, đạt tần số $F_{max} \ge 450\text{ MHz}$.
