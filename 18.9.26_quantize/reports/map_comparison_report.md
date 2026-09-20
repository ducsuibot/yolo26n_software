# Báo cáo đo lường mAP: YOLOv26n FP32 vs Quantized INT8

- **Tập dữ liệu**: DUT_Anti_UAV_YOLO (val, 10 ảnh)
- **Phương pháp tiền xử lý**: Direct Resize 640x640 (Hiệu năng thô cho FPGA)
- **Mục tiêu**: Khảo sát triển khai phần cứng FPGA ZCU104 ứng dụng UAV Tracking

### Bảng kết quả so sánh định lượng

| Chỉ số đánh giá | Baseline (FP32) | Quantized (INT8) | Chênh lệch (Delta) |
| :--- | :---: | :---: | :---: |
| **Dung lượng mô hình** | 9.35 MB | 2.63 MB | **-71.8%** |
| **Precision (P)** | 100.00% | 100.00% | +0.00% |
| **Recall (R)** | 92.97% | 91.81% | -1.17% |
| **F1-Score** | 0.9636 | 0.9573 | -0.0063 |
| **mAP@0.5 (mAP50)** | 97.25% | 93.50% | **-3.75%** |
| **mAP@0.5:0.95 (mAP50-95)** | 70.46% | 64.37% | **-6.08%** |
| **Inference Latency (CPU)** | 26.09 ms | 90.55 ms | +64.46 ms |
| **Throughput (FPS)** | 38.3 FPS | 11.0 FPS | -27.3 FPS |

### Nhận xét & Đánh giá triển khai FPGA ZCU104
1. **Bảo toàn độ chính xác phát hiện**: mAP@0.5 của mô hình INT8 đạt **93.50%**, độ suy giảm chỉ **3.75%** so với FP32.
2. **Độ chính xác vị trí bounding box**: mAP@0.5:0.95 đạt **64.37%** (chênh lệch -6.08%).
3. **Tối ưu bộ nhớ on-chip**: Giảm kích thước trọng số **71.8%** (từ 9.35 MB xuống 2.63 MB), cực kỳ thuận lợi để nạp và tính toán trên BRAM/URAM và DRAM1 của FPGA ZCU104.
