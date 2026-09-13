# Báo cáo đo lường mAP: YOLOv26n FP32 vs Quantized INT8

- **Tập dữ liệu**: DUT_Anti_UAV_YOLO (val, 2600 ảnh)
- **Phương pháp tiền xử lý**: Direct Resize 640x640 (Hiệu năng thô cho FPGA)
- **Mục tiêu**: Khảo sát triển khai phần cứng FPGA ZCU104 ứng dụng UAV Tracking

### Bảng kết quả so sánh định lượng

| Chỉ số đánh giá | Baseline (FP32) | Quantized (INT8) | Chênh lệch (Delta) |
| :--- | :---: | :---: | :---: |
| **Dung lượng mô hình** | 9.35 MB | 2.74 MB | **-70.7%** |
| **Precision (P)** | 92.56% | 86.67% | -5.89% |
| **Recall (R)** | 79.74% | 79.40% | -0.34% |
| **F1-Score** | 0.8567 | 0.8288 | -0.0280 |
| **mAP@0.5 (mAP50)** | 85.56% | 82.40% | **-3.16%** |
| **mAP@0.5:0.95 (mAP50-95)** | 45.04% | 41.40% | **-3.63%** |
| **Inference Latency (CPU)** | 27.06 ms | 52.08 ms | +25.02 ms |
| **Throughput (FPS)** | 37.0 FPS | 19.2 FPS | -17.8 FPS |

### Nhận xét & Đánh giá triển khai FPGA ZCU104
1. **Bảo toàn độ chính xác phát hiện**: mAP@0.5 của mô hình INT8 đạt **82.40%**, độ suy giảm chỉ **3.16%** so với FP32.
2. **Độ chính xác vị trí bounding box**: mAP@0.5:0.95 đạt **41.40%** (chênh lệch -3.63%).
3. **Tối ưu bộ nhớ on-chip**: Giảm kích thước trọng số **70.7%** (từ 9.35 MB xuống 2.74 MB), cực kỳ thuận lợi để nạp và tính toán trên BRAM/URAM và DRAM1 của FPGA ZCU104.
