# HƯỚNG DẪN CÁC SCRIPT KIỂM TRA CODE: LƯỢNG TỬ HÓA & KHẢO SÁT SIGMOID / SILU

Thư mục này tập hợp toàn bộ các mã nguồn (Python scripts) được tổ chức khoa học thành 2 nhóm chuyên biệt phục vụ cho việc kiểm tra, đối soát và chạy thực nghiệm triển khai YOLOv26n lên **FPGA Xilinx ZCU104 (XCZU7EV)**.

---

## 1. Nhóm Script Lượng Tử Hóa Conv (`scripts/quantize/`)

| Tên Script | Vai trò & Chức năng | Đầu vào (Inputs) | Đầu ra (Outputs) | Lệnh chạy mẫu |
| :--- | :--- | :--- | :--- | :--- |
| **`quantize_yolo26n.py`** | **Engine lượng tử hóa tuyến tính 102 lớp Conv:**<br>- Calibrate thang đo $S_X, S_Y$ trên tập validation.<br>- Lượng tử hóa trọng số $W \to q_W$ (INT8), Bias $\to q_B$ (INT32).<br>- Tính toán hệ số nhân dịch bit $M_{\text{mult}} / 2^{M_{\text{shift}}}$ cho DSP 16-bit.<br>- Trích xuất mảng text lưu vào `quantized_weight/` theo bản đồ DRAM.<br>- Xuất file ONNX `best_quantized_int8.onnx`. | `best.onnx`<br>`dram.csv`<br>`weight/`<br>`DUT_Anti_UAV_YOLO/` | `best_quantized_int8.onnx`<br>`quantized_weight/` (102 thư mục node)<br>`quantization_params.json`<br>`quantization_conv_layers.csv & .json` | `/home/quan/miniconda3/envs/container_detect/bin/python 18.9.26_quantize/scripts/quantize/quantize_yolo26n.py` |
| **`verify_quantization.py`** | **Xác minh sai số layer-by-layer:**<br>- So sánh cosine similarity và SNR từng lớp giữa FP32 và INT8.<br>- Kiểm tra tương thích suy luận end-to-end.<br>- Xuất báo cáo kiểm chứng JSON. | `best.onnx`<br>`best_quantized_int8.onnx`<br>`DUT_Anti_UAV_YOLO/` | `verification_report.json`<br>`result_fp32_vs_int8.png` | `/home/quan/miniconda3/envs/container_detect/bin/python 18.9.26_quantize/scripts/quantize/verify_quantization.py` |
| **`test_quantized_detection.py`** | **Kiểm tra trực quan phát hiện UAV:**<br>- Chạy suy luận 1 frame ảnh cụ thể.<br>- Vẽ bounding box so sánh kết quả phát hiện của model FP32 vs INT8. | `best.onnx`<br>`best_quantized_int8.onnx`<br>`00068.jpg` | `result_quantized_detection.jpg`<br>`comparison_fp32_vs_quantized.png` | `/home/quan/miniconda3/envs/container_detect/bin/python 18.9.26_quantize/scripts/quantize/test_quantized_detection.py` |
| **`eval_map.py`** | **Đo mAP chuẩn COCO ban đầu:**<br>- Đo Precision, Recall, F1, mAP@50, mAP@50:95 giữa FP32 và INT8.<br>- Hỗ trợ tùy chỉnh số lượng ảnh và split. | Model FP32 & INT8<br>`DUT_Anti_UAV_YOLO/` | `map_comparison_report.json`<br>`map_comparison_report.md`<br>`map_comparison_chart.png` | `/home/quan/miniconda3/envs/container_detect/bin/python 18.9.26_quantize/scripts/quantize/eval_map.py --num-samples 100` |
| **`benchmark_latency.py`** | **Đo chi tiết độ trễ (Latency & FPS) trên CPU:**<br>- Đo thời gian suy luận lặp lại (mean, median, p95, fps) giữa FP32 vs INT8 Conv vs Pure INT8 Fused LUT.<br>- Phân tích hiệu năng phục vụ so sánh CPU vs FPGA. | Model FP32, INT8, Pure INT8 LUT | Terminal output & Console report | `/home/quan/miniconda3/envs/container_detect/bin/python 18.9.26_quantize/scripts/quantize/benchmark_latency.py --runs 100` |
| **`plot_weight_density.py`** | **Vẽ biểu đồ mật độ trọng số:**<br>- Trích xuất phân bố trọng số các lớp Conv.<br>- Vẽ histogram so sánh FP32 vs INT8. | `best.onnx`<br>`quantized_weight/` | `weight_density_node_0.png`<br>`weight_density_node_21.png`<br>`weight_density_comparison.png` | `/home/quan/miniconda3/envs/container_detect/bin/python 18.9.26_quantize/scripts/quantize/plot_weight_density.py` |

---

## 2. Nhóm Script Khảo Sát Sigmoid / SiLU & Bảng Tra LUT (`scripts/sigmoid/`)

| Tên Script | Vai trò & Chức năng | Đầu vào (Inputs) | Đầu ra (Outputs) | Lệnh chạy mẫu |
| :--- | :--- | :--- | :--- | :--- |
| **`survey_sigmoid_range.py`** | **Khảo sát dải động 88 node Sigmoid:**<br>- Hook toàn bộ 88 tensor đầu vào hàm kích hoạt trên 100 frame validation.<br>- Thống kê $x_{\min}, x_{\max}, \mu, \sigma$, các phân vị $P_{0.1\%} .. P_{99.9\%}$.<br>- Tính tỉ lệ bão hòa ngoài $[-6, +6]$ và vùng tuyến tính $[-2, +2]$.<br>- Đánh giá định lượng sai số MAE, Max Error, SQNR giữa LUT vs PWL. | `best.onnx`<br>`DUT_Anti_UAV_YOLO/images/val/` | `sigmoid_range_summary.csv`<br>`sigmoid_range_profile.json`<br>`sigmoid_input_distribution.png`<br>`sigmoid_pwl_lut_comparison.png`<br>`sigmoid_error_distribution.png` | `/home/quan/miniconda3/envs/container_detect/bin/python 18.9.26_quantize/scripts/sigmoid/survey_sigmoid_range.py` |
| **`build_pure_int8_fused_lut.py`** | **Tạo mô hình thuần INT8 Fused LUT:**<br>- Triệt tiêu 100% `DequantizeLinear` sau lớp Conv.<br>- Xóa bỏ các node `Sigmoid`, `Mul` và `QuantizeLinear`.<br>- Tích hợp bảng tra `Gather` kiểu dữ liệu thuần **`TensorProto.INT8`** (256 byte mỗi bảng).<br>- Dữ liệu chạy trực tiếp dạng INT8 giữa các lớp Conv, không còn bất kỳ phép toán số thực nào. | `best_quantized_int8.onnx` | `best_pure_int8_fused_lut.onnx` (2.65 MB) | `/home/quan/miniconda3/envs/container_detect/bin/python 18.9.26_quantize/scripts/sigmoid/build_pure_int8_fused_lut.py` |
| **`eval_map_lut.py`** | **Đo và so sánh mAP 3 mô hình trên toàn bộ bộ dữ liệu:**<br>- So sánh đồng thời: Baseline FP32 vs INT8 Conv vs INT8 Fused LUT.<br>- Đo đạc trên 2,600 ảnh tập validation theo chuẩn IoU 10 ngưỡng COCO.<br>- Tính Precision, Recall, F1, mAP@50, mAP@50:95, Latency và FPS.<br>- Xuất biểu đồ so sánh 3 mô hình. | `best.onnx`<br>`best_quantized_int8.onnx`<br>`best_pure_int8_fused_lut.onnx`<br>`DUT_Anti_UAV_YOLO/` | `map_lut8_comparison_report.json`<br>`map_lut8_comparison_report.md`<br>`map_lut8_comparison_chart.png` | `/home/quan/miniconda3/envs/container_detect/bin/python 18.9.26_quantize/scripts/sigmoid/eval_map_lut.py --split val` |

---

## 3. Môi trường Thực thi
Tất cả các script trên đều được cấu hình và chạy tương thích hoàn toàn với môi trường Conda:
```bash
/home/quan/miniconda3/envs/container_detect/bin/python <đường_dẫn_script>
```
*Tất cả các script đều chạy từ thư mục gốc của repository `/home/quan/yolo26n_software`.*
