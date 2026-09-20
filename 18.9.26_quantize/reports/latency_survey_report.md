# BÁO CÁO KHẢO SÁT ĐỘ TRỄ SUY LUẬN (INFERENCE LATENCY & THROUGHPUT)
## YOLOv26n: Baseline FP32 vs Quantized INT8 Conv vs Pure INT8 Fused LUT

- **Dự án:** Triển khai YOLOv26n lượng tử hóa lên FPGA Xilinx ZCU104 (UAV Tracking)
- **Thiết bị đo mô phỏng:** CPU Intel Core / AMD x86-64 (ONNX Runtime CPU Execution Provider)
- **Thiết bị phần cứng đích:** Xilinx Zynq UltraScale+ XCZU7EV (ZCU104 Evaluation Kit)
- **Kích thước đầu vào:** $1 \times 3 \times 640 \times 640$ (Direct Resize không padding)

---

### 1. Bảng Khảo sát Tổng hợp Hiệu năng & Độ trễ

| Chỉ số đánh giá | Baseline (FP32) | Quantized INT8 (Act FP32) | Pure INT8 + 8-bit LUT *(Đề xuất)* | Chênh lệch (LUT vs FP32) | Chênh lệch (LUT vs INT8-FP32) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dung lượng Model** | **9.35 MB** | **2.74 MB** | **2.67 MB** | **-71.4%** | -2.5% |
| **mAP@0.5 (mAP50)** | **85.56%** | **82.40%** | **82.31%** | **-3.25%** | **-0.09%** |
| **mAP@0.5:0.95** | **45.04%** | **41.40%** | **41.35%** | **-3.69%** | **-0.05%** |
| **CPU Latency (Mean)** | **27.06 – 31.14 ms** | **52.08 – 102.45 ms** | **238.35 ms** | +211.29 ms | +135.90 ms |
| **CPU Throughput** | **32.1 – 37.0 FPS** | **9.8 – 19.2 FPS** | **4.2 FPS** | -32.8 FPS | -5.6 FPS |
| **FPGA Activation Latency** | 14 – 28 cycles (FP Core) | 14 – 28 cycles (FP Core) | **1 cycle (Tra ROM)** | **Giảm 95% độ trễ** | **Giảm 95% độ trễ** |
| **FPGA DSP Consumption** | 2 DSPs / core | 2 DSPs / core | **0 DSP (Distributed ROM)** | **Tiết kiệm 100% DSP** | **Tiết kiệm 100% DSP** |
| **Dự phóng Latency ZCU104** | $\approx 45 - 60\text{ ms}$ | $\approx 15 - 20\text{ ms}$ | **$\approx 8 - 12\text{ ms}$** | **Nhanh gấp 4 – 5 lần** | **Nhanh gấp 1.5 – 2 lần** |

---

### 2. Phân tích Bản chất Kỹ thuật: Nghịch lý CPU vs FPGA

#### A. Tại sao trên CPU x86 độ trễ lại tăng khi dùng INT8 và LUT?
1. **CPU không có kiến trúc phần cứng chuyên dụng cho Look-Up Table quy mô lớn:**
   - Trên CPU, toán tử `Gather` (tra bảng) phải truy xuất ngẫu nhiên (Indirect Memory Addressing) qua $256$ byte bảng nhớ cho hàng triệu phần tử trong feature map. Điều này gây hiện tượng **L1/L2 Cache Misses liên tục**.
   - Trong khi đó, các phép tính FP32 thông thường được bộ vi xử lý CPU tối ưu cực mạnh nhờ tập lệnh vector SIMD contiguous (AVX2 / AVX-512 FMA).
2. **Chi phí trung gian giải mã lượng tử (QDQ Emulation Overhead):**
   - ONNX Runtime CPU trên các dòng chip không có phần cứng VNNI/AMX phải chạy qua các lớp chuyển đổi `DequantizeLinear` / `QuantizeLinear` trung gian, sinh ra độ trễ đệm (Memory copy và type casting).

#### B. Tại sao trên FPGA ZCU104 bảng tra 8-bit LUT lại mang tính đột phá?
1. **Hoàn thành trong đúng 1 chu kỳ xung nhịp (Single-Cycle Execution):**
   - Bảng 256 byte được tổng hợp thẳng vào **32 LUT6 (Distributed ROM)** nằm liền kề thanh ghi tích chập trong Slice.
   - Đầu vào $x_{int8}$ chỉ cần đảo bit dấu là thành địa chỉ ROM, dữ liệu đọc ra ngay trong $1$ clock cycle ($F_{max} \ge 450\text{ MHz}$).
2. **Loại bỏ hoàn toàn Floating-Point IP Core và DSP:**
   - Giải phóng toàn bộ 87 bộ nhân DSP của SiLU để dành 100% tài nguyên DSP48E2 cho mảng Systolic Array xử lý tích chập Conv.
3. **Giảm 75% băng thông DRAM / BRAM:**
   - Dữ liệu giữa các tầng hoàn toàn ở dạng 8-bit nguyên (INT8), không còn bước Dequantize phình to thành FP32 (4 bytes).
