# BÁO CÁO KHẢO SÁT CHUYÊN SÂU: DẢI ĐỘNG HÀM KÍCH HOẠT SIGMOID & ĐÁNH GIÁ PHƯƠNG PHÁP XẤP XỈ PHẦN CỨNG (LUT VS PWL) TRÊN FPGA ZCU104

**Dự án:** Triển khai Kiến trúc YOLOv26n Lượng tử hóa Lên FPGA Xilinx ZCU104 (Ứng dụng UAV Tracking)  
**Ngày khảo sát:** 19/09/2026  
**Thư mục lưu trữ:** `18.9.26_quantize/sigmoid_analysis/`

---

## 1. Đặt Vấn đề & Thực trạng Phần cứng

Trong mạng YOLOv26n đã lượng tử hóa tuyến tính (Linear Quantization INT8), các khối tích chập (Conv) xử lý dữ liệu hoàn toàn ở dạng số nguyên:
$$\text{acc}_{32} = \text{Conv}(q_W, q_X) + q_{\text{bias}}$$
$$q_Y = \text{clip}\left( \left\lfloor \frac{\text{acc}_{32} \cdot M_{\text{mult}}}{2^{M_{\text{shift}}}} \right\rceil, -128, 127 \right) \in \text{INT8}$$

Tuy nhiên, tại các vị trí hàm phi tuyến (Sigmoid và SiLU), quy trình thông thường đang phải **Dequantize về số thực dấu phẩy động FP32**:
$$x = q_Y \cdot S_Y \quad (\text{Dequantize INT8} \to \text{FP32})$$
$$y = \sigma(x) = \frac{1}{1 + e^{-x}} \quad (\text{FP32 Exponential \& Division})$$
$$\text{SiLU}(x) = x \cdot \sigma(x) \quad (\text{FP32 Multiplication})$$
$$q_{\text{next}} = \text{round}\left( \frac{\text{SiLU}(x)}{S_{\text{next}}} \right) \quad (\text{Quantize FP32} \to \text{INT8})$$

### Rào cản chí mạng trên FPGA Xilinx ZCU104 (Zynq UltraScale+ XCZU7EV):
1. **Lãng phí tài nguyên DSP & Logic:** Tính toán hàm mũ $e^{-x}$ và phép chia số thực đòi hỏi sử dụng Floating-Point IP Core của Vivado (tốn hàng chục DSP48E2 và hàng trăm Slice LUTs cho mỗi kênh).
2. **Nghẽn cổ chai độ trễ (Latency Bottleneck):** Phép tính FP32 mất từ $14 - 28$ chu kỳ clock, làm đứt gãy luồng xử lý streaming dữ liệu từ Systolic Array / BRAM FIFO.
3. **Mục tiêu khảo sát:** Thay thế hoàn toàn phép tính FP32 bằng **Bảng tra trước (Look-Up Table - LUT)** hoặc **Xấp xỉ đoạn tuyến tính (Piecewise Linear - PWL)** hoạt động trực tiếp trên miền Fixed-Point / INT8.

---

## 2. Phát hiện Cấu trúc: 88 Node Sigmoid trong YOLOv26n

Khảo sát toàn diện đồ thị tính toán ONNX (`best.onnx`) phát hiện chính xác **88 node Sigmoid**:

| Nhóm Node | Số lượng | Mẫu kết nối (Graph Pattern) | Bản chất hàm toán học | Chức năng trong YOLOv26n |
| :--- | :---: | :--- | :--- | :--- |
| **Nhóm 1: SiLU Activations** | **87 nodes** | `Conv -> Sigmoid -> Mul` | $\text{SiLU}(x) = x \cdot \sigma(x)$ | Nằm ngay sau mỗi lớp Conv trong Backbone, Neck và các nhánh Head. |
| **Nhóm 2: Head Classification** | **1 node** (Node 363) | `Concat -> Sigmoid -> Concat` | $\sigma(\text{logit}) = \frac{1}{1 + e^{-\text{logit}}}$ | Tính xác suất tin cậy (Confidence Score) cho $8400$ anchor boxes tại Detect Head. |

---

## 3. Kết quả Khảo sát Dải động Thực nghiệm (Empirical Range Profiling)

Khảo sát được thực hiện trên 100 frame trích xuất từ tập dữ liệu bay không người lái `DUT_Anti_UAV_YOLO/images/val/`.

### 3.1. Thống kê Dải động Toàn cục

| Đặc tính Dải động | Nhóm SiLU (87 Conv Nodes) | Nhóm Sigmoid Head (Node 363) | Ý nghĩa Phần cứng |
| :--- | :---: | :---: | :--- |
| **Cực trị (Extreme Min..Max)** | `[-72.5248, +87.1715]` | `[-62.1202, +3.1235]` | Xuất hiện ngoại lai rất hiếm gặp |
| **99.8% Dải tin cậy ($P_{0.1\%} .. P_{99.9\%}$)** | `[-12.9861, +11.6100]` | `[-38.4417, -10.7870]` | **Dải động thực tế cần bao phủ** |
| **90% Dải động trung tâm ($P_5 .. P_{95}$)** | `[-2.2355, +2.8261]` | `[-22.4100, -13.2500]` | Tập trung phần lớn dữ liệu |
| **Giá trị trung bình $\pm$ Độ lệch chuẩn ($\mu \pm \sigma$)** | `+0.1424 ± 1.8475` | `-17.8622 ± 2.5825` | Phân bố lệch rõ rệt |
| **Tỉ lệ vùng tuyến tính ($x \in [-2.0, +2.0]$)** | **$83.34\%$** | $0.00\%$ | Độ nhạy cao với độ dốc tại $x=0$ |
| **Tỉ lệ bão hòa âm ($x < -6.0$)** | $0.85\%$ | **$99.98\%$** | $\sigma(x) < 0.0025 \approx 0$ |
| **Tỉ lệ bão hòa dương ($x > +6.0$)** | $0.25\%$ | $0.00\%$ | $\sigma(x) > 0.9975 \approx 1$ |
| **Tổng tỉ lệ bão hòa ($|x| > 6.0$)** | **$1.10\%$** | **$99.98\%$** | Chỉ cần xấp xỉ trong $[-6.0, +6.0]$ |

### 3.2. Phát hiện Đặc biệt quan trọng đối với Detect Head (Node 363)
- Với bài toán UAV Tracking, trên tổng số $8400$ ô lưới dự đoán của 3 tỷ lệ (P3: $80 \times 80$, P4: $40 \times 40$, P5: $20 \times 20$), hầu như toàn bộ khung hình là nền trời / cảnh vật trống (Background).
- Do đó, **$99.98\%$ logits đầu vào có giá trị rất âm ($x < -6.0$, trung bình là $-17.86$)**, tương ứng xác suất phát hiện $\approx 0$.
- Chỉ có một vài bounding boxes chứa UAV mục tiêu mới có logit dương (đạt cực đại $+3.1235$, cho ra score $\sigma(3.12) \approx 0.9578$).
- **Chiến lược Tối ưu Phần cứng:** Bộ xử lý Detect Head trên FPGA có thể áp dụng cơ chế **Zero-Bypass / Threshold Detection**: Nếu $x < -6.0$ (hoặc bit dấu âm và $|x| \ge 6$), đầu ra trực tiếp được gán bằng $0$ mà **không cần tra bảng hay tính toán**.

---

## 4. Đánh giá & So sánh Chi tiết: LUT vs PWL

Khảo sát đã mô phỏng trên dải biểu diễn $x \in [-8.0, +8.0]$ với các kiến trúc phần cứng khác nhau:

### 4.1. Bảng Thông số Kỹ thuật & Độ chính xác

| Phương pháp Xấp xỉ | Định dạng & Cấu trúc | Max Error (Sig) | MAE (Sig) | SQNR Sig (dB) | Max Error (SiLU) | Chi phí Tài nguyên FPGA (ZCU104) | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **FP32 Ground Truth** | Chuẩn IEEE-754 | $0.000000$ | $0.000000$ | $\infty$ | $0.000000$ | Floating-Point Core (Hàng trăm LUTs, 2 DSPs) | 14 - 28 ck |
| **LUT-256 (8-bit)** | 256 entries $\times$ 8-bit | **$0.007624$** | **$0.000976$** | **$51.11\text{ dB}$** | **$0.007045$** | **32 LUT6 (Distributed ROM)**, **0 DSP**, **0 BRAM** | **1 ck** |
| **LUT-512 (9-bit)** | 512 entries $\times$ 8-bit | $0.003894$ | $0.000488$ | $57.13\text{ dB}$ | $0.003500$ | 64 LUT6 (Distributed ROM), **0 DSP**, **0 BRAM** | 1 ck |
| **LUT-1024 (10-bit)**| 1024 entries $\times$ 8-bit | $0.001941$ | $0.000244$ | $63.15\text{ dB}$ | $0.001741$ | 0.5 BRAM18K (hoặc 128 LUT6), **0 DSP** | 1 ck |
| **PWL-4 (Shift-Add)** | 4 đoạn đối xứng, hệ số $2^{-k}$ | $0.022310$ | $0.004551$ | $40.01\text{ dB}$ | $0.111546$ | 48 LUTs, 24 FFs, **0 DSP** | 2 ck |
| **PWL-8 (Shift-Add)** | 8 đoạn đối xứng, hệ số $2^{-k}$ | $0.064850$ | $0.013850$ | $29.20\text{ dB}$ | $0.259372$ | 95 LUTs, 40 FFs, **0 DSP** | 2 ck |
| **PWL-8 (Optimal)** | 8 đoạn hồi quy tuyến tính | $0.004323$ | $0.000833$ | $55.34\text{ dB}$ | $0.014818$ | 75 LUTs, 30 FFs, **1 DSP48E2** | 2 ck |

> [!NOTE]
> - **Ngưỡng chất lượng INT8:** Một hệ thống lượng tử 8-bit chuẩn có SQNR lý thuyết xấp xỉ $6.02 \times 8 + 1.76 \approx 49.92\text{ dB}$.
> - **LUT-256 đạt $51.11\text{ dB}$**, vượt qua ngưỡng SQNR của dữ liệu 8-bit! Nghĩa là sai số của bảng tra LUT-256 nhỏ hơn cả sai số lượng tử hóa vốn có của mô hình INT8.

---

## 5. Đột phá Kiến trúc: Fused INT8 SiLU LUT

Đối với 87 node SiLU trong YOLOv26n, chúng ta không cần tính riêng rẽ $\sigma(x)$ rồi nhân với $x$. Thay vào đó, áp dụng **Bảng Tra Hợp Nhất (Fused LUT)**:

### Sơ đồ Luồng Xử lý Fused:
```
[Requantized Conv Output: q_Y (INT8)] 
                 │
                 ▼
     ┌───────────────────────┐
     │  silu_fused_lut_256   │  <─── ROM 256 bytes (Chỉ tốn 32 LUT6, 0 DSP, 0 BRAM)
     │  Bảng tra 1 chu kỳ    │
     └───────────────────────┘
                 │
                 ▼
[Next Conv Layer Input: q_next (INT8)]
```

### Công thức Nạp Bảng trước khi Tổng hợp:
Với mỗi giá trị chỉ số nguyên $q_Y \in [-128, 127]$:
$$\text{ROM}[q_Y] = \text{clip}\left( \left\lfloor \frac{(q_Y \cdot S_Y) \cdot \sigma(q_Y \cdot S_Y)}{S_{\text{next}}} \right\rceil, -128, 127 \right) \in \text{INT8}$$

**Ưu điểm vượt trội:**
1. **Triệt tiêu hoàn toàn 4 bước FP32:** Không dequantize, không tính số mũ, không nhân số thực, không requantize.
2. **Độ trễ tối thiểu:** Đúng **1 chu kỳ clock**, tần số hoạt động $F_{\max} \ge 450\text{ MHz}$ trên UltraScale+.
3. **Không tốn DSP:** Tiết kiệm toàn bộ $87$ bộ nhân số thực hoặc DSP blocks để dành riêng cho ma trận tích chập Conv.

---

## 6. Kết luận & Đề xuất Triển khai trên FPGA ZCU104

1. **Khuyến nghị Lựa chọn:**
   - **Nên chọn LUT-256 thay vì PWL:** Trên FPGA Xilinx UltraScale+, 1 bảng tra 256 phần tử 8-bit chỉ tốn đúng **32 LUT6** (chưa đến $0.014\%$ tổng số 230,400 LUT của chip XCZU7EV). LUT cho độ chính xác cao hơn hẳn PWL không dùng DSP, thời gian tính toán chỉ 1 chu kỳ và không gặp hiện tượng gãy khúc đạo hàm tại các điểm nối đoạn.
   - **Đối với SiLU (87 nodes):** Sử dụng mô-đun `silu_fused_lut_256.v` với hệ số nạp sẵn theo cặp scale $(S_Y, S_{\text{next}})$.
   - **Đối với Sigmoid Head (1 node):** Sử dụng mô-đun `sigmoid_lut_256.v` kết hợp điều kiện Zero-Bypass khi logit âm sâu ($x < -6.0$).

2. **Các File Mã Nguồn & Báo Cáo đã được Tổ chức Sạch sẽ:**
   - Script khảo sát: `18.9.26_quantize/sigmoid_analysis/scripts/survey_sigmoid_range.py`
   - Bảng số liệu chi tiết 88 nodes: `18.9.26_quantize/sigmoid_analysis/data/sigmoid_range_summary.csv`
   - Dữ liệu JSON thống kê: `18.9.26_quantize/sigmoid_analysis/data/sigmoid_range_profile.json`
   - Biểu đồ phân bố đầu vào: `18.9.26_quantize/sigmoid_analysis/plots/sigmoid_input_distribution.png`
   - Biểu đồ so sánh đường cong: `18.9.26_quantize/sigmoid_analysis/plots/sigmoid_pwl_lut_comparison.png`
   - Biểu đồ sai số chi tiết: `18.9.26_quantize/sigmoid_analysis/plots/sigmoid_error_distribution.png`
   - Mã nguồn Verilog LUT: `18.9.26_quantize/sigmoid_analysis/verilog/sigmoid_lut_256.v`
   - Mã nguồn Verilog PWL 8 đoạn: `18.9.26_quantize/sigmoid_analysis/verilog/sigmoid_pwl_8seg.v`
   - Mã nguồn Verilog Fused SiLU: `18.9.26_quantize/sigmoid_analysis/verilog/silu_fused_lut_256.v`
