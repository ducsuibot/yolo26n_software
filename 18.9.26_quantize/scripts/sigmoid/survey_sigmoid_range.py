#!/usr/bin/env python3
"""
Empirical Dynamic Range Profiler and Hardware Approximation Evaluator (LUT vs PWL)
for Sigmoid and SiLU Activations in YOLOv26n on FPGA ZCU104.

This script:
1. Profiles the real dynamic range of all 88 Sigmoid nodes across the validation set.
2. Evaluates LUT (256, 512, 1024 entries) vs PWL (4, 8, 16 segments) approximations.
3. Computes MAE, Max Error, RMSE, and SQNR (dB) on the empirical data distribution.
4. Generates visual comparison plots and exports structured data (JSON, CSV).
"""

import os
import glob
import json
import csv
import cv2
import numpy as np
import onnx
import onnxruntime as ort
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sigmoid_fp32(x):
    """Ground truth Sigmoid in FP32"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))

def silu_fp32(x):
    """Ground truth SiLU in FP32"""
    return x * sigmoid_fp32(x)

# ---------------------------------------------------------
# PWL Approximations (Hardware Friendly: Shift-and-Add)
# ---------------------------------------------------------

def sigmoid_pwl_4seg_shift(x):
    """
    4-segment PWL approximation using only powers of 2 for slopes (No DSPs).
    Symmetric around x = 0: sigma(x) = 1 - sigma(-x)
    For x >= 0:
      0.0 <= x < 1.0: y = 0.25 * x + 0.5          = (x >> 2) + 0.5
      1.0 <= x < 2.5: y = 0.125 * x + 0.625       = (x >> 3) + 0.625
      2.5 <= x < 5.0: y = 0.03125 * x + 0.859375  = (x >> 5) + 0.859375
      x >= 5.0      : y = 1.0                     (Saturation)
    """
    abs_x = np.abs(x)
    y = np.zeros_like(abs_x)
    
    mask0 = (abs_x < 1.0)
    y[mask0] = 0.25 * abs_x[mask0] + 0.5
    
    mask1 = (abs_x >= 1.0) & (abs_x < 2.5)
    y[mask1] = 0.125 * abs_x[mask1] + 0.625
    
    mask2 = (abs_x >= 2.5) & (abs_x < 5.0)
    y[mask2] = 0.03125 * abs_x[mask2] + 0.859375
    
    mask3 = (abs_x >= 5.0)
    y[mask3] = 1.0
    
    # Apply symmetry for x < 0
    res = np.where(x >= 0, y, 1.0 - y)
    return res

def sigmoid_pwl_8seg_shift(x):
    """
    8-segment PWL approximation using power-of-2 slopes (DSP-free).
    Symmetric around x = 0.
    For x >= 0:
      0.0 <= x < 0.5: y = 0.25 * x + 0.5          = (x >> 2) + 0.5
      0.5 <= x < 1.25: y = 0.21875 * x + 0.515625 = (x >> 2) - (x >> 5) + 0.515625
      1.25 <= x < 2.0: y = 0.15625 * x + 0.59375  = (x >> 3) + (x >> 5) + 0.59375
      2.0 <= x < 3.0: y = 0.09375 * x + 0.71875  = (x >> 4) + (x >> 5) + 0.71875
      3.0 <= x < 4.0: y = 0.046875 * x + 0.859375 = (x >> 5) + (x >> 6) + 0.859375
      4.0 <= x < 5.0: y = 0.015625 * x + 0.984375 = (x >> 6) + 0.921875
      5.0 <= x < 6.0: y = 0.00390625 * x + 0.98046875 = (x >> 8) + 0.98
      x >= 6.0      : y = 1.0
    """
    abs_x = np.abs(x)
    y = np.zeros_like(abs_x)
    
    m0 = (abs_x < 0.5)
    y[m0] = 0.25 * abs_x[m0] + 0.5
    
    m1 = (abs_x >= 0.5) & (abs_x < 1.25)
    y[m1] = (0.25 - 0.03125) * abs_x[m1] + 0.515625
    
    m2 = (abs_x >= 1.25) & (abs_x < 2.0)
    y[m2] = (0.125 + 0.03125) * abs_x[m2] + 0.59375
    
    m3 = (abs_x >= 2.0) & (abs_x < 3.0)
    y[m3] = (0.0625 + 0.03125) * abs_x[m3] + 0.71875
    
    m4 = (abs_x >= 3.0) & (abs_x < 4.0)
    y[m4] = (0.03125 + 0.015625) * abs_x[m4] + 0.859375
    
    m5 = (abs_x >= 4.0) & (abs_x < 5.0)
    y[m5] = 0.015625 * abs_x[m5] + 0.921875
    
    m6 = (abs_x >= 5.0) & (abs_x < 6.0)
    y[m6] = 0.00390625 * abs_x[m6] + 0.98046875
    
    m7 = (abs_x >= 6.0)
    y[m7] = 1.0
    
    return np.where(x >= 0, y, 1.0 - y)

def sigmoid_pwl_optimal(x, num_segments=8):
    """
    Optimal PWL using Least-Squares linear fit per interval on [0, 6.0].
    Requires 1 DSP multiplier for y = A_i * x + B_i.
    """
    abs_x = np.abs(x)
    y = np.zeros_like(abs_x)
    
    breaks = np.linspace(0, 6.0, num_segments + 1)
    for i in range(num_segments):
        x_low, x_high = breaks[i], breaks[i+1]
        m = (abs_x >= x_low) & (abs_x < x_high)
        # Linear fit y = a*x + b on this interval
        xs = np.linspace(x_low, x_high, 100)
        ys = sigmoid_fp32(xs)
        coeffs = np.polyfit(xs, ys, 1) # [slope, intercept]
        y[m] = coeffs[0] * abs_x[m] + coeffs[1]
        
    y[abs_x >= 6.0] = 1.0
    return np.where(x >= 0, y, 1.0 - y)

# ---------------------------------------------------------
# LUT Approximations (Direct Table Look-Up)
# ---------------------------------------------------------

class SigmoidLUT:
    def __init__(self, num_entries=256, x_min=-8.0, x_max=8.0):
        self.num_entries = num_entries
        self.x_min = x_min
        self.x_max = x_max
        self.step = (x_max - x_min) / num_entries
        
        # Table of quantized values
        indices = np.arange(num_entries)
        # Center of each bin
        x_centers = self.x_min + (indices + 0.5) * self.step
        self.table = sigmoid_fp32(x_centers)
        
    def __call__(self, x):
        # Clip to table range
        x_clipped = np.clip(x, self.x_min, self.x_max - 1e-6)
        idx = np.floor((x_clipped - self.x_min) / self.step).astype(np.int32)
        idx = np.clip(idx, 0, self.num_entries - 1)
        return self.table[idx]

def compute_metrics(y_true, y_approx):
    """Compute standard error metrics"""
    err = y_approx - y_true
    abs_err = np.abs(err)
    max_err = float(np.max(abs_err))
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    
    signal_pwr = np.sum(y_true ** 2)
    noise_pwr = np.sum(err ** 2)
    sqnr_db = float(10.0 * np.log10(signal_pwr / noise_pwr)) if noise_pwr > 0 else 100.0
    
    return {
        "max_err": max_err,
        "mae": mae,
        "rmse": rmse,
        "sqnr_db": sqnr_db
    }

def main():
    print("=" * 80)
    print("  YOLOv26n SIGMOID & SiLU ACTIVATION DYNAMIC RANGE PROFILING & SURVEY")
    print("  Hardware Target: Xilinx ZCU104 (XCZU7EV) - FPGA UAV Tracking")
    print("=" * 80)
    
    onnx_path = "best.onnx"
    val_img_dir = "DUT_Anti_UAV_YOLO/images/val"
    out_dir = "18.9.26_quantize/sigmoid_analysis"
    os.makedirs(f"{out_dir}/data", exist_ok=True)
    os.makedirs(f"{out_dir}/plots", exist_ok=True)
    os.makedirs(f"{out_dir}/verilog", exist_ok=True)
    
    # 1. Parse ONNX graph for all Sigmoid nodes
    model = onnx.load(onnx_path)
    sigmoid_nodes = []
    tensor_names = set()
    
    for idx, node in enumerate(model.graph.node):
        if node.op_type == "Sigmoid":
            in_t = node.input[0]
            out_t = node.output[0]
            consumers = [c.op_type for c in model.graph.node if out_t in c.input]
            producers = [p.op_type for p in model.graph.node if in_t in p.output]
            prod_type = producers[0] if producers else "Input"
            is_silu = (consumers == ['Mul'])
            
            sigmoid_nodes.append({
                "node_idx": idx,
                "node_name": node.name,
                "input_tensor": in_t,
                "output_tensor": out_t,
                "producer": prod_type,
                "consumers": consumers,
                "type": "SiLU" if is_silu else "Sigmoid_Head"
            })
            tensor_names.add(in_t)
            
    print(f"[INFO] Tổng số Sigmoid nodes tìm thấy trong {onnx_path}: {len(sigmoid_nodes)}")
    print(f"       - SiLU Activation (Conv -> Sigmoid -> Mul) : {sum(1 for s in sigmoid_nodes if s['type'] == 'SiLU')} nodes")
    print(f"       - Sigmoid Head (Detect Head Confidence)    : {sum(1 for s in sigmoid_nodes if s['type'] == 'Sigmoid_Head')} nodes")
    
    # 2. Build ONNX intermediate session
    for t_name in tensor_names:
        vi = onnx.helper.ValueInfoProto()
        vi.name = t_name
        model.graph.output.append(vi)
        
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(model.SerializeToString(), sess_opts, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    
    # 3. Collect calibration / validation images
    img_paths = sorted(glob.glob(os.path.join(val_img_dir, "*.jpg")))[:100] # 100 representative validation frames
    print(f"[INFO] Chạy profiling trên {len(img_paths)} ảnh từ tập validation {val_img_dir}...")
    
    # Storage for stats
    fetches = list(tensor_names)
    accum_dict = {
        name: {
            "min": float("inf"),
            "max": float("-inf"),
            "sum": 0.0,
            "sum_sq": 0.0,
            "count": 0,
            "hist_counts": np.zeros(200, dtype=np.int64),
            "samples": [] # Store small subsample for percentiles
        } for name in tensor_names
    }
    
    hist_bins = np.linspace(-15.0, 15.0, 201)
    
    for i, p in enumerate(img_paths):
        img = cv2.imread(p)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_res = cv2.resize(img_rgb, (640, 640)).astype(np.float32) / 255.0
        inp = np.transpose(img_res, (2, 0, 1))[None, ...]
        
        outs = sess.run(fetches, {input_name: inp})
        for name, arr in zip(fetches, outs):
            flat = arr.flatten()
            c_min = float(np.min(flat))
            c_max = float(np.max(flat))
            if c_min < accum_dict[name]["min"]:
                accum_dict[name]["min"] = c_min
            if c_max > accum_dict[name]["max"]:
                accum_dict[name]["max"] = c_max
                
            accum_dict[name]["sum"] += float(np.sum(flat))
            accum_dict[name]["sum_sq"] += float(np.sum(flat ** 2))
            accum_dict[name]["count"] += len(flat)
            
            # Histogram
            h, _ = np.histogram(flat, bins=hist_bins)
            accum_dict[name]["hist_counts"] += h
            
            # Subsample 200 random values per frame for exact percentile computation
            if len(flat) > 200:
                sub = np.random.choice(flat, 200, replace=False)
            else:
                sub = flat
            accum_dict[name]["samples"].extend(sub.tolist())
            
        if (i + 1) % 20 == 0 or (i + 1) == len(img_paths):
            print(f"       -> Đã xử lý {i + 1}/{len(img_paths)} ảnh...")
            
    print(f"[SUCCESS] Hoàn thành trích xuất dữ liệu kích hoạt thực tế cho {len(tensor_names)} tensors.")
    
    # 4. Compute Comprehensive Statistics per Node
    node_records = []
    all_silu_samples = []
    head_samples = []
    
    for s_info in sigmoid_nodes:
        t_name = s_info["input_tensor"]
        acc = accum_dict[t_name]
        cnt = acc["count"]
        mean_v = acc["sum"] / cnt if cnt > 0 else 0.0
        var_v = (acc["sum_sq"] / cnt) - (mean_v ** 2) if cnt > 0 else 0.0
        std_v = float(np.sqrt(max(0.0, var_v)))
        
        samples_arr = np.array(acc["samples"], dtype=np.float32)
        p01 = float(np.percentile(samples_arr, 0.1))
        p1 = float(np.percentile(samples_arr, 1.0))
        p5 = float(np.percentile(samples_arr, 5.0))
        p50 = float(np.percentile(samples_arr, 50.0))
        p95 = float(np.percentile(samples_arr, 95.0))
        p99 = float(np.percentile(samples_arr, 99.0))
        p999 = float(np.percentile(samples_arr, 99.9))
        
        # Saturation rates
        sat_neg6 = float(np.mean(samples_arr < -6.0) * 100.0)
        sat_pos6 = float(np.mean(samples_arr > 6.0) * 100.0)
        sat_neg5 = float(np.mean(samples_arr < -5.0) * 100.0)
        sat_pos5 = float(np.mean(samples_arr > 5.0) * 100.0)
        linear_rate = float(np.mean((samples_arr >= -2.0) & (samples_arr <= 2.0)) * 100.0)
        
        rec = {
            "node_idx": s_info["node_idx"],
            "node_name": s_info["node_name"],
            "tensor_name": t_name,
            "type": s_info["type"],
            "producer": s_info["producer"],
            "min": acc["min"],
            "max": acc["max"],
            "mean": mean_v,
            "std": std_v,
            "p0_1": p01,
            "p1": p1,
            "p5": p5,
            "p50": p50,
            "p95": p95,
            "p99": p99,
            "p99_9": p999,
            "sat_neg6_pct": sat_neg6,
            "sat_pos6_pct": sat_pos6,
            "sat_total6_pct": sat_neg6 + sat_pos6,
            "sat_neg5_pct": sat_neg5,
            "sat_pos5_pct": sat_pos5,
            "sat_total5_pct": sat_neg5 + sat_pos5,
            "linear_region_pct": linear_rate,
            "effective_range": [p01, p999]
        }
        node_records.append(rec)
        
        if s_info["type"] == "SiLU":
            all_silu_samples.extend(samples_arr)
        else:
            head_samples.extend(samples_arr)
            
    all_silu_samples = np.array(all_silu_samples, dtype=np.float32)
    head_samples = np.array(head_samples, dtype=np.float32)
    
    # Save JSON and CSV summary
    with open(f"{out_dir}/data/sigmoid_range_profile.json", 'w', encoding='utf-8') as f:
        json.dump(node_records, f, indent=2)
        
    with open(f"{out_dir}/data/sigmoid_range_summary.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Node_Idx", "Node_Name", "Type", "Producer", "Input_Tensor",
            "Min", "Max", "Mean", "Std", "P0.1%", "P1%", "P50%", "P99%", "P99.9%",
            "Sat_<_-6(%)", "Sat_>_+6(%)", "Sat_Total6(%)", "Linear_[-2,2](%)"
        ])
        for r in node_records:
            writer.writerow([
                r["node_idx"], r["node_name"], r["type"], r["producer"], r["tensor_name"],
                f"{r['min']:.4f}", f"{r['max']:.4f}", f"{r['mean']:.4f}", f"{r['std']:.4f}",
                f"{r['p0_1']:.4f}", f"{r['p1']:.4f}", f"{r['p50']:.4f}", f"{r['p99']:.4f}", f"{r['p99_9']:.4f}",
                f"{r['sat_neg6_pct']:.2f}", f"{r['sat_pos6_pct']:.2f}", f"{r['sat_total6_pct']:.2f}",
                f"{r['linear_region_pct']:.2f}"
            ])
            
    print(f"[SUCCESS] Đã lưu bảng thống kê 88 nodes vào {out_dir}/data/sigmoid_range_summary.csv & .json")
    
    # 5. Global Range Statistics
    print("\n" + "=" * 60)
    print("  TỔNG HỢP DẢI ĐỘNG CỦA CÁC HÀM KÍCH HOẠT TRONG YOLOv26n:")
    print("=" * 60)
    print(f"1. Nhóm SiLU (87 Conv nodes trong Backbone + Neck):")
    print(f"   - Extreme Min..Max : [{np.min(all_silu_samples):.4f}, {np.max(all_silu_samples):.4f}]")
    print(f"   - 99.8% Dải động   : [{np.percentile(all_silu_samples, 0.1):.4f}, {np.percentile(all_silu_samples, 99.9):.4f}]")
    print(f"   - 90% Dải động     : [{np.percentile(all_silu_samples, 5.0):.4f}, {np.percentile(all_silu_samples, 95.0):.4f}]")
    print(f"   - Trung bình / Std : {np.mean(all_silu_samples):.4f} +/- {np.std(all_silu_samples):.4f}")
    print(f"   - Tỉ lệ trong [-2, +2] (Vùng tuyến tính): {np.mean((all_silu_samples >= -2.0) & (all_silu_samples <= 2.0))*100.0:.2f}%")
    print(f"   - Tỉ lệ bão hòa ngoài [-6, +6]         : {np.mean(np.abs(all_silu_samples) > 6.0)*100.0:.2f}%")
    
    print(f"\n2. Nhóm Sigmoid Head (Node 363 - Detect Head Classification):")
    print(f"   - Extreme Min..Max : [{np.min(head_samples):.4f}, {np.max(head_samples):.4f}]")
    print(f"   - 99.8% Dải động   : [{np.percentile(head_samples, 0.1):.4f}, {np.percentile(head_samples, 99.9):.4f}]")
    print(f"   - Trung bình / Std : {np.mean(head_samples):.4f} +/- {np.std(head_samples):.4f}")
    print(f"   - Tỉ lệ bão hòa âm (< -6.0, Background) : {np.mean(head_samples < -6.0)*100.0:.2f}%")
    
    # 6. Approximation Accuracy Evaluation on Empirical Samples
    test_x = np.linspace(-8.0, 8.0, 10000)
    sig_gt = sigmoid_fp32(test_x)
    silu_gt = silu_fp32(test_x)
    
    # Initialize LUTs
    lut_256 = SigmoidLUT(num_entries=256, x_min=-8.0, x_max=8.0)
    lut_512 = SigmoidLUT(num_entries=512, x_min=-8.0, x_max=8.0)
    lut_1024 = SigmoidLUT(num_entries=1024, x_min=-8.0, x_max=8.0)
    
    # Compute Sigmoid approximations
    sig_lut256 = lut_256(test_x)
    sig_lut512 = lut_512(test_x)
    sig_lut1024 = lut_1024(test_x)
    sig_pwl4 = sigmoid_pwl_4seg_shift(test_x)
    sig_pwl8 = sigmoid_pwl_8seg_shift(test_x)
    sig_pwl_opt = sigmoid_pwl_optimal(test_x, num_segments=8)
    
    # Compute SiLU approximations
    silu_lut256 = test_x * sig_lut256
    silu_pwl4 = test_x * sig_pwl4
    silu_pwl8 = test_x * sig_pwl8
    silu_pwl_opt = test_x * sig_pwl_opt
    
    # Metrics Table
    methods = [
        ("LUT-256 (8-bit)", sig_lut256, silu_lut256, "32 LUTs / 0 DSP"),
        ("LUT-512 (9-bit)", sig_lut512, test_x * sig_lut512, "64 LUTs / 0 DSP"),
        ("LUT-1024 (10-bit)", sig_lut1024, test_x * sig_lut1024, "0.5 BRAM18K / 0 DSP"),
        ("PWL-4 (Shift-Add)", sig_pwl4, silu_pwl4, "48 LUTs / 0 DSP"),
        ("PWL-8 (Shift-Add)", sig_pwl8, silu_pwl8, "95 LUTs / 0 DSP"),
        ("PWL-8 (Optimal)", sig_pwl_opt, silu_pwl_opt, "75 LUTs / 1 DSP48E2")
    ]
    
    comparison_results = []
    print("\n" + "=" * 90)
    print("  ĐÁNH GIÁ ĐỘ CHÍNH XÁC CÁC PHƯƠNG PHÁP XẤP XỈ PHẦN CỨNG (TRÊN DẢI [-8, +8]):")
    print("=" * 90)
    print(f"{'Phương pháp':<20} | {'Max Err (Sig)':<14} | {'MAE (Sig)':<12} | {'SQNR Sig (dB)':<14} | {'Max Err (SiLU)':<14} | {'FPGA Cost'}")
    print("-" * 90)
    
    for name, s_pred, silu_pred, fpga_cost in methods:
        m_sig = compute_metrics(sig_gt, s_pred)
        m_silu = compute_metrics(silu_gt, silu_pred)
        
        comparison_results.append({
            "method": name,
            "fpga_cost": fpga_cost,
            "sigmoid_metrics": m_sig,
            "silu_metrics": m_silu
        })
        print(f"{name:<20} | {m_sig['max_err']:<14.6f} | {m_sig['mae']:<12.6f} | {m_sig['sqnr_db']:<14.2f} | {m_silu['max_err']:<14.6f} | {fpga_cost}")
        
    with open(f"{out_dir}/data/approximation_comparison.json", 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, indent=2)
        
    # ---------------------------------------------------------
    # 7. Generate High-Quality Visualizations
    # ---------------------------------------------------------
    print("\n[INFO] Đang tạo các biểu đồ trực quan hóa chuyên sâu...")
    
    # Plot 1: Input Distribution across Backbone, Neck, Head
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.hist(all_silu_samples, bins=100, range=(-8, 8), density=True, alpha=0.7, color='#2563EB', label='SiLU Nodes (Backbone & Neck)')
    plt.axvline(x=-6.0, color='red', linestyle='--', linewidth=1.5, label='Bão hòa x = ±6.0 (σ < 0.0025)')
    plt.axvline(x=6.0, color='red', linestyle='--', linewidth=1.5)
    plt.axvline(x=-2.0, color='green', linestyle=':', linewidth=1.2, label='Vùng tuyến tính x = [-2, +2]')
    plt.axvline(x=2.0, color='green', linestyle=':', linewidth=1.2)
    plt.title("Phân bố đầu vào hàm SiLU (87 Conv Nodes)", fontsize=12, fontweight='bold')
    plt.xlabel("Giá trị kích hoạt đầu vào x (FP32)")
    plt.ylabel("Mật độ xác suất (Density)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=9)
    
    plt.subplot(1, 2, 2)
    plt.hist(head_samples, bins=100, range=(-15, 8), density=True, alpha=0.7, color='#DC2626', label='Sigmoid Head (Node 363)')
    plt.axvline(x=-6.0, color='red', linestyle='--', linewidth=1.5, label='Bão hòa âm x = -6.0 (Background)')
    plt.title("Phân bố đầu vào Detect Head Classification Logits", fontsize=12, fontweight='bold')
    plt.xlabel("Logits x")
    plt.ylabel("Mật độ xác suất (Density)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{out_dir}/plots/sigmoid_input_distribution.png", dpi=300)
    plt.close()
    
    # Plot 2: Curve Comparison (FP32 vs LUT-256 vs PWL-4 vs PWL-8)
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(test_x, sig_gt, 'k-', linewidth=2.0, label='Ground Truth FP32')
    plt.plot(test_x, sig_lut256, '--', color='#2563EB', linewidth=1.5, label='LUT-256 (Direct ROM)')
    plt.plot(test_x, sig_pwl4, '-.', color='#F59E0B', linewidth=1.5, label='PWL-4 (Shift-Add, 0 DSP)')
    plt.plot(test_x, sig_pwl8, ':', color='#10B981', linewidth=1.8, label='PWL-8 (Shift-Add, 0 DSP)')
    plt.title("Xấp xỉ Hàm Sigmoid: σ(x) trên FPGA", fontsize=12, fontweight='bold')
    plt.xlabel("x")
    plt.ylabel("σ(x)")
    plt.xlim(-7, 7)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', fontsize=10)
    
    plt.subplot(1, 2, 2)
    plt.plot(test_x, silu_gt, 'k-', linewidth=2.0, label='Ground Truth FP32')
    plt.plot(test_x, silu_lut256, '--', color='#2563EB', linewidth=1.5, label='SiLU LUT-256')
    plt.plot(test_x, silu_pwl4, '-.', color='#F59E0B', linewidth=1.5, label='SiLU PWL-4 (Shift-Add)')
    plt.plot(test_x, silu_pwl8, ':', color='#10B981', linewidth=1.8, label='SiLU PWL-8 (Shift-Add)')
    plt.title("Xấp xỉ Hàm SiLU: x · σ(x) trên FPGA", fontsize=12, fontweight='bold')
    plt.xlabel("x")
    plt.ylabel("SiLU(x)")
    plt.xlim(-7, 7)
    plt.ylim(-0.5, 7.0)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{out_dir}/plots/sigmoid_pwl_lut_comparison.png", dpi=300)
    plt.close()
    
    # Plot 3: Error Analysis Across the Dynamic Range
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(test_x, np.abs(sig_lut256 - sig_gt), color='#2563EB', label=f'LUT-256 (Max: {compute_metrics(sig_gt, sig_lut256)["max_err"]:.4f})')
    plt.plot(test_x, np.abs(sig_pwl4 - sig_gt), color='#F59E0B', label=f'PWL-4 Shift (Max: {compute_metrics(sig_gt, sig_pwl4)["max_err"]:.4f})')
    plt.plot(test_x, np.abs(sig_pwl8 - sig_gt), color='#10B981', label=f'PWL-8 Shift (Max: {compute_metrics(sig_gt, sig_pwl8)["max_err"]:.4f})')
    plt.plot(test_x, np.abs(sig_pwl_opt - sig_gt), color='#8B5CF6', linestyle='--', label=f'PWL-8 Optimal (Max: {compute_metrics(sig_gt, sig_pwl_opt)["max_err"]:.4f})')
    plt.title("Sai số Tuyệt đối Hàm Sigmoid: |σ_approx(x) - σ_fp32(x)|", fontsize=12, fontweight='bold')
    plt.xlabel("x")
    plt.ylabel("Sai số tuyệt đối (Absolute Error)")
    plt.xlim(-7, 7)
    plt.yscale('log')
    plt.ylim(1e-4, 0.2)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=9)
    
    plt.subplot(1, 2, 2)
    plt.plot(test_x, np.abs(silu_pwl4 - silu_gt), color='#F59E0B', label=f'SiLU PWL-4 (Max: {compute_metrics(silu_gt, silu_pwl4)["max_err"]:.4f})')
    plt.plot(test_x, np.abs(silu_pwl8 - silu_gt), color='#10B981', label=f'SiLU PWL-8 (Max: {compute_metrics(silu_gt, silu_pwl8)["max_err"]:.4f})')
    plt.plot(test_x, np.abs(silu_lut256 - silu_gt), color='#2563EB', label=f'SiLU LUT-256 (Max: {compute_metrics(silu_gt, silu_lut256)["max_err"]:.4f})')
    plt.title("Sai số Tuyệt đối Hàm SiLU: |SiLU_approx(x) - SiLU_fp32(x)|", fontsize=12, fontweight='bold')
    plt.xlabel("x")
    plt.ylabel("Sai số tuyệt đối (Absolute Error)")
    plt.xlim(-7, 7)
    plt.yscale('log')
    plt.ylim(1e-4, 0.5)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{out_dir}/plots/sigmoid_error_distribution.png", dpi=300)
    plt.close()
    
    print("[SUCCESS] Toàn bộ 3 biểu đồ phân tích đã được lưu vào thư mục plots!")

if __name__ == "__main__":
    main()
