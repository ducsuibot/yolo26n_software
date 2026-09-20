#!/usr/bin/env python3
"""
Weight Density Spectrum and Distribution Visualization
Compares FP32 Baseline vs Dequantized INT8 for representative layers:
- Node 0 Conv (P1 - Input layer: 16 filters 3x3x3 = 432 weights)
- Node 21 Conv (P3 - Deep backbone layer: 64 filters 64x3x3 = 36,864 weights)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Cài đặt style đồ thị chuyên nghiệp
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def load_weight_data(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.extend([float(x) for x in line.strip().split() if x])
    return np.array(data, dtype=np.float32)

def plot_layer_density(layer_name, node_id, fp32_path, quant_rec, output_png):
    w_fp32 = load_weight_data(fp32_path)
    
    scale_sym = quant_rec["symmetric"]["scale"]
    z_sym = quant_rec["symmetric"]["zero_point"]
    
    # INT8 discrete values
    w_int8 = np.clip(np.round(w_fp32 / scale_sym), -128, 127).astype(np.int8)
    w_dequant = (w_int8.astype(np.float32) * scale_sym).astype(np.float32)
    error = w_fp32 - w_dequant
    
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    # ----------------------------------------------------
    # Subplot 1: Phổ mật độ (PDF / KDE) so sánh FP32 vs INT8 Dequantized
    # ----------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    bins = min(60, len(np.unique(w_int8)))
    
    ax1.hist(w_fp32, bins=bins, density=True, alpha=0.55, color='#1f77b4', 
             edgecolor='#0b559f', label=f'FP32 Baseline (N={len(w_fp32)})')
    ax1.hist(w_dequant, bins=bins, density=True, alpha=0.45, color='#ff7f0e', 
             edgecolor='#d95f02', linestyle='--', label='INT8 Dequantized (Symmetric)')
    
    ax1.set_title(f'1. Phổ Mật Độ Trọng Số (Weight Density Spectrum)\n{layer_name}', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Giá trị trọng số (Weight Value)', fontsize=10)
    ax1.set_ylabel('Mật độ xác suất (Probability Density)', fontsize=10)
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # ----------------------------------------------------
    # Subplot 2: Phân bố số nguyên rời rạc INT8 [-128, 127]
    # ----------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    int_bins = np.arange(np.min(w_int8) - 0.5, np.max(w_int8) + 1.5, 1)
    
    ax2.hist(w_int8, bins=int_bins, color='#2ca02c', edgecolor='#1b611b', alpha=0.75, rwidth=0.85)
    ax2.axvline(0, color='red', linestyle='--', linewidth=1.2, label='Zero-Point (z = 0)')
    
    ax2.set_title(f'2. Phân Bố Số Nguyên Rời Rạc INT8 Hardware\nMin INT8: {np.min(w_int8)} | Max INT8: {np.max(w_int8)}', 
                  fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Giá trị nguyên INT8 [q_min = -128, q_max = 127]', fontsize=10)
    ax2.set_ylabel('Tần số xuất hiện (Count)', fontsize=10)
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # ----------------------------------------------------
    # Subplot 3: Phổ phân bố sai số lượng tử hóa e = W_fp32 - W_dequant
    # ----------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    max_bound = scale_sym / 2.0
    
    ax3.hist(error, bins=50, color='#9467bd', edgecolor='#5c3566', alpha=0.7, density=True)
    ax3.axvline(-max_bound, color='red', linestyle='--', linewidth=1.5, label=f'Bound -s/2 ({-max_bound:.6f})')
    ax3.axvline(max_bound, color='red', linestyle='--', linewidth=1.5, label=f'Bound +s/2 (+{max_bound:.6f})')
    ax3.axvline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    ax3.set_title('3. Phổ Phân Bố Sai Số Lượng Tử Hóa (Quantization Error: e = W - Ŵ)\nChứng minh sai số bị chặn chặt chẽ bởi [-s/2, +s/2]', 
                  fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel('Sai số định lượng (Error)', fontsize=10)
    ax3.set_ylabel('Mật độ (Density)', fontsize=10)
    ax3.legend(loc='upper right', frameon=True)
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    # ----------------------------------------------------
    # Subplot 4: Bảng tổng hợp tham số phần cứng và chỉ số định lượng
    # ----------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    stats_data = [
        ["Thông Số / Chỉ Số", "Giá Trị Định Lượng", "Đơn Vị / Ý Nghĩa"],
        ["Tầng mạng (Layer)", f"Node {node_id} (Conv)", layer_name],
        ["Tổng số tham số", f"{len(w_fp32):,}", "weights"],
        ["FP32 Min / Max", f"[{np.min(w_fp32):.6f}, {np.max(w_fp32):.6f}]", "Dynamic range"],
        ["Scale s (Symmetric)", f"{scale_sym:.8e}", "Hệ số tỷ lệ lượng tử hóa"],
        ["Zero-point z", f"{z_sym}", "Gốc 0 (Symmetric z=0)"],
        ["INT8 Range (q_min / q_max)", f"[{np.min(w_int8)}, {np.max(w_int8)}]", "Phạm vi số nguyên"],
        ["Sai số cực đại (Max Err)", f"{np.max(np.abs(error)):.6f}", f"Lý thuyết ≤ {max_bound:.6f}"],
        ["Sai số trung bình (MAE)", f"{np.mean(np.abs(error)):.6f}", "Mean Absolute Error"],
        ["Sai số bình phương (MSE)", f"{np.mean(error**2):.8e}", "Mean Squared Error"],
        ["Tỷ số SQNR", f"{quant_rec['symmetric']['sqnr_db']:.2f} dB", "Signal-to-Quant-Noise Ratio"],
        ["Độ tương đồng Cosine", f"{quant_rec['symmetric']['cosine_sim']:.7f}", "Cosine Similarity (Max=1.0)"],
        ["Địa chỉ DRAM_1", f"{quant_rec.get('dram_start', 'N/A')} - {quant_rec.get('dram_end', 'N/A')}", "Ánh xạ bộ nhớ FPGA"]
    ]
    
    table = ax4.table(cellText=stats_data, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.05, 1.4)
    
    # Định dạng header bảng
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#1f77b4')
        elif row % 2 == 1:
            cell.set_facecolor('#f7f9fa')
        else:
            cell.set_facecolor('#ffffff')
            
    ax4.set_title('4. Bảng Tổng Hợp Tham Số Lượng Tử Hóa & Hardware Map', 
                  fontsize=12, fontweight='bold', pad=15)
    
    plt.suptitle(f"KHẢO SÁT PHỔ MẬT ĐỘ TRỌNG SỐ TRƯỚC VÀ SAU QUANTIZE INT8\n{layer_name} - YOLOv26n", 
                 fontsize=15, fontweight='bold', y=0.98)
    
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Đã lưu biểu đồ: {output_png}")

def main():
    print("=" * 60)
    print("  VẼ PHỔ MẬT ĐỘ TRỌNG SỐ (WEIGHT DENSITY) TRƯỚC & SAU QUANTIZE")
    print("=" * 60)
    
    with open("quantization_params.json", 'r', encoding='utf-8') as f:
        quant_records = json.load(f)
        
    rec_by_node_type = {}
    for r in quant_records:
        key = f"node_{r['node_id']}_{r['param_type']}"
        rec_by_node_type[key] = r
        
    # 1. Lớp Node 0 Conv: Đầu vào Backbone P1
    plot_layer_density(
        layer_name="Module 0: Conv (P1) - Lớp Đầu Vào Backbone (3x3x3)",
        node_id=0,
        fp32_path="weight/backbone/module_0_conv_P1/node_0_weight.txt",
        quant_rec=rec_by_node_type["node_0_weight"],
        output_png="weight_density_node_0.png"
    )
    
    # 2. Lớp Node 21 Conv: Lớp sâu trong Backbone P3
    plot_layer_density(
        layer_name="Module 3: Conv (P3) - Lớp Sâu Backbone (64x64x3x3)",
        node_id=21,
        fp32_path="weight/backbone/module_3_conv_P3/node_21_weight.txt",
        quant_rec=rec_by_node_type["node_21_weight"],
        output_png="weight_density_node_21.png"
    )
    
    # 3. Vẽ biểu đồ so sánh trực quan cả 2 lớp trên cùng 1 hình tổng quan
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Node 0
    w0_fp32 = load_weight_data("weight/backbone/module_0_conv_P1/node_0_weight.txt")
    s0 = rec_by_node_type["node_0_weight"]["symmetric"]["scale"]
    w0_int8 = np.clip(np.round(w0_fp32 / s0), -128, 127)
    w0_dq = w0_int8 * s0
    
    axes[0].hist(w0_fp32, bins=40, density=True, alpha=0.55, color='#1f77b4', label='FP32 Original')
    axes[0].hist(w0_dq, bins=40, density=True, alpha=0.45, color='#ff7f0e', linestyle='--', label='INT8 Dequantized')
    axes[0].set_title(f"Node 0 Conv (Input P1)\nShape: 16x3x3x3 | SQNR: {rec_by_node_type['node_0_weight']['symmetric']['sqnr_db']:.2f} dB", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Weight Value", fontsize=10)
    axes[0].set_ylabel("Probability Density", fontsize=10)
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # Node 21
    w21_fp32 = load_weight_data("weight/backbone/module_3_conv_P3/node_21_weight.txt")
    s21 = rec_by_node_type["node_21_weight"]["symmetric"]["scale"]
    w21_int8 = np.clip(np.round(w21_fp32 / s21), -128, 127)
    w21_dq = w21_int8 * s21
    
    axes[1].hist(w21_fp32, bins=60, density=True, alpha=0.55, color='#2ca02c', label='FP32 Original')
    axes[1].hist(w21_dq, bins=60, density=True, alpha=0.45, color='#d62728', linestyle='--', label='INT8 Dequantized')
    axes[1].set_title(f"Node 21 Conv (Deep P3)\nShape: 64x64x3x3 | SQNR: {rec_by_node_type['node_21_weight']['symmetric']['sqnr_db']:.2f} dB", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Weight Value", fontsize=10)
    axes[1].set_ylabel("Probability Density", fontsize=10)
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.suptitle("SO SÁNH PHỔ MẬT ĐỘ TRỌNG SỐ: NODE 0 (INPUT) VS NODE 21 (DEEP P3)", fontsize=13, fontweight='bold')
    plt.savefig("weight_density_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("[SUCCESS] Đã lưu biểu đồ so sánh: weight_density_comparison.png")
    print("=" * 60)

if __name__ == "__main__":
    main()
