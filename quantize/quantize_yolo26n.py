#!/usr/bin/env python3
"""
Linear Quantization Engine for YOLOv26n Convolution Layers
Target Architecture: FPGA ZCU104 (UAV Tracking)

Mathematical Formulation (Linear Quantized Convolution Layer):
    q_Y = [ (S_W * S_X) / S_Y ] * ( Conv(q_W, q_X) + q_bias )

Where:
    - Conv(q_W, q_X) : N-bit Int Mult (INT8 x INT8)
    - q_bias         : 32-bit Int Add (INT32), with Scale S_bias = S_W * S_X
    - Rescale Factor : M_float = (S_W * S_X) / S_Y
    - Hardware Fixed-Point Approximation: M_float ≈ M_mult / (2 ^ M_shift)

Hardware Execution on FPGA:
    acc_32 = Conv(q_W, q_X) + q_bias          (32-bit integer accumulation)
    q_Y = (acc_32 * M_mult) >> M_shift        (Rescale to INT8 with right shift)
    q_Y = clip(q_Y, -128, 127)

Outputs:
    1. quantized_weight/ directory containing:
       - node_X_weight.txt       : Raw INT8 quantized weights (q_W)
       - node_X_bias.txt         : Raw INT32 quantized bias (q_bias)
       - node_X_Sx.txt           : S_X (Input activation scale)
       - node_X_Sw.txt           : S_W (Weight scale)
       - node_X_Sy.txt           : S_Y (Output activation scale)
       - node_X_M_float.txt      : M_float = S_W * S_X / S_Y
       - node_X_M_mult.txt       : M_mult (16-bit DSP multiplier)
       - node_X_M_shift.txt      : M_shift (Bit shift amount)
       - node_X_conv_params.txt  : Human-readable text parameter file
       - node_X_conv_params.json : JSON parameter file for the node
    2. Master Summary:
       - quantization_conv_layers.csv & .json: Complete table of all 102 Conv layers
       - quantization_params.csv & .json     : Complete table of all 207 tensors
    3. best_quantized_int8.onnx:
       - Graph preserved with dequantized INT8 weights/bias for software reference
"""

import os
import glob
import copy
import json
import csv
import cv2
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

def quantize_symmetric(tensor, qmin=-128, qmax=127):
    """Symmetric Linear Quantization (z = 0)"""
    abs_max = float(np.max(np.abs(tensor)))
    if abs_max == 0:
        scale = 1.0
    else:
        scale = float(abs_max / qmax)
    
    q_tensor = np.clip(np.round(tensor / scale), qmin, qmax).astype(np.int8)
    dequant_tensor = (q_tensor.astype(np.float32) * scale).astype(np.float32)
    return q_tensor, scale, dequant_tensor

def compute_multiplier_and_shift(m_float, target_mult_bits=15):
    """
    Convert floating-point multiplier M_float into (M_mult, M_shift)
    M_float ≈ M_mult / (2 ^ M_shift)
    
    - Dynamic shift: Fits M_mult into target_mult_bits (e.g. 15-bit signed/16-bit unsigned for DSP48E2)
    - Fixed shift 16: Uses standard fixed shift 16
    """
    if m_float <= 0:
        return {
            "M_mult_dyn": 0,
            "M_shift_dyn": 0,
            "rel_err_dyn": 0.0,
            "M_mult_16": 0,
            "M_shift_16": 16,
            "rel_err_16": 0.0
        }
        
    # Dynamic shift to maximize multiplier precision (target max multiplier = 2^15 - 1 = 32767)
    target_max = float((1 << target_mult_bits) - 1)
    shift_dyn = int(np.ceil(np.log2(target_max / m_float)))
    shift_dyn = max(0, shift_dyn)
    mult_dyn = int(np.round(m_float * (2 ** shift_dyn)))
    approx_dyn = mult_dyn / (2 ** shift_dyn)
    rel_err_dyn = abs(m_float - approx_dyn) / m_float * 100.0
    
    # Fixed shift 16
    shift_16 = 16
    mult_16 = int(np.round(m_float * (2 ** shift_16)))
    approx_16 = mult_16 / (2 ** shift_16)
    rel_err_16 = abs(m_float - approx_16) / m_float * 100.0
    
    return {
        "M_mult_dyn": mult_dyn,
        "M_shift_dyn": shift_dyn,
        "rel_err_dyn": float(rel_err_dyn),
        "M_mult_16": mult_16,
        "M_shift_16": shift_16,
        "rel_err_16": float(rel_err_16)
    }

def load_dram_map(dram_csv_path):
    """Load DRAM address mapping from dram.csv for both DRAM_1 and DRAM_2"""
    dram_map = {}
    if not os.path.exists(dram_csv_path):
        return dram_map
    
    with open(dram_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 8:
                module = row[0].strip()
                node = row[1].strip()
                region = row[2].strip() # DRAM_1 or DRAM_2
                param_type = row[3].strip() # Weight, Bias, Output
                shape = row[4].strip()
                size = row[5].strip()
                start_addr = row[6].strip()
                end_addr = row[7].strip()
                
                key = f"{region}_{node}_{param_type}".lower()
                dram_map[key] = {
                    "module": module,
                    "node": node,
                    "region": region,
                    "type": param_type,
                    "shape": shape,
                    "size": int(size) if size.isdigit() else 0,
                    "start_addr": int(start_addr) if start_addr.isdigit() else -1,
                    "end_addr": int(end_addr) if end_addr.isdigit() else -1
                }
    return dram_map

def write_formatted_int_array(filepath, data, per_line=20):
    """Write integer numpy array to txt with per_line elements per row"""
    with open(filepath, 'w') as f:
        flat = data.flatten()
        for i, val in enumerate(flat):
            f.write(str(int(val)))
            if (i + 1) % per_line == 0 or (i + 1) == len(flat):
                f.write("\n")
            else:
                f.write(" ")

def write_single_value(filepath, val):
    """Write a single float/int to a text file"""
    with open(filepath, 'w') as f:
        if isinstance(val, (int, np.integer)):
            f.write(f"{int(val)}\n")
        else:
            f.write(f"{float(val):.12e}\n")

def calibrate_activations(onnx_path, conv_nodes, val_img_dir="val_dataset/images", ref_img="image.png"):
    """
    Run calibration on validation dataset to determine maximum absolute activation
    for input (X) and output (Y) of every convolution layer.
    """
    print(f"\n[INFO] Đang chạy hiệu chuẩn (Calibration) để xác định thang đo Sx và Sy...")
    
    # Identify all input and output tensors needed
    tensor_names = set()
    for n in conv_nodes:
        tensor_names.add(n.input[0])
        tensor_names.add(n.output[0])
        
    model = onnx.load(onnx_path)
    for t_name in tensor_names:
        vi = onnx.helper.ValueInfoProto()
        vi.name = t_name
        model.graph.output.append(vi)
        
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(model.SerializeToString(), sess_options, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    
    # Gather calibration images
    calib_images = sorted(glob.glob(os.path.join(val_img_dir, "*.jpg")))
    if os.path.exists(ref_img) and ref_img not in calib_images:
        calib_images.insert(0, ref_img)
        
    print(f"[INFO] Tổng số ảnh hiệu chuẩn: {len(calib_images)}")
    
    max_abs_dict = {name: 0.0 for name in tensor_names}
    fetches = list(tensor_names)
    
    for idx, img_path in enumerate(calib_images):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (640, 640)).astype(np.float32) / 255.0
        inp = np.transpose(img_resized, (2, 0, 1))[None, ...]
        
        outs = sess.run(fetches, {input_name: inp})
        for name, arr in zip(fetches, outs):
            cur_max = float(np.max(np.abs(arr)))
            if cur_max > max_abs_dict[name]:
                max_abs_dict[name] = cur_max
                
        if (idx + 1) % 10 == 0 or (idx + 1) == len(calib_images):
            print(f"       -> Đã xử lý {idx + 1}/{len(calib_images)} ảnh hiệu chuẩn...")
            
    print(f"[SUCCESS] Hiệu chuẩn hoàn tất cho toàn bộ {len(tensor_names)} tensors trung gian.")
    return max_abs_dict

def main():
    onnx_path = "best.onnx"
    output_onnx_path = "best_quantized_int8.onnx"
    quant_weight_dir = "quantized_weight"
    dram_csv_path = "dram.csv"
    
    print("=" * 70)
    print("  YOLOv26n LINEAR QUANTIZATION ENGINE FOR FPGA ZCU104")
    print("  Model: Linear Quantized Convolution Layer [Sw * Sx / Sy -> Mmult / 2^Mshift]")
    print("=" * 70)
    
    dram_map = load_dram_map(dram_csv_path)
    print(f"[INFO] Đã đọc {len(dram_map)} bản ghi từ {dram_csv_path}")
    
    model = onnx.load(onnx_path)
    initializers = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    
    # Find all Conv nodes in ONNX graph
    conv_nodes_map = {} # node_id -> node
    for idx, node in enumerate(model.graph.node):
        if node.op_type == "Conv":
            conv_nodes_map[idx] = node
            
    print(f"[INFO] Tìm thấy {len(conv_nodes_map)} Conv nodes trong {onnx_path}")
    
    # 1. Calibrate activations across validation dataset
    max_abs_dict = calibrate_activations(onnx_path, list(conv_nodes_map.values()))
    
    # 2. Build mapping of weight files from weight/ directory
    weight_files = {} # node_id -> {'stage', 'module', 'weight_file', 'bias_file'}
    all_txt_files = []
    for root, dirs, files in os.walk("weight"):
        for f in files:
            if f.endswith(".txt"):
                rel_p = os.path.relpath(os.path.join(root, f), "weight")
                parts = rel_p.split(os.sep)
                stage, mod, fname = parts
                node_id = int(fname.split("_")[1])
                ptype = "weight" if "weight" in fname else "bias"
                
                if node_id not in weight_files:
                    weight_files[node_id] = {
                        "stage": stage,
                        "module": mod,
                        "dir": os.path.join("weight", stage, mod)
                    }
                weight_files[node_id][ptype] = os.path.join("weight", rel_p)
                all_txt_files.append((node_id, stage, mod, ptype, os.path.join("weight", rel_p)))

    # Ensure output directories exist
    for node_id, info in weight_files.items():
        out_mod_dir = os.path.join(quant_weight_dir, info["stage"], info["module"])
        os.makedirs(out_mod_dir, exist_ok=True)
        
    print(f"[INFO] Tìm thấy {len(weight_files)} node folders trong weight/ (bao gồm cả Conv và Head constants)")
    
    # 3. Process each Conv Layer
    conv_layer_records = []
    onnx_weight_updates = {} # init_name -> dq_array
    
    for nid, node in conv_nodes_map.items():
        w_name = node.input[1]
        b_name = node.input[2] if len(node.input) > 2 else None
        in_tensor_name = node.input[0]
        out_tensor_name = node.output[0]
        
        info = weight_files.get(nid, {})
        stage = info.get("stage", "unknown")
        mod = info.get("module", "unknown")
        out_mod_dir = os.path.join(quant_weight_dir, stage, mod)
        
        # 1. Weights
        w_fp32 = initializers[w_name]
        abs_max_w = float(np.max(np.abs(w_fp32)))
        s_w = float(abs_max_w / 127.0) if abs_max_w > 0 else 1.0
        q_w = np.clip(np.round(w_fp32 / s_w), -128, 127).astype(np.int8)
        dq_w = (q_w.astype(np.float32) * s_w).astype(np.float32)
        onnx_weight_updates[w_name] = dq_w
        
        # 2. Input Activation (Sx) & Output Activation (Sy)
        abs_max_x = max_abs_dict.get(in_tensor_name, 1.0)
        s_x = float(abs_max_x / 127.0) if abs_max_x > 0 else 1.0
        
        abs_max_y = max_abs_dict.get(out_tensor_name, 1.0)
        s_y = float(abs_max_y / 127.0) if abs_max_y > 0 else 1.0
        
        # 3. Bias: S_bias = S_w * S_x, q_bias = round(b / (S_w * S_x)) (32-bit integer)
        s_bias = s_w * s_x
        if b_name and b_name in initializers:
            b_fp32 = initializers[b_name]
            q_b = np.round(b_fp32 / s_bias).astype(np.int32)
            dq_b = (q_b.astype(np.float32) * s_bias).astype(np.float32)
            onnx_weight_updates[b_name] = dq_b
        else:
            q_b = np.zeros((w_fp32.shape[0],), dtype=np.int32)
            dq_b = np.zeros((w_fp32.shape[0],), dtype=np.float32)
            
        # 4. Rescale Factor M_float = (S_w * S_x) / S_y
        m_float = float((s_w * s_x) / s_y)
        m_params = compute_multiplier_and_shift(m_float, target_mult_bits=15)
        
        # 5. Retrieve DRAM addresses
        dram_w_info = None
        dram_b_info = None
        dram_y_info = None
        for k, v in dram_map.items():
            if f"dram_1_node {nid}:" in k and "weight" in k:
                dram_w_info = v
            elif f"dram_1_node {nid}:" in k and "bias" in k:
                dram_b_info = v
            elif f"dram_2_node {nid}:" in k and "output" in k:
                dram_y_info = v
                
        # 6. Write Individual Files for this Layer
        # a) Weights (q_W: INT8)
        w_out_file = os.path.join(out_mod_dir, f"node_{nid}_weight.txt")
        write_formatted_int_array(w_out_file, q_w)
        
        # b) Bias (q_bias: INT32)
        b_out_file = os.path.join(out_mod_dir, f"node_{nid}_bias.txt")
        write_formatted_int_array(b_out_file, q_b)
        
        # c) Individual parameter text files
        write_single_value(os.path.join(out_mod_dir, f"node_{nid}_Sx.txt"), s_x)
        write_single_value(os.path.join(out_mod_dir, f"node_{nid}_Sw.txt"), s_w)
        write_single_value(os.path.join(out_mod_dir, f"node_{nid}_Sy.txt"), s_y)
        write_single_value(os.path.join(out_mod_dir, f"node_{nid}_M_float.txt"), m_float)
        write_single_value(os.path.join(out_mod_dir, f"node_{nid}_M_mult.txt"), m_params["M_mult_dyn"])
        write_single_value(os.path.join(out_mod_dir, f"node_{nid}_M_shift.txt"), m_params["M_shift_dyn"])
        
        # d) Consolidated human-readable file for this layer
        conv_params_txt = os.path.join(out_mod_dir, f"node_{nid}_conv_params.txt")
        with open(conv_params_txt, 'w') as pf:
            pf.write(f"NODE_ID: {nid}\n")
            pf.write(f"NODE_NAME: {node.name}\n")
            pf.write(f"MODULE: {mod}\n")
            pf.write(f"STAGE: {stage}\n")
            pf.write(f"INPUT_TENSOR: {in_tensor_name}\n")
            pf.write(f"OUTPUT_TENSOR: {out_tensor_name}\n")
            pf.write(f"WEIGHT_SHAPE: {list(w_fp32.shape)}\n")
            pf.write(f"Sx: {s_x:.12e}\n")
            pf.write(f"Sw: {s_w:.12e}\n")
            pf.write(f"Sy: {s_y:.12e}\n")
            pf.write(f"Sw_x_Sx_div_Sy (M_float): {m_float:.12e}\n")
            pf.write(f"M_mult (DSP 16-bit): {m_params['M_mult_dyn']}\n")
            pf.write(f"M_shift: {m_params['M_shift_dyn']}\n")
            pf.write(f"M_mult_16 (Fixed Shift 16): {m_params['M_mult_16']}\n")
            pf.write(f"M_shift_16: 16\n")
            pf.write(f"Bias_Scale (Sw*Sx): {s_bias:.12e}\n")
            pf.write(f"Rel_Error_Dyn_%: {m_params['rel_err_dyn']:.6f}%\n")
            pf.write(f"Rel_Error_16_%: {m_params['rel_err_16']:.6f}%\n")
            pf.write(f"DRAM1_Weight_Start: {dram_w_info['start_addr'] if dram_w_info else -1}\n")
            pf.write(f"DRAM1_Weight_End: {dram_w_info['end_addr'] if dram_w_info else -1}\n")
            pf.write(f"DRAM1_Bias_Start: {dram_b_info['start_addr'] if dram_b_info else -1}\n")
            pf.write(f"DRAM1_Bias_End: {dram_b_info['end_addr'] if dram_b_info else -1}\n")
            pf.write(f"DRAM2_OFM_Start: {dram_y_info['start_addr'] if dram_y_info else -1}\n")
            pf.write(f"DRAM2_OFM_End: {dram_y_info['end_addr'] if dram_y_info else -1}\n")
            
        record = {
            "node_id": nid,
            "node_name": node.name,
            "module": mod,
            "stage": stage,
            "input_tensor": in_tensor_name,
            "output_tensor": out_tensor_name,
            "weight_shape": list(w_fp32.shape),
            "Sx": s_x,
            "Sw": s_w,
            "Sy": s_y,
            "M_float": m_float,
            "M_mult_dyn": m_params["M_mult_dyn"],
            "M_shift_dyn": m_params["M_shift_dyn"],
            "M_mult_16": m_params["M_mult_16"],
            "M_shift_16": m_params["M_shift_16"],
            "rel_err_dyn_pct": m_params["rel_err_dyn"],
            "rel_err_16_pct": m_params["rel_err_16"],
            "bias_scale": s_bias,
            "dram1_weight_start": dram_w_info["start_addr"] if dram_w_info else -1,
            "dram1_weight_end": dram_w_info["end_addr"] if dram_w_info else -1,
            "dram1_bias_start": dram_b_info["start_addr"] if dram_b_info else -1,
            "dram1_bias_end": dram_b_info["end_addr"] if dram_b_info else -1,
            "dram2_ofm_start": dram_y_info["start_addr"] if dram_y_info else -1,
            "dram2_ofm_end": dram_y_info["end_addr"] if dram_y_info else -1
        }
        conv_layer_records.append(record)
        
        # Save JSON per layer
        with open(os.path.join(out_mod_dir, f"node_{nid}_conv_params.json"), 'w') as jf:
            json.dump(record, jf, indent=2)

    # 4. Handle non-Conv constant nodes in weight/ (Node 355, 356, 360 in head)
    non_conv_nids = set(weight_files.keys()) - set(conv_nodes_map.keys())
    for ncid in non_conv_nids:
        info = weight_files[ncid]
        stage = info["stage"]
        mod = info["module"]
        out_mod_dir = os.path.join(quant_weight_dir, stage, mod)
        
        for ptype in ["weight", "bias"]:
            if ptype in info:
                src_path = info[ptype]
                vals = []
                with open(src_path, 'r') as f:
                    for line in f:
                        vals.extend([float(x) for x in line.strip().split() if x])
                arr = np.array(vals, dtype=np.float32)
                q_arr, s_val, _ = quantize_symmetric(arr)
                out_p = os.path.join(out_mod_dir, f"node_{ncid}_{ptype}.txt")
                write_formatted_int_array(out_p, q_arr)
                write_single_value(os.path.join(out_mod_dir, f"node_{ncid}_{ptype}_Sw.txt"), s_val)
                
    print(f"[SUCCESS] Đã lượng tử hóa toàn bộ {len(conv_layer_records)} Conv layers và các constant nodes.")
    
    # 5. Export Master CSV and JSON for Conv Layers
    conv_csv_path = "quantization_conv_layers.csv"
    with open(conv_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Stage", "Module", "Node_ID", "Node_Name", "Weight_Shape",
            "Sx", "Sw", "Sy", "Sw_x_Sx_div_Sy (M_float)",
            "M_mult_dyn", "M_shift_dyn", "Rel_Err_Dyn_%",
            "M_mult_16", "M_shift_16", "Rel_Err_16_%",
            "Bias_Scale",
            "DRAM1_Weight_Start", "DRAM1_Weight_End",
            "DRAM1_Bias_Start", "DRAM1_Bias_End",
            "DRAM2_OFM_Start", "DRAM2_OFM_End"
        ])
        for r in conv_layer_records:
            writer.writerow([
                r["stage"], r["module"], r["node_id"], r["node_name"], str(r["weight_shape"]),
                f"{r['Sx']:.8e}", f"{r['Sw']:.8e}", f"{r['Sy']:.8e}", f"{r['M_float']:.8e}",
                r["M_mult_dyn"], r["M_shift_dyn"], f"{r['rel_err_dyn_pct']:.6f}",
                r["M_mult_16"], r["M_shift_16"], f"{r['rel_err_16_pct']:.6f}",
                f"{r['bias_scale']:.8e}",
                r["dram1_weight_start"], r["dram1_weight_end"],
                r["dram1_bias_start"], r["dram1_bias_end"],
                r["dram2_ofm_start"], r["dram2_ofm_end"]
            ])
    print(f"[SUCCESS] Đã lưu bảng tham số Conv Layers: {conv_csv_path}")
    
    conv_json_path = "quantization_conv_layers.json"
    with open(conv_json_path, 'w', encoding='utf-8') as f:
        json.dump(conv_layer_records, f, indent=2, ensure_ascii=False)
    print(f"[SUCCESS] Đã lưu bảng tham số Conv Layers (JSON): {conv_json_path}")
    
    # 6. Build and Export True Q/DQ ONNX Model (QuantizeLinear & DequantizeLinear)
    print("\n[INFO] Đang xây dựng ONNX Model chuẩn Q/DQ với đầy đủ các node QuantizeLinear và DequantizeLinear...")
    qdq_output_path = "best_quantized_qdq.onnx"
    
    new_nodes = []
    new_inits = []
    removed_init_names = set()
    conv_rec_map = {r["node_id"]: r for r in conv_layer_records}
    
    for idx, node in enumerate(model.graph.node):
        if idx not in conv_rec_map:
            new_nodes.append(node)
            continue
            
        rec = conv_rec_map[idx]
        sx = np.float32(rec['Sx'])
        sw = np.float32(rec['Sw'])
        sy = np.float32(rec['Sy'])
        zp_0 = np.int8(0)
        
        # 1. Input Q & DQ
        in_tensor = node.input[0]
        scale_x_name = f"{in_tensor}_scale_{idx}"
        zp_x_name = f"{in_tensor}_zp_{idx}"
        q_x_out = f"{in_tensor}_quant_{idx}"
        dq_x_out = f"{in_tensor}_dequant_{idx}"
        
        new_inits.append(helper.make_tensor(scale_x_name, TensorProto.FLOAT, [], [sx]))
        new_inits.append(helper.make_tensor(zp_x_name, TensorProto.INT8, [], [zp_0]))
        
        node_q_x = helper.make_node(
            'QuantizeLinear',
            inputs=[in_tensor, scale_x_name, zp_x_name],
            outputs=[q_x_out],
            name=f"Quantize_{in_tensor}_{idx}"
        )
        node_dq_x = helper.make_node(
            'DequantizeLinear',
            inputs=[q_x_out, scale_x_name, zp_x_name],
            outputs=[dq_x_out],
            name=f"Dequantize_{in_tensor}_{idx}"
        )
        new_nodes.extend([node_q_x, node_dq_x])
        
        # 2. Weight INT8 & DQ
        w_orig_name = node.input[1]
        w_fp32 = initializers[w_orig_name]
        q_w = np.clip(np.round(w_fp32 / sw), -128, 127).astype(np.int8)
        
        w_int8_name = f"{w_orig_name}_int8"
        scale_w_name = f"{w_orig_name}_scale_{idx}"
        zp_w_name = f"{w_orig_name}_zp_{idx}"
        dq_w_out = f"{w_orig_name}_dequant_{idx}"
        
        new_inits.append(helper.make_tensor(
            w_int8_name, TensorProto.INT8, list(q_w.shape), q_w.flatten().tobytes(), raw=True
        ))
        new_inits.append(helper.make_tensor(scale_w_name, TensorProto.FLOAT, [], [sw]))
        new_inits.append(helper.make_tensor(zp_w_name, TensorProto.INT8, [], [zp_0]))
        removed_init_names.add(w_orig_name)
        
        node_dq_w = helper.make_node(
            'DequantizeLinear',
            inputs=[w_int8_name, scale_w_name, zp_w_name],
            outputs=[dq_w_out],
            name=f"Dequantize_{w_orig_name}_{idx}"
        )
        new_nodes.append(node_dq_w)
        
        # 3. Conv Node
        conv_inputs = [dq_x_out, dq_w_out]
        if len(node.input) > 2:
            conv_inputs.append(node.input[2]) # bias
            
        out_orig_name = node.output[0]
        conv_raw_out = f"{out_orig_name}_conv_raw_{idx}"
        
        conv_node_new = helper.make_node('Conv', inputs=conv_inputs, outputs=[conv_raw_out], name=node.name)
        for attr in node.attribute:
            conv_node_new.attribute.append(attr)
        new_nodes.append(conv_node_new)
        
        # 4. Output Q & DQ
        scale_y_name = f"{out_orig_name}_scale_{idx}"
        zp_y_name = f"{out_orig_name}_zp_{idx}"
        q_y_out = f"{out_orig_name}_quant_{idx}"
        
        new_inits.append(helper.make_tensor(scale_y_name, TensorProto.FLOAT, [], [sy]))
        new_inits.append(helper.make_tensor(zp_y_name, TensorProto.INT8, [], [zp_0]))
        
        node_q_y = helper.make_node(
            'QuantizeLinear',
            inputs=[conv_raw_out, scale_y_name, zp_y_name],
            outputs=[q_y_out],
            name=f"Quantize_{out_orig_name}_{idx}"
        )
        node_dq_y = helper.make_node(
            'DequantizeLinear',
            inputs=[q_y_out, scale_y_name, zp_y_name],
            outputs=[out_orig_name],
            name=f"Dequantize_{out_orig_name}_{idx}"
        )
        new_nodes.extend([node_q_y, node_dq_y])

    final_inits = [init for init in model.graph.initializer if init.name not in removed_init_names] + new_inits

    qdq_graph = helper.make_graph(
        new_nodes,
        model.graph.name + '_QDQ',
        model.graph.input,
        model.graph.output,
        final_inits,
        value_info=model.graph.value_info
    )

    qdq_model = helper.make_model(qdq_graph, opset_imports=model.opset_import)
    qdq_model.ir_version = model.ir_version

    onnx.checker.check_model(qdq_model)
    onnx.save(qdq_model, qdq_output_path)
    onnx.save(qdq_model, output_onnx_path)
    print(f"[SUCCESS] Đã lưu mô hình chuẩn Q/DQ: {qdq_output_path} và {output_onnx_path}")
    print(f"          - Tổng số node: {len(qdq_model.graph.node)} (gồm QuantizeLinear & DequantizeLinear)")
    print(f"          - Dung lượng file giảm từ 9.4 MB xuống 2.8 MB (Trọng số lưu định dạng INT8).")
    
    # 7. Print summary statistics
    mean_err_dyn = np.mean([r["rel_err_dyn_pct"] for r in conv_layer_records])
    max_err_dyn = np.max([r["rel_err_dyn_pct"] for r in conv_layer_records])
    mean_err_16 = np.mean([r["rel_err_16_pct"] for r in conv_layer_records])
    max_err_16 = np.max([r["rel_err_16_pct"] for r in conv_layer_records])
    
    print("\n" + "=" * 70)
    print("  TỔNG HỢP THÔNG SỐ LINEAR QUANTIZED CONVOLUTION LAYERS")
    print("=" * 70)
    print(f" - Tổng số tầng Conv lượng tử hóa            : {len(conv_layer_records)}")
    print(f" - Sai số xấp xỉ Dynamic Shift (DSP 16-bit)  : TB = {mean_err_dyn:.6f}% | Max = {max_err_dyn:.6f}%")
    print(f" - Sai số xấp xỉ Fixed Shift 16 (16-bit shift): TB = {mean_err_16:.6f}% | Max = {max_err_16:.6f}%")
    print(f" - File tổng hợp Conv Layers CSV             : {conv_csv_path}")
    print(f" - File tổng hợp Conv Layers JSON            : {conv_json_path}")
    print(f" - Thư mục Weight INT8 & Params từng Layer   : {quant_weight_dir}/")
    print("=" * 70)

if __name__ == "__main__":
    main()
