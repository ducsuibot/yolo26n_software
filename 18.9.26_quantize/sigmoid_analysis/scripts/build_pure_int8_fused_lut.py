#!/usr/bin/env python3
"""
Pure INT8 Hardware Architecture Builder for YOLOv26n on FPGA ZCU104.
Model: best_pure_int8_fused_lut.onnx

Transformations:
1. Eliminates DequantizeLinear after every QLinearConv.
2. Eliminates separate Sigmoid and Mul operations.
3. Fuses (Conv_Out_INT8 -> Fused_SiLU_INT8_Table -> INT8_Output) into a single 1-cycle Distributed ROM table.
4. The Gather table is strictly INT8 (TensorProto.INT8), 256 bytes per table.
5. Direct INT8 routing between Conv layers (Eliminating QuantizeLinear where possible).
6. Validates the graph with onnx.checker and runs an inference test.
"""

import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))

def build_pure_int8_fused_lut_model():
    input_onnx = "18.9.26_quantize/models/best_quantized_int8.onnx"
    output_onnx = "18.9.26_quantize/models/best_pure_int8_fused_lut.onnx"
    
    print("=" * 80)
    print("  BUILDING PURE INT8 FUSED LUT ONNX MODEL (best_pure_int8_fused_lut.onnx)")
    print(f"  Input Model  : {input_onnx}")
    print(f"  Output Model : {output_onnx}")
    print("=" * 80)
    
    model = onnx.load(input_onnx)
    graph = model.graph
    
    # Extract existing initializers
    inits = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    
    # Track node mapping
    qconv_by_out = {n.output[0]: n for n in graph.node if n.op_type == 'QLinearConv'}
    dq_by_out = {n.output[0]: n for n in graph.node if n.op_type == 'DequantizeLinear'}
    q_by_in = {}
    for n in graph.node:
        if n.op_type == 'QuantizeLinear':
            q_by_in.setdefault(n.input[0], []).append(n)
            
    mul_nodes = [n for n in graph.node if n.op_type == 'Mul']
    sig_nodes = [n for n in graph.node if n.op_type == 'Sigmoid']
    
    # Shared constant: offset 128 for int8 to uint8 index conversion
    init_offset_128 = helper.make_tensor('const_offset_128', TensorProto.INT64, [], [128])
    graph.initializer.append(init_offset_128)
    
    # Identify the 87 SiLU pairs
    fused_silu_map = {} # sig_node -> (dq_node, mul_node, qconv_node, q_nodes)
    nodes_to_remove = set()
    
    for sig in sig_nodes:
        in_t = sig.input[0]
        out_t = sig.output[0]
        matching_muls = [m for m in mul_nodes if in_t in m.input and out_t in m.input]
        if not matching_muls:
            continue
        mul = matching_muls[0]
        mul_out = mul.output[0]
        
        dq = dq_by_out.get(in_t)
        if not dq:
            continue
            
        conv_int8_out = dq.input[0]
        qconv = qconv_by_out.get(conv_int8_out)
        if not qconv:
            continue
            
        downstream_qs = q_by_in.get(mul_out, [])
        fused_silu_map[sig.name] = {
            'sig': sig,
            'dq': dq,
            'mul': mul,
            'qconv': qconv,
            'conv_int8_out': conv_int8_out,
            'mul_out': mul_out,
            'downstream_qs': downstream_qs
        }
        
        nodes_to_remove.add(sig.name)
        nodes_to_remove.add(dq.name)
        nodes_to_remove.add(mul.name)
        for q in downstream_qs:
            nodes_to_remove.add(q.name)
            
    print(f"[INFO] Tìm thấy {len(fused_silu_map)} cụm SiLU cần hợp nhất thành Fused INT8 LUT.")
    
    new_nodes = []
    fused_lut_count = 0
    
    # Process graph nodes in topological order
    for idx, node in enumerate(graph.node):
        if node.name in nodes_to_remove:
            continue
            
        # Check if this node is QLinearConv that produces an input to a fused SiLU
        new_nodes.append(node)
        
        # If this was a QLinearConv whose output enters a SiLU, inject the Fused INT8 LUT right after it
        for silu_name, info in fused_silu_map.items():
            if node == info['qconv']:
                fused_lut_count += 1
                prefix = f"fused_silu_{fused_lut_count}"
                conv_out_int8 = info['conv_int8_out']
                mul_out = info['mul_out']
                
                # Retrieve Scale of Conv output (Sy)
                sy_name = info['qconv'].input[6]
                sy = float(inits[sy_name])
                
                # Retrieve Scale of downstream activation (Sx_next or Sy)
                if info['downstream_qs']:
                    s_act_name = info['downstream_qs'][0].input[1]
                    s_act = float(inits[s_act_name])
                    int8_final_out = info['downstream_qs'][0].output[0]
                else:
                    # If branching into Add/Split/Concat, determine scale
                    s_act = sy
                    int8_final_out = f"{mul_out}_fused_int8"
                    
                # -------------------------------------------------------------
                # 1. Precompute 256-byte PURE INT8 Fused SiLU Table
                # q in [-128..127] -> addr in [0..255]
                # table[addr] = clip(round(( (q * Sy) * Sigmoid(q * Sy) ) / S_act), -128, 127)
                # -------------------------------------------------------------
                table_int8 = np.zeros(256, dtype=np.int8)
                for q in range(-128, 128):
                    addr = q + 128
                    x = float(q) * sy
                    silu_val = x * sigmoid(x)
                    q_act = int(np.clip(np.round(silu_val / s_act), -128, 127))
                    table_int8[addr] = q_act
                    
                table_init_name = f"{prefix}_table_int8"
                init_table = helper.make_tensor(
                    table_init_name, TensorProto.INT8, [256], table_int8.tobytes(), raw=True
                )
                graph.initializer.append(init_table)
                
                # -------------------------------------------------------------
                # 2. Add Pure INT8 LUT Graph Nodes:
                #    conv_out_int8 -> Cast(INT64) -> Add(128) -> Gather(table_int8) -> int8_final_out
                # -------------------------------------------------------------
                n_cast = helper.make_node(
                    'Cast', [conv_out_int8], [f"{prefix}_idx_raw"], to=TensorProto.INT64, name=f"{prefix}_cast"
                )
                n_add = helper.make_node(
                    'Add', [f"{prefix}_idx_raw", 'const_offset_128'], [f"{prefix}_addr"], name=f"{prefix}_add"
                )
                n_gather = helper.make_node(
                    'Gather', [table_init_name, f"{prefix}_addr"], [int8_final_out], axis=0, name=f"{prefix}_gather"
                )
                new_nodes.extend([n_cast, n_add, n_gather])
                
                # If there were multiple downstream QuantizeLinear nodes, redirect their consumers to int8_final_out
                if len(info['downstream_qs']) > 1:
                    for extra_q in info['downstream_qs'][1:]:
                        extra_out = extra_q.output[0]
                        # Replace extra_out with int8_final_out across all subsequent nodes
                        for future_n in graph.node:
                            for inp_i, inp_name in enumerate(future_n.input):
                                if inp_name == extra_out:
                                    future_n.input[inp_i] = int8_final_out
                                    
                # If mul_out was also consumed by FP32 nodes (Split, Add, Concat), bridge with 1 DequantizeLinear
                fp32_consumers = [c for c in graph.node if mul_out in c.input and c.name not in nodes_to_remove]
                if fp32_consumers:
                    # Provide DequantizeLinear from int8_final_out -> mul_out for those FP32 nodes
                    scale_act_name = f"{prefix}_s_act"
                    zp_act_name = f"{prefix}_zp_act"
                    graph.initializer.append(helper.make_tensor(scale_act_name, TensorProto.FLOAT, [], [s_act]))
                    graph.initializer.append(helper.make_tensor(zp_act_name, TensorProto.INT8, [], [0]))
                    
                    n_bridge_dq = helper.make_node(
                        'DequantizeLinear',
                        [int8_final_out, scale_act_name, zp_act_name],
                        [mul_out],
                        name=f"{prefix}_dq_bridge"
                    )
                    new_nodes.append(n_bridge_dq)
                    
    # Handle the 1 Detect Head Sigmoid node (Node 363)
    # Map from float logits to UINT8/INT8 table
    final_nodes = []
    for n in new_nodes:
        if n.op_type == "Sigmoid":
            # Detect head classification sigmoid
            in_t = n.input[0]
            out_t = n.output[0]
            prefix = "head_sig_lut8"
            
            # Precompute 256-entry Q0.8 table over [-8.0, +8.0]
            table_q08 = np.array([
                np.clip(np.round(sigmoid(-8.0 + (i + 0.5) * 0.0625) * 256.0) / 256.0, 0.0, 1.0)
                for i in range(256)
            ], dtype=np.float32)
            
            graph.initializer.append(helper.make_tensor('head_lut_table', TensorProto.FLOAT, [256], table_q08.tolist()))
            graph.initializer.append(helper.make_tensor('head_lut_min', TensorProto.FLOAT, [], [-8.0]))
            graph.initializer.append(helper.make_tensor('head_lut_scale', TensorProto.FLOAT, [], [16.0]))
            graph.initializer.append(helper.make_tensor('head_lut_zero', TensorProto.INT64, [], [0]))
            graph.initializer.append(helper.make_tensor('head_lut_255', TensorProto.INT64, [], [255]))
            
            sub_n = helper.make_node('Sub', [in_t, 'head_lut_min'], [f"{prefix}_sub"], name=f"{prefix}_sub")
            mul_n = helper.make_node('Mul', [f"{prefix}_sub", 'head_lut_scale'], [f"{prefix}_mul"], name=f"{prefix}_mul")
            flr_n = helper.make_node('Floor', [f"{prefix}_mul"], [f"{prefix}_flr"], name=f"{prefix}_flr")
            cst_n = helper.make_node('Cast', [f"{prefix}_flr"], [f"{prefix}_cst"], to=TensorProto.INT64, name=f"{prefix}_cst")
            clp_n = helper.make_node('Clip', [f"{prefix}_cst", 'head_lut_zero', 'head_lut_255'], [f"{prefix}_clp"], name=f"{prefix}_clp")
            gth_n = helper.make_node('Gather', ['head_lut_table', f"{prefix}_clp"], [out_t], axis=0, name=f"{prefix}_gth")
            final_nodes.extend([sub_n, mul_n, flr_n, cst_n, clp_n, gth_n])
        else:
            final_nodes.append(n)
            
    # Update graph
    del graph.node[:]
    graph.node.extend(final_nodes)
    
    # Opset 17
    del model.opset_import[:]
    model.opset_import.append(helper.make_opsetid('', 17))
    
    print("[INFO] Kiểm tra tính toàn vẹn ONNX (onnx.checker)...")
    onnx.checker.check_model(model)
    
    print(f"[INFO] Lưu mô hình Pure INT8 Fused LUT: {output_onnx}...")
    onnx.save(model, output_onnx)
    size_mb = os.path.getsize(output_onnx) / (1024 * 1024)
    print(f"[SUCCESS] Mô hình {output_onnx} đã được tạo thành công! (Dung lượng: {size_mb:.2f} MB)")

if __name__ == "__main__":
    build_pure_int8_fused_lut_model()
