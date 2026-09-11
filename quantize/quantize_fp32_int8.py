import os
import shutil
import glob
import onnx
from onnx import numpy_helper
import numpy as np
from ultralytics import YOLO



#=====================================================================
# BƯỚC 2: CẤU HÌNH YAML VÀ EXPORT ONNX INT8
# =====================================================================
yaml_content = f"""
path: {"C:/Users/PC/Downloads/calib_100"}
train: .
val: .

nc: 1
names:
  0: uav
"""

yaml_path = "C:/Users/PC/Downloads/dataset_calib.yaml"
with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)

print(f"[✓] Đã tạo file cấu hình: {yaml_path}")

# Nạp model
model = YOLO("C:/Users/PC/Downloads/best (2).pt")

print("[*] Đang thực hiện Lượng hóa INT8 (ONNX Static Quantization)...")

# Hứng đường dẫn file ONNX INT8 do Ultralytics xuất ra
onnx_exported_path = model.export(
    format="onnx", 
    int8=True, 
    data=yaml_path
)

print(f"[✓] Export thành công file ONNX INT8: {onnx_exported_path}\n")

# =====================================================================
# BƯỚC 3: TRÍCH XUẤT WEIGHT INT8 & BIAS INT32 RA FILE .TXT (DECIMAL)
# =====================================================================
def extract_conv_params(onnx_path, output_dir="C:/Users/PC/Downloads/conv_params"):
    os.makedirs(output_dir, exist_ok=True)
    model_onnx = onnx.load(onnx_path)
    graph = model_onnx.graph

    init_dict = {init.name: numpy_helper.to_array(init) for init in graph.initializer}

    print(f"[*] Đang trích xuất thông số từ: {onnx_path}")
    print(f"[*] Thư mục lưu kết quả: {output_dir}/\n")

    summary_path = os.path.join(output_dir, "conv_layers_summary.txt")
    conv_count = 0

    with open(summary_path, "w", encoding="utf-8") as f_sum:
        f_sum.write("STT, NODE_NAME, WEIGHT_FILE, WEIGHT_SHAPE, BIAS_FILE, BIAS_SHAPE\n")

        for node in graph.node:
            if node.op_type in ['Conv', 'QLinearConv', 'ConvInteger']:
                conv_count += 1
                node_name = node.name if node.name else f"Conv_{conv_count}"

                if node.op_type in ['Conv', 'ConvInteger']:
                    w_name = node.input[1] if len(node.input) > 1 else None
                    b_name = node.input[2] if len(node.input) > 2 else None
                elif node.op_type == 'QLinearConv':
                    w_name = node.input[3] if len(node.input) > 3 else None
                    b_name = node.input[8] if len(node.input) > 8 else None

                # --- XỬ LÝ WEIGHT ---
                w_file_str, w_shape_str = "None", "None"
                if w_name and w_name in init_dict:
                    w_data = init_dict[w_name]
                    w_filename = f"layer_{conv_count:02d}_conv_weight_{w_data.dtype}.txt"
                    w_path = os.path.join(output_dir, w_filename)

                    fmt_type = '%d' if np.issubdtype(w_data.dtype, np.integer) else '%.6f'
                    np.savetxt(w_path, w_data.flatten(), fmt=fmt_type)

                    w_file_str = w_filename
                    w_shape_str = str(list(w_data.shape))

                # --- XỬ LÝ BIAS ---
                b_file_str, b_shape_str = "None", "None"
                if b_name and b_name in init_dict:
                    b_data = init_dict[b_name]
                    b_filename = f"layer_{conv_count:02d}_conv_bias_{b_data.dtype}.txt"
                    b_path = os.path.join(output_dir, b_filename)

                    fmt_type = '%d' if np.issubdtype(b_data.dtype, np.integer) else '%.6f'
                    np.savetxt(b_path, b_data.flatten(), fmt=fmt_type)

                    b_file_str = b_filename
                    b_shape_str = str(list(b_data.shape))

                print(f"[{conv_count:02d}] {node_name:<35} | Weight: {w_shape_str:<18} -> {w_file_str}")
                if b_file_str != "None":
                    print(f"     {'':<35} | Bias:   {b_shape_str:<18} -> {b_file_str}")

                f_sum.write(f"{conv_count}, {node_name}, {w_file_str}, {w_shape_str}, {b_file_str}, {b_shape_str}\n")

    print(f"\n[✓] Thành công! Đã trích xuất xong {conv_count} khối Conv.")
    print(f"[✓] File tóm tắt thông số lưu tại: {summary_path}")

# Gọi hàm trích xuất
extract_conv_params(onnx_exported_path)