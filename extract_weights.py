# #!/usr/bin/env python3

# import onnx
# import onnxruntime as ort
# import numpy as np
# import os
# import cv2

# def write_tensor_to_txt(filepath, data):
#     with open(filepath, 'w') as f:
#         flat = data.flatten()
#         for i, v in enumerate(flat):
#             f.write(f"{float(v):.6f}")
#             if (i + 1) % 10 == 0:
#                 f.write("\n")
#             else:
#                 f.write(" ")
#         f.write("\n")

# def process_image(img_path):
#     print(f"[INFO] Đọc ảnh từ: {img_path}")
#     img = cv2.imread(img_path)
#     if img is None:
#         raise ValueError(f"Không thể đọc ảnh: {img_path}")
    
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (640, 640)) 
#     img = img.astype(np.float32) / 255.0
#     img = np.transpose(img, (2, 0, 1))
#     img = np.expand_dims(img, axis=0)
#     return img

# def extract_weights_and_golden(onnx_path, img_path, backbone_dir, neck_dir):
#     model = onnx.load(onnx_path)
#     initializers = {init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer}

#     # 1. Trích xuất Weights & Biases
#     backbone_modules = [
#         ("module_0_conv_P1", [("model.0.conv.weight", "node_0_weight.txt"), ("model.0.conv.bias", "node_0_bias.txt")]),
#         ("module_1_conv_P2", [("model.1.conv.weight", "node_3_weight.txt"), ("model.1.conv.bias", "node_3_bias.txt")]),
#         ("module_2_c3k2_P2", [
#             ("model.2.cv1.conv.weight", "node_6_weight.txt"), ("model.2.cv1.conv.bias", "node_6_bias.txt"),
#             ("model.2.m.0.cv1.conv.weight", "node_10_weight.txt"), ("model.2.m.0.cv1.conv.bias", "node_10_bias.txt"),
#             ("model.2.m.0.cv2.conv.weight", "node_13_weight.txt"), ("model.2.m.0.cv2.conv.bias", "node_13_bias.txt"),
#             ("model.2.cv2.conv.weight", "node_18_weight.txt"), ("model.2.cv2.conv.bias", "node_18_bias.txt"),
#         ]),
#         ("module_3_conv_P3", [("model.3.conv.weight", "node_21_weight.txt"), ("model.3.conv.bias", "node_21_bias.txt")]),
#         ("module_4_c3k2_P3", [
#             ("model.4.cv1.conv.weight", "node_24_weight.txt"), ("model.4.cv1.conv.bias", "node_24_bias.txt"),
#             ("model.4.m.0.cv1.conv.weight", "node_28_weight.txt"), ("model.4.m.0.cv1.conv.bias", "node_28_bias.txt"),
#             ("model.4.m.0.cv2.conv.weight", "node_31_weight.txt"), ("model.4.m.0.cv2.conv.bias", "node_31_bias.txt"),
#             ("model.4.cv2.conv.weight", "node_36_weight.txt"), ("model.4.cv2.conv.bias", "node_36_bias.txt"),
#         ]),
#         ("module_5_conv_P4", [("model.5.conv.weight", "node_39_weight.txt"), ("model.5.conv.bias", "node_39_bias.txt")]),
#         ("module_6_c3k2_P4", [
#             ("model.6.cv1.conv.weight", "node_42_weight.txt"), ("model.6.cv1.conv.bias", "node_42_bias.txt"),
#             ("model.6.m.0.cv1.conv.weight", "node_46_weight.txt"), ("model.6.m.0.cv1.conv.bias", "node_46_bias.txt"),
#             ("model.6.m.0.cv2.conv.weight", "node_47_weight.txt"), ("model.6.m.0.cv2.conv.bias", "node_47_bias.txt"),
#             ("model.6.m.0.m.0.cv1.conv.weight", "node_52_weight.txt"), ("model.6.m.0.m.0.cv1.conv.bias", "node_52_bias.txt"),
#             ("model.6.m.0.m.0.cv2.conv.weight", "node_55_weight.txt"), ("model.6.m.0.m.0.cv2.conv.bias", "node_55_bias.txt"),
#             ("model.6.m.0.m.1.cv1.conv.weight", "node_59_weight.txt"), ("model.6.m.0.m.1.cv1.conv.bias", "node_59_bias.txt"),
#             ("model.6.m.0.m.1.cv2.conv.weight", "node_62_weight.txt"), ("model.6.m.0.m.1.cv2.conv.bias", "node_62_bias.txt"),
#             ("model.6.m.0.cv3.conv.weight", "node_67_weight.txt"), ("model.6.m.0.cv3.conv.bias", "node_67_bias.txt"),
#             ("model.6.cv2.conv.weight", "node_71_weight.txt"), ("model.6.cv2.conv.bias", "node_71_bias.txt"),
#         ]),
#         ("module_7_conv_P5", [("model.7.conv.weight", "node_74_weight.txt"), ("model.7.conv.bias", "node_74_bias.txt")]),
#         ("module_8_c3k2_P5", [
#             ("model.8.cv1.conv.weight", "node_77_weight.txt"), ("model.8.cv1.conv.bias", "node_77_bias.txt"),
#             ("model.8.m.0.cv1.conv.weight", "node_81_weight.txt"), ("model.8.m.0.cv1.conv.bias", "node_81_bias.txt"),
#             ("model.8.m.0.cv2.conv.weight", "node_82_weight.txt"), ("model.8.m.0.cv2.conv.bias", "node_82_bias.txt"),
#             ("model.8.m.0.m.0.cv1.conv.weight", "node_87_weight.txt"), ("model.8.m.0.m.0.cv1.conv.bias", "node_87_bias.txt"),
#             ("model.8.m.0.m.0.cv2.conv.weight", "node_90_weight.txt"), ("model.8.m.0.m.0.cv2.conv.bias", "node_90_bias.txt"),
#             ("model.8.m.0.m.1.cv1.conv.weight", "node_94_weight.txt"), ("model.8.m.0.m.1.cv1.conv.bias", "node_94_bias.txt"),
#             ("model.8.m.0.m.1.cv2.conv.weight", "node_97_weight.txt"), ("model.8.m.0.m.1.cv2.conv.bias", "node_97_bias.txt"),
#             ("model.8.m.0.cv3.conv.weight", "node_102_weight.txt"), ("model.8.m.0.cv3.conv.bias", "node_102_bias.txt"),
#             ("model.8.cv2.conv.weight", "node_106_weight.txt"), ("model.8.cv2.conv.bias", "node_106_bias.txt"),
#         ]),
#     ]

#     neck_modules = [
#         ("module_9_sppf", [
#             ("model.9.cv1.conv.weight", "node_109_weight.txt"), ("model.9.cv1.conv.bias", "node_109_bias.txt"),
#             ("model.9.cv2.conv.weight", "node_114_weight.txt"), ("model.9.cv2.conv.bias", "node_114_bias.txt"),
#         ]),
#         ("module_10_c2psa", [
#             # cv1 (Node 118)
#             ("model.10.cv1.conv.weight", "node_118_weight.txt"),
#             ("model.10.cv1.conv.bias", "node_118_bias.txt"),
            
#             # QKV (Node 122)
#             ("model.10.m.0.attn.qkv.weight", "node_122_weight.txt"),
#             ("model.10.m.0.attn.qkv.bias", "node_122_bias.txt"),
#             ("model.10.m.0.attn.qkv.conv.weight", "node_122_weight.txt"),
#             ("model.10.m.0.attn.qkv.conv.bias", "node_122_bias.txt"),
            
#             # PE / Depthwise (Node 128)
#             ("model.10.m.0.attn.pe.weight", "node_128_weight.txt"),
#             ("model.10.m.0.attn.pe.bias", "node_128_bias.txt"),
#             ("model.10.m.0.attn.pe.conv.weight", "node_128_weight.txt"),
#             ("model.10.m.0.attn.pe.conv.bias", "node_128_bias.txt"),
            
#             # Proj (Node 135)
#             ("model.10.m.0.attn.proj.weight", "node_135_weight.txt"),
#             ("model.10.m.0.attn.proj.bias", "node_135_bias.txt"),
#             ("model.10.m.0.attn.proj.conv.weight", "node_135_weight.txt"),
#             ("model.10.m.0.attn.proj.conv.bias", "node_135_bias.txt"),
            
#             # FFN cv1 (Node 137) - Thêm trường hợp ffn.0
#             ("model.10.m.0.ffn.0.conv.weight", "node_137_weight.txt"),
#             ("model.10.m.0.ffn.0.conv.bias", "node_137_bias.txt"),
#             ("model.10.m.0.ffn.cv1.conv.weight", "node_137_weight.txt"),
#             ("model.10.m.0.ffn.cv1.conv.bias", "node_137_bias.txt"),
            
#             # FFN cv2 (Node 140) - Thêm trường hợp ffn.1
#             ("model.10.m.0.ffn.1.conv.weight", "node_140_weight.txt"),
#             ("model.10.m.0.ffn.1.conv.bias", "node_140_bias.txt"),
#             ("model.10.m.0.ffn.cv2.conv.weight", "node_140_weight.txt"),
#             ("model.10.m.0.ffn.cv2.conv.bias", "node_140_bias.txt"),
            
#             # cv2 đầu ra (Node 143)
#             ("model.10.cv2.conv.weight", "node_143_weight.txt"),
#             ("model.10.cv2.conv.bias", "node_143_bias.txt"),
#         ]),
#         ("module_13_c3k2", [
#             ("model.13.cv1.conv.weight", "node_148_weight.txt"), ("model.13.cv1.conv.bias", "node_148_bias.txt"),
#             ("model.13.m.0.cv1.conv.weight", "node_152_weight.txt"), ("model.13.m.0.cv1.conv.bias", "node_152_bias.txt"),
#             ("model.13.m.0.cv2.conv.weight", "node_153_weight.txt"), ("model.13.m.0.cv2.conv.bias", "node_153_bias.txt"),
#             ("model.13.m.0.m.0.cv1.conv.weight", "node_158_weight.txt"), ("model.13.m.0.m.0.cv1.conv.bias", "node_158_bias.txt"),
#             ("model.13.m.0.m.0.cv2.conv.weight", "node_161_weight.txt"), ("model.13.m.0.m.0.cv2.conv.bias", "node_161_bias.txt"),
#             ("model.13.m.0.m.1.cv1.conv.weight", "node_165_weight.txt"), ("model.13.m.0.m.1.cv1.conv.bias", "node_165_bias.txt"),
#             ("model.13.m.0.m.1.cv2.conv.weight", "node_168_weight.txt"), ("model.13.m.0.m.1.cv2.conv.bias", "node_168_bias.txt"),
#             ("model.13.m.0.cv3.conv.weight", "node_173_weight.txt"), ("model.13.m.0.cv3.conv.bias", "node_173_bias.txt"),
#             ("model.13.cv2.conv.weight", "node_177_weight.txt"), ("model.13.cv2.conv.bias", "node_177_bias.txt"),
#         ]),
#         ("module_16_c3k2", [
#             ("model.16.cv1.conv.weight", "node_182_weight.txt"), ("model.16.cv1.conv.bias", "node_182_bias.txt"),
#             ("model.16.m.0.cv1.conv.weight", "node_186_weight.txt"), ("model.16.m.0.cv1.conv.bias", "node_186_bias.txt"),
#             ("model.16.m.0.cv2.conv.weight", "node_187_weight.txt"), ("model.16.m.0.cv2.conv.bias", "node_187_bias.txt"),
#             ("model.16.m.0.m.0.cv1.conv.weight", "node_192_weight.txt"), ("model.16.m.0.m.0.cv1.conv.bias", "node_192_bias.txt"),
#             ("model.16.m.0.m.0.cv2.conv.weight", "node_195_weight.txt"), ("model.16.m.0.m.0.cv2.conv.bias", "node_195_bias.txt"),
#             ("model.16.m.0.m.1.cv1.conv.weight", "node_199_weight.txt"), ("model.16.m.0.m.1.cv1.conv.bias", "node_199_bias.txt"),
#             ("model.16.m.0.m.1.cv2.conv.weight", "node_202_weight.txt"), ("model.16.m.0.m.1.cv2.conv.bias", "node_202_bias.txt"),
#             ("model.16.m.0.cv3.conv.weight", "node_207_weight.txt"), ("model.16.m.0.cv3.conv.bias", "node_207_bias.txt"),
#             ("model.16.cv2.conv.weight", "node_211_weight.txt"), ("model.16.cv2.conv.bias", "node_211_bias.txt"),
#         ]),
#         ("module_17_conv", [("model.17.conv.weight", "node_214_weight.txt"), ("model.17.conv.bias", "node_214_bias.txt")]),
#         ("module_19_c3k2", [
#             ("model.19.cv1.conv.weight", "node_226_weight.txt"), ("model.19.cv1.conv.bias", "node_226_bias.txt"),
#             ("model.19.m.0.cv1.conv.weight", "node_238_weight.txt"), ("model.19.m.0.cv1.conv.bias", "node_238_bias.txt"),
#             ("model.19.m.0.cv2.conv.weight", "node_239_weight.txt"), ("model.19.m.0.cv2.conv.bias", "node_239_bias.txt"),
#             ("model.19.m.0.m.0.cv1.conv.weight", "node_247_weight.txt"), ("model.19.m.0.m.0.cv1.conv.bias", "node_247_bias.txt"),
#             ("model.19.m.0.m.0.cv2.conv.weight", "node_253_weight.txt"), ("model.19.m.0.m.0.cv2.conv.bias", "node_253_bias.txt"),
#             ("model.19.m.0.m.1.cv1.conv.weight", "node_257_weight.txt"), ("model.19.m.0.m.1.cv1.conv.bias", "node_257_bias.txt"),
#             ("model.19.m.0.m.1.cv2.conv.weight", "node_260_weight.txt"), ("model.19.m.0.m.1.cv2.conv.bias", "node_260_bias.txt"),
#             ("model.19.m.0.cv3.conv.weight", "node_265_weight.txt"), ("model.19.m.0.cv3.conv.bias", "node_265_bias.txt"),
#             ("model.19.cv2.conv.weight", "node_269_weight.txt"), ("model.19.cv2.conv.bias", "node_269_bias.txt"),
#         ]),
#         ("module_20_conv", [("model.20.conv.weight", "node_272_weight.txt"), ("model.20.conv.bias", "node_272_bias.txt")]),
#     ]

#     print("\n[INFO] Đang trích xuất cấu trúc Weights Backbone...")
#     for mod_name, param_mapping in backbone_modules:
#         mod_dir = os.path.join(backbone_dir, mod_name)
#         os.makedirs(mod_dir, exist_ok=True)
#         for onnx_name, txt_name in param_mapping:
#             if onnx_name in initializers:
#                 write_tensor_to_txt(os.path.join(mod_dir, txt_name), initializers[onnx_name])

#     print("\n[INFO] Đang trích xuất cấu trúc Weights Neck...")
#     for mod_name, param_mapping in neck_modules:
#         mod_dir = os.path.join(neck_dir, mod_name)
#         os.makedirs(mod_dir, exist_ok=True)
#         for onnx_name, txt_name in param_mapping:
#             if onnx_name in initializers:
#                 write_tensor_to_txt(os.path.join(mod_dir, txt_name), initializers[onnx_name])
#             elif "scale" in onnx_name: 
#                 try:
#                     for init in model.graph.initializer:
#                         if "scale" in init.name: 
#                             write_tensor_to_txt(os.path.join(mod_dir, txt_name), onnx.numpy_helper.to_array(init))
#                             break
#                 except Exception as e:
#                      print(f"Lỗi khi trích xuất scale: {e}")

#     # 2. Xử lý Image và lưu ra txt
#     print("\n[INFO] Đang xử lý Input Image...")
#     input_img = process_image(img_path)
#     write_tensor_to_txt("input_images.txt", input_img)
#     print(f"[INFO] Đã xuất ảnh đầu vào ra input_images.txt")

#     # 3. Can thiệp vào đồ thị ONNX lấy kết quả từng NODE
#     print("\n[INFO] Đang trích xuất Golden Output cho TỪNG NODE trong Backbone và Neck...")
#     golden_dir = "golden_outputs"
#     os.makedirs(golden_dir, exist_ok=True)

#     backbone_tensor_names = []
#     # Các node cần bỏ qua (thuộc Detect Head)
#     skip_nodes = {215, 216, 218, 219, 227, 228, 230, 231, 233, 234, 236, 237, 240, 242, 243, 245, 246, 248, 250, 252, 273, 274, 276, 277, 279}

#     for i, node in enumerate(model.graph.node):
#         if i > 279:  # Giới hạn ở Node 279 (hết Neck)
#             break
#         if i in skip_nodes:
#             continue
#         tensor_name = node.output[0]
#         backbone_tensor_names.append(tensor_name)
        
#         intermediate_layer_value_info = onnx.helper.ValueInfoProto()
#         intermediate_layer_value_info.name = tensor_name
#         model.graph.output.append(intermediate_layer_value_info)

#     sess_options = ort.SessionOptions()
#     sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
#     sess = ort.InferenceSession(model.SerializeToString(), sess_options, providers=['CPUExecutionProvider'])
    
#     input_name = sess.get_inputs()[0].name
#     outputs = sess.run(backbone_tensor_names, {input_name: input_img})
    
#     out_idx = 0
#     for i in range(280):
#         if i in skip_nodes:
#             continue
#         filepath = os.path.join(golden_dir, f"node_{i}_golden.txt")
#         write_tensor_to_txt(filepath, outputs[out_idx])
#         out_idx += 1
        
#         if i == 108 or i == 117 or i == 145 or i == 179 or i == 213 or i == 271 or i == 278:
#             print(f"-> Đã xuất Node {i} ra {filepath}")
    
#     print("...")

# if __name__ == "__main__":
#     onnx_path = "/home/hiura/Hiura/demo/best.onnx"
#     img_path = "/home/hiura/Hiura/demo/image.png"
#     backbone_dir = "weight/backbone"
#     neck_dir = "weight/neck"
    
#     extract_weights_and_golden(onnx_path, img_path, backbone_dir, neck_dir)
#     print("\n✅ Hoàn tất toàn bộ!")
#!/usr/bin/env python3
#!/usr/bin/env python3

import onnx
import onnxruntime as ort
import numpy as np
import os
import cv2

def write_tensor_to_txt(filepath, data):
    with open(filepath, 'w') as f:
        flat = data.flatten()
        for i, v in enumerate(flat):
            f.write(f"{float(v):.6f}")
            if (i + 1) % 10 == 0:
                f.write("\n")
            else:
                f.write(" ")
        f.write("\n")

def process_image(img_path):
    print(f"[INFO] Đọc ảnh từ: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640)) 
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def extract_weights_and_golden(onnx_path, img_path, backbone_dir, neck_dir, head_dir):
    model = onnx.load(onnx_path)
    initializers = {init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer}

    def extract_module_weights(modules, base_dir):
        for mod_name, param_mapping in modules:
            mod_dir = os.path.join(base_dir, mod_name)
            os.makedirs(mod_dir, exist_ok=True)
            for possible_names, txt_name in param_mapping:
                if isinstance(possible_names, str):
                    possible_names = [possible_names]
                for onnx_name in possible_names:
                    if onnx_name in initializers:
                        write_tensor_to_txt(os.path.join(mod_dir, txt_name), initializers[onnx_name])
                        break
                    elif "scale" in onnx_name: 
                        for init in model.graph.initializer:
                            if "scale" in init.name: 
                                write_tensor_to_txt(os.path.join(mod_dir, txt_name), onnx.numpy_helper.to_array(init))
                                break
                        break

    # --- 1. BACKBONE & NECK (Giữ nguyên vì đã chạy đúng 100%) ---
    backbone_modules = [
        ("module_0_conv_P1", [("model.0.conv.weight", "node_0_weight.txt"), ("model.0.conv.bias", "node_0_bias.txt")]),
        ("module_1_conv_P2", [("model.1.conv.weight", "node_3_weight.txt"), ("model.1.conv.bias", "node_3_bias.txt")]),
        ("module_2_c3k2_P2", [
            ("model.2.cv1.conv.weight", "node_6_weight.txt"), ("model.2.cv1.conv.bias", "node_6_bias.txt"),
            ("model.2.m.0.cv1.conv.weight", "node_10_weight.txt"), ("model.2.m.0.cv1.conv.bias", "node_10_bias.txt"),
            ("model.2.m.0.cv2.conv.weight", "node_13_weight.txt"), ("model.2.m.0.cv2.conv.bias", "node_13_bias.txt"),
            ("model.2.cv2.conv.weight", "node_18_weight.txt"), ("model.2.cv2.conv.bias", "node_18_bias.txt"),
        ]),
        ("module_3_conv_P3", [("model.3.conv.weight", "node_21_weight.txt"), ("model.3.conv.bias", "node_21_bias.txt")]),
        ("module_4_c3k2_P3", [
            ("model.4.cv1.conv.weight", "node_24_weight.txt"), ("model.4.cv1.conv.bias", "node_24_bias.txt"),
            ("model.4.m.0.cv1.conv.weight", "node_28_weight.txt"), ("model.4.m.0.cv1.conv.bias", "node_28_bias.txt"),
            ("model.4.m.0.cv2.conv.weight", "node_31_weight.txt"), ("model.4.m.0.cv2.conv.bias", "node_31_bias.txt"),
            ("model.4.cv2.conv.weight", "node_36_weight.txt"), ("model.4.cv2.conv.bias", "node_36_bias.txt"),
        ]),
        ("module_5_conv_P4", [("model.5.conv.weight", "node_39_weight.txt"), ("model.5.conv.bias", "node_39_bias.txt")]),
        ("module_6_c3k2_P4", [
            ("model.6.cv1.conv.weight", "node_42_weight.txt"), ("model.6.cv1.conv.bias", "node_42_bias.txt"),
            ("model.6.m.0.cv1.conv.weight", "node_46_weight.txt"), ("model.6.m.0.cv1.conv.bias", "node_46_bias.txt"),
            ("model.6.m.0.cv2.conv.weight", "node_47_weight.txt"), ("model.6.m.0.cv2.conv.bias", "node_47_bias.txt"),
            ("model.6.m.0.m.0.cv1.conv.weight", "node_52_weight.txt"), ("model.6.m.0.m.0.cv1.conv.bias", "node_52_bias.txt"),
            ("model.6.m.0.m.0.cv2.conv.weight", "node_55_weight.txt"), ("model.6.m.0.m.0.cv2.conv.bias", "node_55_bias.txt"),
            ("model.6.m.0.m.1.cv1.conv.weight", "node_59_weight.txt"), ("model.6.m.0.m.1.cv1.conv.bias", "node_59_bias.txt"),
            ("model.6.m.0.m.1.cv2.conv.weight", "node_62_weight.txt"), ("model.6.m.0.m.1.cv2.conv.bias", "node_62_bias.txt"),
            ("model.6.m.0.cv3.conv.weight", "node_67_weight.txt"), ("model.6.m.0.cv3.conv.bias", "node_67_bias.txt"),
            ("model.6.cv2.conv.weight", "node_71_weight.txt"), ("model.6.cv2.conv.bias", "node_71_bias.txt"),
        ]),
        ("module_7_conv_P5", [("model.7.conv.weight", "node_74_weight.txt"), ("model.7.conv.bias", "node_74_bias.txt")]),
        ("module_8_c3k2_P5", [
            ("model.8.cv1.conv.weight", "node_77_weight.txt"), ("model.8.cv1.conv.bias", "node_77_bias.txt"),
            ("model.8.m.0.cv1.conv.weight", "node_81_weight.txt"), ("model.8.m.0.cv1.conv.bias", "node_81_bias.txt"),
            ("model.8.m.0.cv2.conv.weight", "node_82_weight.txt"), ("model.8.m.0.cv2.conv.bias", "node_82_bias.txt"),
            ("model.8.m.0.m.0.cv1.conv.weight", "node_87_weight.txt"), ("model.8.m.0.m.0.cv1.conv.bias", "node_87_bias.txt"),
            ("model.8.m.0.m.0.cv2.conv.weight", "node_90_weight.txt"), ("model.8.m.0.m.0.cv2.conv.bias", "node_90_bias.txt"),
            ("model.8.m.0.m.1.cv1.conv.weight", "node_94_weight.txt"), ("model.8.m.0.m.1.cv1.conv.bias", "node_94_bias.txt"),
            ("model.8.m.0.m.1.cv2.conv.weight", "node_97_weight.txt"), ("model.8.m.0.m.1.cv2.conv.bias", "node_97_bias.txt"),
            ("model.8.m.0.cv3.conv.weight", "node_102_weight.txt"), ("model.8.m.0.cv3.conv.bias", "node_102_bias.txt"),
            ("model.8.cv2.conv.weight", "node_106_weight.txt"), ("model.8.cv2.conv.bias", "node_106_bias.txt"),
        ])
    ]
    neck_modules = [
        ("module_9_sppf", [
            ("model.9.cv1.conv.weight", "node_109_weight.txt"), ("model.9.cv1.conv.bias", "node_109_bias.txt"),
            ("model.9.cv2.conv.weight", "node_114_weight.txt"), ("model.9.cv2.conv.bias", "node_114_bias.txt"),
        ]),
        ("module_10_c2psa", [
            ("model.10.cv1.conv.weight", "node_118_weight.txt"), ("model.10.cv1.conv.bias", "node_118_bias.txt"),
            (["model.10.m.0.attn.qkv.weight", "model.10.m.0.attn.qkv.conv.weight"], "node_122_weight.txt"),
            (["model.10.m.0.attn.qkv.bias", "model.10.m.0.attn.qkv.conv.bias"], "node_122_bias.txt"),
            (["model.10.m.0.attn.pe.weight", "model.10.m.0.attn.pe.conv.weight"], "node_128_weight.txt"),
            (["model.10.m.0.attn.pe.bias", "model.10.m.0.attn.pe.conv.bias"], "node_128_bias.txt"),
            (["model.10.m.0.attn.proj.weight", "model.10.m.0.attn.proj.conv.weight"], "node_135_weight.txt"),
            (["model.10.m.0.attn.proj.bias", "model.10.m.0.attn.proj.conv.bias"], "node_135_bias.txt"),
            (["model.10.m.0.ffn.0.conv.weight", "model.10.m.0.ffn.cv1.conv.weight"], "node_137_weight.txt"),
            (["model.10.m.0.ffn.0.conv.bias", "model.10.m.0.ffn.cv1.conv.bias"], "node_137_bias.txt"),
            (["model.10.m.0.ffn.1.conv.weight", "model.10.m.0.ffn.cv2.conv.weight"], "node_140_weight.txt"),
            (["model.10.m.0.ffn.1.conv.bias", "model.10.m.0.ffn.cv2.conv.bias"], "node_140_bias.txt"),
            ("model.10.cv2.conv.weight", "node_143_weight.txt"), ("model.10.cv2.conv.bias", "node_143_bias.txt"),
        ]),
        ("module_13_c3k2", [
            ("model.13.cv1.conv.weight", "node_148_weight.txt"), ("model.13.cv1.conv.bias", "node_148_bias.txt"),
            ("model.13.m.0.cv1.conv.weight", "node_152_weight.txt"), ("model.13.m.0.cv1.conv.bias", "node_152_bias.txt"),
            ("model.13.m.0.cv2.conv.weight", "node_153_weight.txt"), ("model.13.m.0.cv2.conv.bias", "node_153_bias.txt"),
            ("model.13.m.0.m.0.cv1.conv.weight", "node_158_weight.txt"), ("model.13.m.0.m.0.cv1.conv.bias", "node_158_bias.txt"),
            ("model.13.m.0.m.0.cv2.conv.weight", "node_161_weight.txt"), ("model.13.m.0.m.0.cv2.conv.bias", "node_161_bias.txt"),
            ("model.13.m.0.m.1.cv1.conv.weight", "node_165_weight.txt"), ("model.13.m.0.m.1.cv1.conv.bias", "node_165_bias.txt"),
            ("model.13.m.0.m.1.cv2.conv.weight", "node_168_weight.txt"), ("model.13.m.0.m.1.cv2.conv.bias", "node_168_bias.txt"),
            ("model.13.m.0.cv3.conv.weight", "node_173_weight.txt"), ("model.13.m.0.cv3.conv.bias", "node_173_bias.txt"),
            ("model.13.cv2.conv.weight", "node_177_weight.txt"), ("model.13.cv2.conv.bias", "node_177_bias.txt"),
        ]),
        ("module_16_c3k2", [
            ("model.16.cv1.conv.weight", "node_182_weight.txt"), ("model.16.cv1.conv.bias", "node_182_bias.txt"),
            ("model.16.m.0.cv1.conv.weight", "node_186_weight.txt"), ("model.16.m.0.cv1.conv.bias", "node_186_bias.txt"),
            ("model.16.m.0.cv2.conv.weight", "node_187_weight.txt"), ("model.16.m.0.cv2.conv.bias", "node_187_bias.txt"),
            ("model.16.m.0.m.0.cv1.conv.weight", "node_192_weight.txt"), ("model.16.m.0.m.0.cv1.conv.bias", "node_192_bias.txt"),
            ("model.16.m.0.m.0.cv2.conv.weight", "node_195_weight.txt"), ("model.16.m.0.m.0.cv2.conv.bias", "node_195_bias.txt"),
            ("model.16.m.0.m.1.cv1.conv.weight", "node_199_weight.txt"), ("model.16.m.0.m.1.cv1.conv.bias", "node_199_bias.txt"),
            ("model.16.m.0.m.1.cv2.conv.weight", "node_202_weight.txt"), ("model.16.m.0.m.1.cv2.conv.bias", "node_202_bias.txt"),
            ("model.16.m.0.cv3.conv.weight", "node_207_weight.txt"), ("model.16.m.0.cv3.conv.bias", "node_207_bias.txt"),
            ("model.16.cv2.conv.weight", "node_211_weight.txt"), ("model.16.cv2.conv.bias", "node_211_bias.txt"),
        ]),
        ("module_17_conv", [("model.17.conv.weight", "node_214_weight.txt"), ("model.17.conv.bias", "node_214_bias.txt")]),
        ("module_19_c3k2", [
            ("model.19.cv1.conv.weight", "node_226_weight.txt"), ("model.19.cv1.conv.bias", "node_226_bias.txt"),
            ("model.19.m.0.cv1.conv.weight", "node_238_weight.txt"), ("model.19.m.0.cv1.conv.bias", "node_238_bias.txt"),
            ("model.19.m.0.cv2.conv.weight", "node_239_weight.txt"), ("model.19.m.0.cv2.conv.bias", "node_239_bias.txt"),
            ("model.19.m.0.m.0.cv1.conv.weight", "node_247_weight.txt"), ("model.19.m.0.m.0.cv1.conv.bias", "node_247_bias.txt"),
            ("model.19.m.0.m.0.cv2.conv.weight", "node_253_weight.txt"), ("model.19.m.0.m.0.cv2.conv.bias", "node_253_bias.txt"),
            ("model.19.m.0.m.1.cv1.conv.weight", "node_257_weight.txt"), ("model.19.m.0.m.1.cv1.conv.bias", "node_257_bias.txt"),
            ("model.19.m.0.m.1.cv2.conv.weight", "node_260_weight.txt"), ("model.19.m.0.m.1.cv2.conv.bias", "node_260_bias.txt"),
            ("model.19.m.0.cv3.conv.weight", "node_265_weight.txt"), ("model.19.m.0.cv3.conv.bias", "node_265_bias.txt"),
            ("model.19.cv2.conv.weight", "node_269_weight.txt"), ("model.19.cv2.conv.bias", "node_269_bias.txt"),
        ]),
        ("module_20_conv", [("model.20.conv.weight", "node_272_weight.txt"), ("model.20.conv.bias", "node_272_bias.txt")]),
    ]
    print("\n[INFO] Đang trích xuất cấu trúc Weights Backbone...")
    extract_module_weights(backbone_modules, backbone_dir)
    print("\n[INFO] Đang trích xuất cấu trúc Weights Neck...")
    extract_module_weights(neck_modules, neck_dir)

    # --- 2. XỬ LÝ KHỐI HEAD BẰNG ĐỊA CHỈ TRỰC TIẾP TỪ NODE ID (100% Chống trượt) ---
    print("\n[INFO] Đang trích xuất cấu trúc Weights Head bằng Node ID...")
    
    # Liệt kê chính xác ID của các Node "Conv" chứa Weight/Bias ở phần Head
    head_conv_nodes = {
        "module_22_c3k2": [284, 296, 302, 309, 315, 322, 324, 327, 330],
        "module_23_detect": [
            215, 224, 233, 216, 225, 234, 243, 250,
            273, 282, 291, 274, 283, 292, 299, 305,
            333, 339, 345, 334, 340, 346, 351, 359
        ]
    }

    for mod_name, node_ids in head_conv_nodes.items():
        mod_dir = os.path.join(head_dir, mod_name)
        os.makedirs(mod_dir, exist_ok=True)
        
        for nid in node_ids:
            node = model.graph.node[nid]
            if node.op_type != "Conv":
                continue
            
            # Quét Input thứ 1 (Weight)
            if len(node.input) > 1:
                w_name = node.input[1]
                if w_name in initializers:
                    write_tensor_to_txt(os.path.join(mod_dir, f"node_{nid}_weight.txt"), initializers[w_name])
            
            # Quét Input thứ 2 (Bias)
            if len(node.input) > 2:
                b_name = node.input[2]
                if b_name in initializers:
                    write_tensor_to_txt(os.path.join(mod_dir, f"node_{nid}_bias.txt"), initializers[b_name])

    # Trích xuất Anchors & Strides constants (Node 355, 356, 360)
    print("\n[INFO] Đang trích xuất các hệ số Constant (Anchors/Strides)...")
    head_constants = {
        355: "node_355_weight.txt",
        356: "node_356_weight.txt",
        360: "node_360_weight.txt"
    }
    const_dir = os.path.join(head_dir, "module_23_detect")
    os.makedirs(const_dir, exist_ok=True)
    
    for nid, filename in head_constants.items():
        node = model.graph.node[nid]
        for inp in node.input:
            if inp in initializers:
                write_tensor_to_txt(os.path.join(const_dir, filename), initializers[inp])
                break

    # --- 3. XỬ LÝ IMAGE & GOLDEN OUTPUT ---
    print("\n[INFO] Đang xử lý Input Image...")
    input_img = process_image(img_path)
    write_tensor_to_txt("input_images.txt", input_img)

    print("\n[INFO] Đang trích xuất Golden Output cho TỪNG NODE...")
    golden_dir = "golden_outputs"
    os.makedirs(golden_dir, exist_ok=True)

    backbone_tensor_names = []
    for i, node in enumerate(model.graph.node):
        if i > 367:  # Giới hạn đến Node 367
            break
        tensor_name = node.output[0]
        backbone_tensor_names.append(tensor_name)
        intermediate_layer_value_info = onnx.helper.ValueInfoProto()
        intermediate_layer_value_info.name = tensor_name
        model.graph.output.append(intermediate_layer_value_info)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(model.SerializeToString(), sess_options, providers=['CPUExecutionProvider'])
    
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(backbone_tensor_names, {input_name: input_img})
    
    for i in range(len(outputs)):
        filepath = os.path.join(golden_dir, f"node_{i}_golden.txt")
        write_tensor_to_txt(filepath, outputs[i])
        
        if i == 108 or i == 278 or i == 367:
            print(f"-> Đã xuất Node {i} ra {filepath}")
    
    print("\n✅ Hoàn tất toàn bộ!")

if __name__ == "__main__":
    onnx_path = "/home/hiura/Hiura/demo/best.onnx"
    img_path = "/home/hiura/Hiura/demo/image.png"
    backbone_dir = "weight/backbone"
    neck_dir = "weight/neck"
    head_dir = "weight/head"
    
    extract_weights_and_golden(onnx_path, img_path, backbone_dir, neck_dir, head_dir)