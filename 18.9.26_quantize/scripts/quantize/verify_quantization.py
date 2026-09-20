#!/usr/bin/env python3
"""
Comprehensive 4-Level Verification Engine for Quantized YOLOv26n
- Level 1: Numerical bounds & Error statistics for all 207 weight/bias layers
- Level 2: Layer-wise activation comparison (FP32 vs INT8 vs Golden Outputs)
- Level 3: End-to-end visual object detection verification on reference image
- Level 4: Validation set mAP benchmark (mAP@50, mAP@50:95, Precision, Recall, F1)
"""

import os
import json
import cv2
import numpy as np
import onnx
import onnxruntime as ort
import matplotlib.pyplot as plt

def level_1_numerical_verification(quant_params_path="quantization_params.json"):
    print("\n" + "=" * 60)
    print("  LEVEL 1: KIỂM CHỨNG TOÁN HỌC & SAI SỐ SỐ HỌC (207 TENSORS)")
    print("=" * 60)
    
    with open(quant_params_path, 'r', encoding='utf-8') as f:
        records = json.load(f)
        
    total = len(records)
    pass_bounds_sym = 0
    pass_bounds_asym = 0
    cos_sims_sym = []
    sqnr_sym = []
    
    for r in records:
        scale_sym = r["symmetric"]["scale"]
        max_err_sym = r["symmetric"]["max_err"]
        theoretical_bound_sym = scale_sym / 2.0 + 1e-6
        if max_err_sym <= theoretical_bound_sym:
            pass_bounds_sym += 1
            
        scale_asym = r["asymmetric"]["scale"]
        max_err_asym = r["asymmetric"]["max_err"]
        theoretical_bound_asym = scale_asym / 2.0 + 1e-6
        if max_err_asym <= theoretical_bound_asym:
            pass_bounds_asym += 1
            
        cos_sims_sym.append(r["symmetric"]["cosine_sim"])
        sqnr_sym.append(r["symmetric"]["sqnr_db"])
        
    avg_cos = float(np.mean(cos_sims_sym))
    min_cos = float(np.min(cos_sims_sym))
    avg_sqnr = float(np.mean(sqnr_sym))
    min_sqnr = float(np.min(sqnr_sym))
    
    print(f" - Tổng số tensor kiểm tra               : {total}")
    print(f" - Symmetric INT8 thỏa mãn chặn |e| ≤ s/2 : {pass_bounds_sym}/{total} ({(pass_bounds_sym/total)*100:.1f}%)")
    print(f" - Asymmetric INT8 thỏa mãn chặn |e| ≤ s/2: {pass_bounds_asym}/{total} ({(pass_bounds_asym/total)*100:.1f}%)")
    print(f" - Tỷ số SQNR trung bình                  : {avg_sqnr:.2f} dB (Min: {min_sqnr:.2f} dB)")
    print(f" - Cosine Similarity trung bình           : {avg_cos:.6f} (Min: {min_cos:.6f})")
    
    status = "PASS" if pass_bounds_sym == total and avg_cos > 0.999 else "FAIL"
    print(f" => KẾT QUẢ LEVEL 1: [{status}]")
    
    return {
        "status": status,
        "total_tensors": total,
        "pass_bound_rate": pass_bounds_sym / total,
        "avg_sqnr_db": avg_sqnr,
        "min_sqnr_db": min_sqnr,
        "avg_cosine_sim": avg_cos,
        "min_cosine_sim": min_cos
    }

def level_2_layerwise_activation_verification(fp32_model_path="best.onnx", 
                                              int8_model_path="best_quantized_int8.onnx",
                                              img_path="image.png",
                                              golden_dir="golden_outputs"):
    print("\n" + "=" * 60)
    print("  LEVEL 2: SO SÁNH ACTIVATION TỪNG TẦNG (FP32 VS INT8 VS GOLDEN)")
    print("=" * 60)
    
    # Load and preprocess image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    inp = (img_resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
    
    # Check intermediate nodes
    check_nodes = [0, 3, 21, 108, 278, 365]
    node_descriptions = {
        0: "Node 0 (Conv P1 - Input Layer)",
        3: "Node 3 (Conv P2 - Backbone)",
        21: "Node 21 (Conv P3 - Backbone)",
        108: "Node 108 (Backbone Final Output)",
        278: "Node 278 (Neck Output)",
        365: "Node 365 (Head Detect 8400 Anchors)"
    }
    
    m_base = onnx.load(fp32_model_path)
    target_tensor_names = [m_base.graph.node[idx].output[0] for idx in check_nodes]
    
    def get_intermediate_outputs(model_path, target_names):
        m = onnx.load(model_path)
        existing_outs = set(o.name for o in m.graph.output)
        for name in target_names:
            if name not in existing_outs:
                vi = onnx.helper.ValueInfoProto()
                vi.name = name
                m.graph.output.append(vi)
            
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(m.SerializeToString(), sess_options, providers=['CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name
        outs = sess.run(target_names, {input_name: inp})
        return {idx: outs[i] for i, idx in enumerate(check_nodes)}

    print("[INFO] Đang chạy suy luận FP32 và INT8 trên toàn bộ intermediate nodes...")
    outs_fp32 = get_intermediate_outputs(fp32_model_path, target_tensor_names)
    outs_int8 = get_intermediate_outputs(int8_model_path, target_tensor_names)
    
    results = []
    for nid in check_nodes:
        f_act = outs_fp32[nid].flatten()
        i_act = outs_int8[nid].flatten()
        
        diff = np.abs(f_act - i_act)
        mae = float(np.mean(diff))
        max_err = float(np.max(diff))
        
        norm_prod = np.linalg.norm(f_act) * np.linalg.norm(i_act)
        cos_sim = float(np.dot(f_act, i_act) / norm_prod) if norm_prod > 0 else 1.0
        
        # Compare with golden if exists
        golden_file = os.path.join(golden_dir, f"node_{nid}_golden.txt")
        golden_mae = None
        if os.path.exists(golden_file):
            golden_data = []
            with open(golden_file, 'r') as gf:
                for line in gf:
                    golden_data.extend([float(x) for x in line.strip().split() if x])
            g_act = np.array(golden_data, dtype=np.float32)
            if len(g_act) == len(i_act):
                golden_mae = float(np.mean(np.abs(i_act - g_act)))
                
        results.append({
            "node_id": nid,
            "description": node_descriptions[nid],
            "shape": list(outs_fp32[nid].shape),
            "mae_fp32_int8": mae,
            "max_err": max_err,
            "cos_sim": cos_sim,
            "golden_mae": golden_mae
        })
        
        gold_str = f"| Golden MAE: {golden_mae:.6f}" if golden_mae is not None else ""
        print(f" -> {node_descriptions[nid]:<36} | MAE: {mae:.6f} | Cos Sim: {cos_sim:.6f} {gold_str}")
        
    avg_mae = float(np.mean([r["mae_fp32_int8"] for r in results]))
    avg_cos = float(np.mean([r["cos_sim"] for r in results]))
    status = "PASS" if avg_cos > 0.98 else "WARN"
    print(f" => KẾT QUẢ LEVEL 2: [{status}] (Average Activation Cosine Sim: {avg_cos:.6f})")
    
    return {
        "status": status,
        "layers": results,
        "avg_mae": avg_mae,
        "avg_cos_sim": avg_cos
    }

def level_3_end_to_end_detection(fp32_model_path="best.onnx",
                                int8_model_path="best_quantized_int8.onnx",
                                img_path="image.png",
                                output_vis_path="result_fp32_vs_int8.png"):
    print("\n" + "=" * 60)
    print("  LEVEL 3: KIỂM CHỨNG BOUDING BOX & IOU END-TO-END TRÊN REFERENCE IMAGE")
    print("=" * 60)
    
    img = cv2.imread(img_path)
    orig_h, orig_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    inp = (img_resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
    
    sess_fp32 = ort.InferenceSession(fp32_model_path, providers=['CPUExecutionProvider'])
    sess_int8 = ort.InferenceSession(int8_model_path, providers=['CPUExecutionProvider'])
    
    out_fp32 = sess_fp32.run(None, {'images': inp})[0][0]
    out_int8 = sess_int8.run(None, {'images': inp})[0][0]
    
    def extract_boxes(dets, conf_thresh=0.25):
        boxes = []
        scores = []
        for d in dets:
            x1, y1, x2, y2, score, cls_id = d
            if score > conf_thresh:
                boxes.append([float(x1), float(y1), float(x2), float(y2)])
                scores.append(float(score))
        return boxes, scores
        
    boxes_fp32, scores_fp32 = extract_boxes(out_fp32, 0.25)
    boxes_int8, scores_int8 = extract_boxes(out_int8, 0.25)
    
    print(f" - FP32 Detections (conf > 0.25): {len(boxes_fp32)} objects")
    for i, (b, s) in enumerate(zip(boxes_fp32, scores_fp32)):
        print(f"   [FP32 #{i+1}] bbox=({b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}), score={s:.4f}")
        
    print(f" - INT8 Detections (conf > 0.25): {len(boxes_int8)} objects")
    for i, (b, s) in enumerate(zip(boxes_int8, scores_int8)):
        print(f"   [INT8 #{i+1}] bbox=({b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}), score={s:.4f}")
        
    # Tính IoU giữa box tốt nhất của FP32 và INT8
    best_iou = 0.0
    if len(boxes_fp32) > 0 and len(boxes_int8) > 0:
        b1 = boxes_fp32[0]
        b2 = boxes_int8[0]
        
        xi1 = max(b1[0], b2[0])
        yi1 = max(b1[1], b2[1])
        xi2 = min(b1[2], b2[2])
        yi2 = min(b1[3], b2[3])
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        b1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
        b2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union_area = b1_area + b2_area - inter_area
        best_iou = inter_area / union_area if union_area > 0 else 0.0
        
    print(f" - Bounding Box IoU (FP32 vs INT8): {best_iou:.4f} ({(best_iou*100):.2f}%)")
    
    # Vẽ ảnh so sánh side-by-side
    img_fp32_vis = img.copy()
    img_int8_vis = img.copy()
    
    for b, s in zip(boxes_fp32, scores_fp32):
        rx1, ry1 = int(b[0] * orig_w / 640), int(b[1] * orig_h / 640)
        rx2, ry2 = int(b[2] * orig_w / 640), int(b[3] * orig_h / 640)
        cv2.rectangle(img_fp32_vis, (rx1, ry1), (rx2, ry2), (0, 255, 0), 3)
        cv2.putText(img_fp32_vis, f"FP32: {s:.3f}", (rx1, max(15, ry1 - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
    for b, s in zip(boxes_int8, scores_int8):
        rx1, ry1 = int(b[0] * orig_w / 640), int(b[1] * orig_h / 640)
        rx2, ry2 = int(b[2] * orig_w / 640), int(b[3] * orig_h / 640)
        cv2.rectangle(img_int8_vis, (rx1, ry1), (rx2, ry2), (0, 165, 255), 3)
        cv2.putText(img_int8_vis, f"INT8: {s:.3f}", (rx1, max(15, ry1 - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    
    combined = np.hstack([img_fp32_vis, img_int8_vis])
    cv2.imwrite(output_vis_path, combined)
    print(f"[SUCCESS] Đã lưu ảnh so sánh trực quan tại: {output_vis_path}")
    
    status = "PASS" if best_iou > 0.85 else "WARN"
    print(f" => KẾT QUẢ LEVEL 3: [{status}]")
    
    return {
        "status": status,
        "boxes_fp32": boxes_fp32,
        "scores_fp32": scores_fp32,
        "boxes_int8": boxes_int8,
        "scores_int8": scores_int8,
        "best_iou": float(best_iou)
    }

def compute_ap(recalls, precisions):
    """Compute Average Precision using 101-point interpolation or area under PR curve"""
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return float(ap)

def level_4_map_benchmark(fp32_model_path="best.onnx",
                          int8_model_path="best_quantized_int8.onnx",
                          val_img_dir="DUT_Anti_UAV_YOLO/images/val",
                          val_lbl_dir="DUT_Anti_UAV_YOLO/labels/val",
                          max_samples=50):
    print("\n" + "=" * 60)
    print("  LEVEL 4: ĐO ĐẠC SUY GIẢM mAP TRÊN TẬP VALIDATION ĐỘC LẬP")
    print("=" * 60)
    
    if not os.path.exists(val_img_dir):
        if os.path.exists("val_dataset/images"):
            val_img_dir = "val_dataset/images"
            val_lbl_dir = "val_dataset/labels"
        else:
            print(f"[ERROR] Không tìm thấy thư mục {val_img_dir}")
            return {}
        
    img_files = [f for f in os.listdir(val_img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    img_files.sort()
    if max_samples and len(img_files) > max_samples:
        img_files = img_files[:max_samples]
    
    if len(img_files) == 0:
        print(f"[ERROR] Không có ảnh nào trong {val_img_dir}")
        return {}
        
    print(f"[INFO] Bắt đầu benchmark trên {len(img_files)} ảnh validation...")
    
    sess_fp32 = ort.InferenceSession(fp32_model_path, providers=['CPUExecutionProvider'])
    sess_int8 = ort.InferenceSession(int8_model_path, providers=['CPUExecutionProvider'])
    
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    
    def evaluate_model(sess):
        # Lưu trữ detections và ground-truth cho toàn bộ tập
        all_gt_boxes = [] # list of lists of [x1, y1, x2, y2]
        all_pred_boxes = [] # list of lists of [x1, y1, x2, y2, score]
        
        for img_name in img_files:
            base_name = os.path.splitext(img_name)[0]
            lbl_file = os.path.join(val_lbl_dir, f"{base_name}.txt")
            
            # Đọc ground-truth
            gt_boxes = []
            if os.path.exists(lbl_file):
                with open(lbl_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            # YOLO format: cls, cx, cy, w, h in normalized [0, 1]
                            cx, cy, w, h = [float(x) for x in parts[1:5]]
                            # Convert to 640x640 absolute coords
                            x1 = (cx - w / 2) * 640
                            y1 = (cy - h / 2) * 640
                            x2 = (cx + w / 2) * 640
                            y2 = (cy + h / 2) * 640
                            gt_boxes.append([x1, y1, x2, y2])
            all_gt_boxes.append(gt_boxes)
            
            # Đọc ảnh và inference
            img_path = os.path.join(val_img_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                all_pred_boxes.append([])
                continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (640, 640))
            inp = (img_resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
            
            out = sess.run(None, {'images': inp})[0][0]
            # out: [300, 6] -> x1, y1, x2, y2, score, class
            preds = []
            for d in out:
                if d[4] > 0.1: # Conf threshold 0.1 for mAP
                    preds.append([float(d[0]), float(d[1]), float(d[2]), float(d[3]), float(d[4])])
            all_pred_boxes.append(preds)
            
        # Tính mAP@0.5 và mAP@0.5:0.95
        ap_per_iou = []
        precisions_50 = 0
        recalls_50 = 0
        
        for iou_thresh in iou_thresholds:
            tp_list = []
            fp_list = []
            scores_list = []
            n_positives = sum(len(gt) for gt in all_gt_boxes)
            
            for gts, preds in zip(all_gt_boxes, all_pred_boxes):
                matched = [False] * len(gts)
                # Sort preds by score descending
                preds_sorted = sorted(preds, key=lambda x: x[4], reverse=True)
                
                for p in preds_sorted:
                    scores_list.append(p[4])
                    best_iou = 0.0
                    best_idx = -1
                    for gi, g in enumerate(gts):
                        xi1 = max(p[0], g[0])
                        yi1 = max(p[1], g[1])
                        xi2 = min(p[2], g[2])
                        yi2 = min(p[3], g[3])
                        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                        union = (p[2] - p[0]) * (p[3] - p[1]) + (g[2] - g[0]) * (g[3] - g[1]) - inter
                        iou = inter / union if union > 0 else 0
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = gi
                            
                    if best_iou >= iou_thresh and best_idx >= 0 and not matched[best_idx]:
                        tp_list.append(1)
                        fp_list.append(0)
                        matched[best_idx] = True
                    else:
                        tp_list.append(0)
                        fp_list.append(1)
                        
            if n_positives == 0 or len(tp_list) == 0:
                ap_per_iou.append(0.0)
                continue
                
            tp_cumsum = np.cumsum(tp_list)
            fp_cumsum = np.cumsum(fp_list)
            recalls = tp_cumsum / n_positives
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
            ap = compute_ap(recalls, precisions)
            ap_per_iou.append(ap)
            
            if abs(iou_thresh - 0.5) < 1e-4:
                precisions_50 = float(precisions[-1]) if len(precisions) > 0 else 0.0
                recalls_50 = float(recalls[-1]) if len(recalls) > 0 else 0.0
                
        map50 = float(ap_per_iou[0])
        map50_95 = float(np.mean(ap_per_iou))
        
        f1 = (2 * precisions_50 * recalls_50 / (precisions_50 + recalls_50)) if (precisions_50 + recalls_50) > 0 else 0.0
        
        return {
            "mAP50": map50,
            "mAP50_95": map50_95,
            "precision": precisions_50,
            "recall": recalls_50,
            "f1": f1
        }
        
    print("[INFO] Đang đánh giá mAP mô hình FP32 Baseline...")
    metrics_fp32 = evaluate_model(sess_fp32)
    print(f" -> FP32   : mAP@50 = {metrics_fp32['mAP50']*100:.2f}% | mAP@50:95 = {metrics_fp32['mAP50_95']*100:.2f}% | Precision = {metrics_fp32['precision']*100:.2f}% | Recall = {metrics_fp32['recall']*100:.2f}%")
    
    print("[INFO] Đang đánh giá mAP mô hình INT8 Quantized...")
    metrics_int8 = evaluate_model(sess_int8)
    print(f" -> INT8   : mAP@50 = {metrics_int8['mAP50']*100:.2f}% | mAP@50:95 = {metrics_int8['mAP50_95']*100:.2f}% | Precision = {metrics_int8['precision']*100:.2f}% | Recall = {metrics_int8['recall']*100:.2f}%")
    
    delta_map50 = (metrics_fp32["mAP50"] - metrics_int8["mAP50"]) * 100
    delta_map50_95 = (metrics_fp32["mAP50_95"] - metrics_int8["mAP50_95"]) * 100
    
    print("-" * 60)
    print(f" => SỤT GIẢM mAP@50    : ΔmAP@50    = {delta_map50:+.2f}% ({'Tốt, suy giảm không đáng kể' if abs(delta_map50) < 2.0 else 'Cần lưu ý'})")
    print(f" => SỤT GIẢM mAP@50:95 : ΔmAP@50:95 = {delta_map50_95:+.2f}%")
    print("=" * 60)
    
    return {
        "fp32": metrics_fp32,
        "int8": metrics_int8,
        "delta_map50_pct": float(delta_map50),
        "delta_map50_95_pct": float(delta_map50_95)
    }

def main():
    print("=" * 60)
    print("  HỆ THỐNG KIỂM CHỨNG TOÀN DIỆN 4 CẤP ĐỘ (VERIFICATION ENGINE)")
    print("=" * 60)
    
    report = {}
    report["level_1"] = level_1_numerical_verification()
    report["level_2"] = level_2_layerwise_activation_verification()
    report["level_3"] = level_3_end_to_end_detection()
    report["level_4"] = level_4_map_benchmark()
    
    report_file = "verification_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[SUCCESS] Đã lưu báo cáo kiểm chứng chi tiết tại: {report_file}")

if __name__ == "__main__":
    main()
