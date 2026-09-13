#!/usr/bin/env python3
"""
Evaluation Engine: Đo và so sánh mAP của mô hình YOLOv26n FP32 vs Quantized INT8
Dataset: DUT_Anti_UAV_YOLO (100 ảnh mẫu)
Phương pháp: Đánh giá hiệu năng thô (Direct Resize 640x640, không thêm padding/letterbox)
Target: Khảo sát triển khai FPGA ZCU104 (UAV Tracking)
"""

import os
import glob
import time
import json
import argparse
import cv2
import torch
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics.utils.metrics import box_iou, ap_per_class

def load_ground_truth(label_path, orig_w, orig_h):
    """Đọc nhãn ground truth format YOLO: class_id xc yc bw bh (chuẩn hóa [0, 1])"""
    boxes = []
    classes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(float(parts[0]))
                    xc, yc, bw, bh = map(float, parts[1:5])
                    x1 = (xc - bw / 2.0) * orig_w
                    y1 = (yc - bh / 2.0) * orig_h
                    x2 = (xc + bw / 2.0) * orig_w
                    y2 = (yc + bh / 2.0) * orig_h
                    boxes.append([x1, y1, x2, y2])
                    classes.append(cls_id)
    return boxes, classes

def run_evaluation(model_path, image_paths, label_dir, conf_thresh=0.001):
    """
    Chạy suy luận và tính toán ma trận kết quả (True Positive theo 10 ngưỡng IoU 0.50:0.05:0.95)
    Tiền xử lý: Direct Resize 640x640 chuẩn hóa / 255.0 (đúng theo pipeline thô phần cứng)
    """
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    
    # 10 ngưỡng IoU từ 0.50 đến 0.95
    iouv = torch.linspace(0.5, 0.95, 10)
    
    tp_list = []
    conf_list = []
    pred_cls_list = []
    target_cls_list = []
    
    inference_times = []
    total_gt_count = 0
    total_det_count = 0
    
    # Warmup
    dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
    sess.run(None, {input_name: dummy})
    
    model_desc = os.path.basename(model_path)
    pbar = tqdm(image_paths, desc=f"Eval {model_desc:<25}", unit="img")
    for img_path in pbar:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(label_dir, f"{base_name}.txt")
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        orig_h, orig_w = img.shape[:2]
        
        # 1. Ground Truth
        gt_boxes, gt_classes = load_ground_truth(lbl_path, orig_w, orig_h)
        total_gt_count += len(gt_classes)
        target_cls_list.extend(gt_classes)
        
        gt_boxes_t = torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes else torch.zeros((0, 4))
        gt_classes_t = torch.tensor(gt_classes, dtype=torch.int64) if gt_classes else torch.zeros((0,), dtype=torch.int64)
        
        # 2. Tiền xử lý thô: Direct Resize 640x640
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (640, 640), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        inp = np.transpose(img_resized, (2, 0, 1))[None, ...]
        
        # 3. Đo thời gian suy luận
        t_start = time.perf_counter()
        out = sess.run(None, {input_name: inp})[0][0] # Shape: (300, 6) -> [x1, y1, x2, y2, score, cls]
        t_end = time.perf_counter()
        inference_times.append((t_end - t_start) * 1000.0) # ms
        
        # 4. Lọc dự đoán theo conf_thresh
        valid_mask = out[:, 4] > conf_thresh
        preds = out[valid_mask]
        total_det_count += len(preds)
        
        if len(preds) > 0:
            pred_boxes = []
            for row in preds:
                x1, y1, x2, y2 = row[:4]
                # Scale trực tiếp từ 640x640 về ảnh gốc
                rx1 = float(x1 * orig_w / 640.0)
                ry1 = float(y1 * orig_h / 640.0)
                rx2 = float(x2 * orig_w / 640.0)
                ry2 = float(y2 * orig_h / 640.0)
                pred_boxes.append([rx1, ry1, rx2, ry2])
                
            pred_boxes_t = torch.tensor(pred_boxes, dtype=torch.float32)
            pred_scores = preds[:, 4]
            pred_classes_t = torch.tensor(preds[:, 5], dtype=torch.int64)
            
            conf_list.extend(pred_scores.tolist())
            pred_cls_list.extend(pred_classes_t.tolist())
            
            # 5. Khớp IoU và đánh giá True Positive
            if len(gt_classes_t) > 0:
                iou = box_iou(gt_boxes_t, pred_boxes_t)
                correct = np.zeros((pred_boxes_t.shape[0], 10), dtype=bool)
                correct_class = (gt_classes_t[:, None] == pred_classes_t).numpy()
                iou_np = (iou.numpy() * correct_class)
                
                for i_iou, th in enumerate(iouv.numpy()):
                    matches = np.nonzero(iou_np >= th)
                    matches = np.array(matches).T
                    if matches.shape[0]:
                        if matches.shape[0] > 1:
                            matches = matches[iou_np[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                        correct[matches[:, 1].astype(int), i_iou] = True
                tp_list.append(correct)
            else:
                tp_list.append(np.zeros((pred_boxes_t.shape[0], 10), dtype=bool))
                
    tp = np.concatenate(tp_list, axis=0) if tp_list else np.zeros((0, 10), dtype=bool)
    conf = np.array(conf_list)
    pred_cls = np.array(pred_cls_list)
    target_cls = np.array(target_cls_list)
    
    # Tính toán AP, Precision, Recall chuẩn COCO
    results = ap_per_class(tp, conf, pred_cls, target_cls, names={0: 'uav'}, plot=False)
    # results: (tp, fp, p, r, f1, ap, unique_classes)
    p = float(results[2][0]) if len(results[2]) > 0 else 0.0
    r = float(results[3][0]) if len(results[3]) > 0 else 0.0
    f1 = float(results[4][0]) if len(results[4]) > 0 else 0.0
    ap50 = float(results[5][0, 0]) if len(results[5]) > 0 else 0.0
    ap50_95 = float(results[5][0].mean()) if len(results[5]) > 0 else 0.0
    
    mean_lat = float(np.mean(inference_times))
    fps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0
    file_size_mb = float(os.path.getsize(model_path) / (1024 * 1024))
    
    return {
        "model_name": os.path.basename(model_path),
        "model_path": model_path,
        "file_size_mb": file_size_mb,
        "total_images": len(image_paths),
        "total_gt": total_gt_count,
        "total_detections": total_det_count,
        "precision": p,
        "recall": r,
        "f1": f1,
        "map50": ap50,
        "map50_95": ap50_95,
        "latency_ms": mean_lat,
        "fps": fps,
        "results_raw": results
    }

def main():
    parser = argparse.ArgumentParser(description="Đo và so sánh mAP của mô hình YOLOv26n FP32 vs Quantized INT8")
    parser.add_argument("--dataset-dir", type=str, default="DUT_Anti_UAV_YOLO", help="Đường dẫn thư mục dataset")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train", "all"], help="Tập split cần đánh giá (mặc định: val)")
    parser.add_argument("--num-samples", type=int, default=-1, help="Số lượng ảnh cần đánh giá (-1 là toàn bộ ảnh)")
    parser.add_argument("--fp32-model", type=str, default="best.onnx", help="Đường dẫn model FP32 ONNX")
    parser.add_argument("--int8-model", type=str, default="best_quantized_int8.onnx", help="Đường dẫn model INT8 ONNX")
    parser.add_argument("--conf-thresh", type=float, default=0.001, help="Ngưỡng confidence để tính mAP")
    args = parser.parse_args()

    dataset_img_dir = os.path.join(args.dataset_dir, "images", args.split)
    dataset_lbl_dir = os.path.join(args.dataset_dir, "labels", args.split)
    
    fp32_model_path = args.fp32_model
    int8_model_path = args.int8_model
    
    # Lấy danh sách ảnh
    if args.split == "all":
        all_imgs = []
        for s in ["val", "test", "train"]:
            s_dir = os.path.join(args.dataset_dir, "images", s)
            all_imgs.extend(sorted(glob.glob(os.path.join(s_dir, "*.jpg"))))
    else:
        all_imgs = sorted(glob.glob(os.path.join(dataset_img_dir, "*.jpg")))

    if len(all_imgs) == 0:
        print(f"[ERROR] Không tìm thấy ảnh trong {dataset_img_dir}")
        return
        
    if args.num_samples is not None and args.num_samples > 0:
        eval_imgs = all_imgs[:args.num_samples]
    else:
        eval_imgs = all_imgs
        
    split_name = args.split if args.split != "all" else "toàn bộ (train+val+test)"
    print("=" * 75)
    print(f"  KHẢO SÁT HIỆU NĂNG THÔ MÔ HÌNH YOLOV26N TRÊN FPGA ZCU104")
    print(f"  Dataset: DUT_Anti_UAV_YOLO ({split_name}) | Số lượng ảnh: {len(eval_imgs)}")
    print(f"  Tiền xử lý: Direct Resize 640x640 (Không thêm padding/letterbox)")
    print("=" * 75)
    
    # 1. Đánh giá Baseline FP32
    print(f"\n[1/2] Đang đánh giá mô hình Baseline FP32 ({fp32_model_path})...")
    res_fp32 = run_evaluation(fp32_model_path, eval_imgs, dataset_lbl_dir, conf_thresh=args.conf_thresh)
    print(f"      -> FP32 Hoàn tất: mAP50 = {res_fp32['map50']*100:.2f}%, mAP50-95 = {res_fp32['map50_95']*100:.2f}%, Latency = {res_fp32['latency_ms']:.2f}ms")
    
    # 2. Đánh giá Quantized INT8
    print(f"\n[2/2] Đang đánh giá mô hình Quantized INT8 ({int8_model_path})...")
    res_int8 = run_evaluation(int8_model_path, eval_imgs, dataset_lbl_dir, conf_thresh=args.conf_thresh)
    print(f"      -> INT8 Hoàn tất: mAP50 = {res_int8['map50']*100:.2f}%, mAP50-95 = {res_int8['map50_95']*100:.2f}%, Latency = {res_int8['latency_ms']:.2f}ms")
    
    # 3. Tính toán độ chênh lệch (Delta)
    size_reduction = (1.0 - res_int8["file_size_mb"] / res_fp32["file_size_mb"]) * 100.0
    delta_p = (res_int8["precision"] - res_fp32["precision"]) * 100.0
    delta_r = (res_int8["recall"] - res_fp32["recall"]) * 100.0
    delta_f1 = (res_int8["f1"] - res_fp32["f1"]) * 100.0
    delta_map50 = (res_int8["map50"] - res_fp32["map50"]) * 100.0
    delta_map50_95 = (res_int8["map50_95"] - res_fp32["map50_95"]) * 100.0
    
    # 4. In bảng so sánh chi tiết
    print("\n" + "=" * 78)
    print("                    BẢNG SO SÁNH HIỆU NĂNG: FP32 VS QUANTIZED INT8")
    print("=" * 78)
    print(f"{'Chỉ số đánh giá (Metric)':<30} | {'Baseline (FP32)':<15} | {'Quantized (INT8)':<16} | {'Chênh lệch (Delta)':<12}")
    print("-" * 78)
    print(f"{'Dung lượng mô hình (Model Size)':<30} | {res_fp32['file_size_mb']:>11.2f} MB | {res_int8['file_size_mb']:>12.2f} MB | {-size_reduction:>+10.1f} %")
    print(f"{'Tổng số Ground Truth UAV':<30} | {res_fp32['total_gt']:>14} | {res_int8['total_gt']:>15} | {'-':>12}")
    print(f"{'Tổng số Box phát hiện':<30} | {res_fp32['total_detections']:>14} | {res_int8['total_detections']:>15} | {res_int8['total_detections'] - res_fp32['total_detections']:>+12}")
    print(f"{'Precision (P)':<30} | {res_fp32['precision']*100:>13.2f} % | {res_int8['precision']*100:>14.2f} % | {delta_p:>+10.2f} %")
    print(f"{'Recall (R)':<30} | {res_fp32['recall']*100:>13.2f} % | {res_int8['recall']*100:>14.2f} % | {delta_r:>+10.2f} %")
    print(f"{'F1-Score':<30} | {res_fp32['f1']:>14.4f} | {res_int8['f1']:>15.4f} | {delta_f1/100.0:>+12.4f}")
    print(f"{'mAP@0.5 (mAP50)':<30} | {res_fp32['map50']*100:>13.2f} % | {res_int8['map50']*100:>14.2f} % | {delta_map50:>+10.2f} %")
    print(f"{'mAP@0.5:0.95 (mAP50-95)':<30} | {res_fp32['map50_95']*100:>13.2f} % | {res_int8['map50_95']*100:>14.2f} % | {delta_map50_95:>+10.2f} %")
    print(f"{'Inference Latency (CPU avg)':<30} | {res_fp32['latency_ms']:>11.2f} ms | {res_int8['latency_ms']:>12.2f} ms | {res_int8['latency_ms'] - res_fp32['latency_ms']:>+10.2f} ms")
    print(f"{'Throughput (FPS)':<30} | {res_fp32['fps']:>11.1f} fps| {res_int8['fps']:>12.1f} fps| {res_int8['fps'] - res_fp32['fps']:>+10.1f} fps")
    print("=" * 78)
    
    # 5. Lưu kết quả ra JSON và Markdown
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": f"{args.dataset_dir} ({split_name})",
        "num_images_evaluated": len(eval_imgs),
        "preprocessing": "direct_resize_640x640",
        "fp32": {
            "model": res_fp32["model_name"],
            "size_mb": res_fp32["file_size_mb"],
            "precision": res_fp32["precision"],
            "recall": res_fp32["recall"],
            "f1": res_fp32["f1"],
            "map50": res_fp32["map50"],
            "map50_95": res_fp32["map50_95"],
            "latency_ms": res_fp32["latency_ms"],
            "fps": res_fp32["fps"]
        },
        "int8": {
            "model": res_int8["model_name"],
            "size_mb": res_int8["file_size_mb"],
            "precision": res_int8["precision"],
            "recall": res_int8["recall"],
            "f1": res_int8["f1"],
            "map50": res_int8["map50"],
            "map50_95": res_int8["map50_95"],
            "latency_ms": res_int8["latency_ms"],
            "fps": res_int8["fps"]
        },
        "delta": {
            "size_reduction_pct": size_reduction,
            "delta_precision_pct": delta_p,
            "delta_recall_pct": delta_r,
            "delta_f1": delta_f1 / 100.0,
            "delta_map50_pct": delta_map50,
            "delta_map50_95_pct": delta_map50_95
        }
    }
    
    json_path = "map_comparison_report.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"\n[SUCCESS] Đã lưu báo cáo chi tiết dạng JSON tại: {json_path}")
    
    md_path = "map_comparison_report.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Báo cáo đo lường mAP: YOLOv26n FP32 vs Quantized INT8\n\n")
        f.write(f"- **Tập dữ liệu**: DUT_Anti_UAV_YOLO ({split_name}, {len(eval_imgs)} ảnh)\n")
        f.write(f"- **Phương pháp tiền xử lý**: Direct Resize 640x640 (Hiệu năng thô cho FPGA)\n")
        f.write(f"- **Mục tiêu**: Khảo sát triển khai phần cứng FPGA ZCU104 ứng dụng UAV Tracking\n\n")
        f.write("### Bảng kết quả so sánh định lượng\n\n")
        f.write("| Chỉ số đánh giá | Baseline (FP32) | Quantized (INT8) | Chênh lệch (Delta) |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **Dung lượng mô hình** | {res_fp32['file_size_mb']:.2f} MB | {res_int8['file_size_mb']:.2f} MB | **-{size_reduction:.1f}%** |\n")
        f.write(f"| **Precision (P)** | {res_fp32['precision']*100:.2f}% | {res_int8['precision']*100:.2f}% | {delta_p:+.2f}% |\n")
        f.write(f"| **Recall (R)** | {res_fp32['recall']*100:.2f}% | {res_int8['recall']*100:.2f}% | {delta_r:+.2f}% |\n")
        f.write(f"| **F1-Score** | {res_fp32['f1']:.4f} | {res_int8['f1']:.4f} | {delta_f1/100.0:+.4f} |\n")
        f.write(f"| **mAP@0.5 (mAP50)** | {res_fp32['map50']*100:.2f}% | {res_int8['map50']*100:.2f}% | **{delta_map50:+.2f}%** |\n")
        f.write(f"| **mAP@0.5:0.95 (mAP50-95)** | {res_fp32['map50_95']*100:.2f}% | {res_int8['map50_95']*100:.2f}% | **{delta_map50_95:+.2f}%** |\n")
        f.write(f"| **Inference Latency (CPU)** | {res_fp32['latency_ms']:.2f} ms | {res_int8['latency_ms']:.2f} ms | {res_int8['latency_ms'] - res_fp32['latency_ms']:+.2f} ms |\n")
        f.write(f"| **Throughput (FPS)** | {res_fp32['fps']:.1f} FPS | {res_int8['fps']:.1f} FPS | {res_int8['fps'] - res_fp32['fps']:+.1f} FPS |\n\n")
        f.write("### Nhận xét & Đánh giá triển khai FPGA ZCU104\n")
        f.write(f"1. **Bảo toàn độ chính xác phát hiện**: mAP@0.5 của mô hình INT8 đạt **{res_int8['map50']*100:.2f}%**, độ suy giảm chỉ **{abs(delta_map50):.2f}%** so với FP32.\n")
        f.write(f"2. **Độ chính xác vị trí bounding box**: mAP@0.5:0.95 đạt **{res_int8['map50_95']*100:.2f}%** (chênh lệch {delta_map50_95:+.2f}%).\n")
        f.write(f"3. **Tối ưu bộ nhớ on-chip**: Giảm kích thước trọng số **{size_reduction:.1f}%** (từ {res_fp32['file_size_mb']:.2f} MB xuống {res_int8['file_size_mb']:.2f} MB), cực kỳ thuận lợi để nạp và tính toán trên BRAM/URAM và DRAM1 của FPGA ZCU104.\n")
    print(f"[SUCCESS] Đã lưu báo cáo Markdown tại: {md_path}")
    
    # 6. Vẽ biểu đồ so sánh cột (mAP, Precision, Recall)
    try:
        metrics_names = ['Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95']
        fp32_vals = [res_fp32['precision']*100, res_fp32['recall']*100, res_fp32['map50']*100, res_fp32['map50_95']*100]
        int8_vals = [res_int8['precision']*100, res_int8['recall']*100, res_int8['map50']*100, res_int8['map50_95']*100]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        plt.figure(figsize=(9, 5.5))
        plt.bar(x - width/2, fp32_vals, width, label='Baseline FP32', color='#1f77b4', edgecolor='black', alpha=0.85)
        plt.bar(x + width/2, int8_vals, width, label='Quantized INT8', color='#ff7f0e', edgecolor='black', alpha=0.85)
        
        for i in range(len(metrics_names)):
            plt.text(x[i] - width/2, fp32_vals[i] + 1.0, f"{fp32_vals[i]:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
            plt.text(x[i] + width/2, int8_vals[i] + 1.0, f"{int8_vals[i]:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
            
        plt.ylabel('Phần trăm (%)', fontsize=11)
        plt.title(f'So sánh độ chính xác YOLOv26n: FP32 Baseline vs Quantized INT8\n(DUT_Anti_UAV_YOLO - {len(eval_imgs)} ảnh {split_name}, Direct Resize)', fontsize=12, fontweight='bold')
        plt.xticks(x, metrics_names, fontsize=10)
        plt.ylim(0, 115)
        plt.legend(loc='upper right', frameon=True)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        plot_path = "map_comparison_chart.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"[SUCCESS] Đã lưu biểu đồ so sánh cột tại: {plot_path}")
    except Exception as e:
        print(f"[WARNING] Không thể tạo biểu đồ: {e}")

if __name__ == "__main__":
    main()
