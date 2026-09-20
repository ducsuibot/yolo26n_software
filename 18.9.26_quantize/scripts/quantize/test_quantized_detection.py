#!/usr/bin/env python3
"""
Test Object Detection on image.png with Quantized INT8 Model vs FP32 Baseline
"""

import os
import cv2
import numpy as np
import onnxruntime as ort

def run_detection(model_path, inp_tensor, orig_shape, conf_thresh=0.25):
    orig_h, orig_w = orig_shape
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: inp_tensor})[0][0]
    
    boxes = []
    scores = []
    classes = []
    
    for row in out:
        x1, y1, x2, y2, score, cls_id = row
        if score > conf_thresh:
            rx1 = float(x1 * orig_w / 640.0)
            ry1 = float(y1 * orig_h / 640.0)
            rx2 = float(x2 * orig_w / 640.0)
            ry2 = float(y2 * orig_h / 640.0)
            boxes.append([rx1, ry1, rx2, ry2])
            scores.append(float(score))
            classes.append(int(cls_id))
            
    return boxes, scores, classes

def calculate_iou(b1, b2):
    xi1 = max(b1[0], b2[0])
    yi1 = max(b1[1], b2[1])
    xi2 = min(b1[2], b2[2])
    yi2 = min(b1[3], b2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def main():
    img_path = "image.png"
    fp32_model = "best.onnx"
    int8_model = "best_quantized_int8.onnx"
    
    if not os.path.exists(img_path):
        fallback = "DUT_Anti_UAV_YOLO/images/val/00068.jpg"
        if os.path.exists(fallback):
            img_path = fallback
        else:
            print(f"[ERROR] Không tìm thấy {img_path}")
            return
            
    if not os.path.exists(int8_model) and os.path.exists(os.path.join("quantize", int8_model)):
        int8_model = os.path.join("quantize", int8_model)
    if not os.path.exists(fp32_model) and os.path.exists(os.path.join("..", fp32_model)):
        fp32_model = os.path.join("..", fp32_model)
        
    img = cv2.imread(img_path)
    orig_h, orig_w = img.shape[:2]
    
    # Preprocessing (letterbox / direct resize 640x640)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640)).astype(np.float32) / 255.0
    inp = np.transpose(img_resized, (2, 0, 1))[None, ...]
    
    # Run FP32 Inference
    boxes_fp32, scores_fp32, cls_fp32 = run_detection(fp32_model, inp, (orig_h, orig_w), conf_thresh=0.25)
    
    # Run INT8 Inference
    boxes_int8, scores_int8, cls_int8 = run_detection(int8_model, inp, (orig_h, orig_w), conf_thresh=0.25)
    
    print("=" * 65)
    print("  KẾT QUẢ NHẬN DIỆN TRÊN ẢNH: " + img_path)
    print(f"  Kích thước ảnh gốc: {orig_w} x {orig_h} | Model input: 640 x 640")
    print("=" * 65)
    
    print(f"\n[1] MÔ HÌNH GỐC FP32 ({fp32_model}):")
    print(f"    Số lượng đối tượng phát hiện: {len(boxes_fp32)}")
    for i, (b, s, c) in enumerate(zip(boxes_fp32, scores_fp32, cls_fp32)):
        print(f"    #{i+1}: BBox = [{b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}] | Confidence = {s*100:.2f}% | Class = {c}")
        
    print(f"\n[2] MÔ HÌNH SAU LƯỢNG TỬ HÓA INT8 ({int8_model}):")
    print(f"    Số lượng đối tượng phát hiện: {len(boxes_int8)}")
    for i, (b, s, c) in enumerate(zip(boxes_int8, scores_int8, cls_int8)):
        print(f"    #{i+1}: BBox = [{b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}] | Confidence = {s*100:.2f}% | Class = {c}")
        
    # Tính toán sai lệch giữa FP32 và INT8
    if len(boxes_fp32) > 0 and len(boxes_int8) > 0:
        b_f = boxes_fp32[0]
        b_i = boxes_int8[0]
        iou = calculate_iou(b_f, b_i)
        dx1 = abs(b_f[0] - b_i[0])
        dy1 = abs(b_f[1] - b_i[1])
        dx2 = abs(b_f[2] - b_i[2])
        dy2 = abs(b_f[3] - b_i[3])
        dscore = abs(scores_fp32[0] - scores_int8[0]) * 100
        
        print("\n" + "-" * 65)
        print("  ĐÁNH GIÁ ĐỘ CHÍNH XÁC INT8 SO VỚI FP32:")
        print("-" * 65)
        print(f"  - Chỉ số IoU (Intersection over Union) : {iou*100:.2f}%")
        print(f"  - Độ lệch tọa độ góc trên bên trái (x1, y1): Δx1 = {dx1:.2f} px, Δy1 = {dy1:.2f} px")
        print(f"  - Độ lệch tọa độ góc dưới bên phải (x2, y2): Δx2 = {dx2:.2f} px, Δy2 = {dy2:.2f} px")
        print(f"  - Độ lệch Confidence Score                 : ΔScore = {dscore:.2f}% (FP32: {scores_fp32[0]*100:.2f}% vs INT8: {scores_int8[0]*100:.2f}%)")
        print("-" * 65)
        
    # Vẽ ảnh kết quả độc lập cho INT8
    img_int8 = img.copy()
    for b, s, c in zip(boxes_int8, scores_int8, cls_int8):
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        # Vẽ Bounding Box (Màu cam nổi bật cho INT8)
        cv2.rectangle(img_int8, (x1, y1), (x2, y2), (0, 165, 255), 2)
        label = f"UAV INT8: {s*100:.1f}%"
        
        # Background cho label
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_int8, (x1, max(0, y1 - th - 10)), (x1 + tw + 6, y1), (0, 165, 255), -1)
        cv2.putText(img_int8, label, (x1 + 3, max(th + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Tạo khung Zoom-in nhỏ góc phải màn hình
        h_crop = max(10, y2 - y1)
        w_crop = max(10, x2 - x1)
        pad_y = int(h_crop * 0.5)
        pad_x = int(w_crop * 0.5)
        cy1, cy2 = max(0, y1 - pad_y), min(orig_h, y2 + pad_y)
        cx1, cx2 = max(0, x1 - pad_x), min(orig_w, x2 + pad_x)
        crop = img[cy1:cy2, cx1:cx2]
        if crop.shape[0] > 0 and crop.shape[1] > 0:
            zoom_size = 180
            zoom_crop = cv2.resize(crop, (zoom_size, zoom_size))
            cv2.rectangle(zoom_crop, (0, 0), (zoom_size - 1, zoom_size - 1), (0, 165, 255), 3)
            cv2.putText(zoom_crop, "Zoom In (INT8)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            # Đặt vào góc trên bên phải của ảnh
            pos_y = 20
            pos_x = orig_w - zoom_size - 20
            img_int8[pos_y:pos_y + zoom_size, pos_x:pos_x + zoom_size] = zoom_crop
            
    out_single_path = "result_quantized_detection.jpg"
    cv2.imwrite(out_single_path, img_int8)
    print(f"\n[SUCCESS] Đã lưu ảnh kết quả nhận diện INT8 tại: {out_single_path}")
    
    # Vẽ ảnh so sánh side-by-side (FP32 vs INT8)
    img_fp32 = img.copy()
    for b, s, c in zip(boxes_fp32, scores_fp32, cls_fp32):
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        cv2.rectangle(img_fp32, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"UAV FP32: {s*100:.1f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_fp32, (x1, max(0, y1 - th - 10)), (x1 + tw + 6, y1), (0, 255, 0), -1)
        cv2.putText(img_fp32, label, (x1 + 3, max(th + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Thêm tiêu đề cho 2 ảnh
    cv2.putText(img_fp32, "BASELINE FP32 MODEL", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(img_int8, "QUANTIZED INT8 MODEL", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
    
    # Resize ghép side-by-side
    scale = 0.75
    h_s, w_s = int(orig_h * scale), int(orig_w * scale)
    vis_fp32 = cv2.resize(img_fp32, (w_s, h_s))
    vis_int8 = cv2.resize(img_int8, (w_s, h_s))
    combined = np.hstack([vis_fp32, vis_int8])
    
    out_comparison_path = "comparison_fp32_vs_quantized.png"
    cv2.imwrite(out_comparison_path, combined)
    print(f"[SUCCESS] Đã lưu ảnh so sánh trực quan FP32 vs INT8 tại: {out_comparison_path}")
    print("=" * 65)

if __name__ == "__main__":
    main()
