import numpy as np
import cv2
import os

def load_tensor_from_txt(filepath, shape):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.extend([float(x) for x in line.split()])
    return np.array(data, dtype=np.float32).reshape(shape)

def main():
    # Đường dẫn file output node 365 (shape: 1 x 8400 x 5) từ C engine
    node_365_path = "output_nodes/head/node_365_output.txt"
    img_path = "image.png"

    if not os.path.exists(node_365_path):
        print(f"[ERROR] Không tìm thấy file {node_365_path}. Hãy chạy file C trước!")
        return

    print("[INFO] Đang đọc tensor từ node_365_output.txt...")
    tensor_data = load_tensor_from_txt(node_365_path, (1, 8400, 5))[0]  # Shape: (8400, 5)

    # Đọc ảnh gốc để vẽ box
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Không đọc được ảnh từ {img_path}")
        return
    
    orig_h, orig_w = img.shape[:2]

    # Ngưỡng lọcConfidence Score
    conf_threshold = 0.25
    boxes = []
    scores = []

    for row in tensor_data:
        x1, y1, x2, y2, score = row
        if score > conf_threshold:
            boxes.append([x1, y1, x2, y2])
            scores.append(float(score))

    # Chuyển đổi sang định dạng cho OpenCV NMSBoxes [x_min, y_min, width, height]
    cv_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        # Scale tọa độ từ khung hình 640x640 về kích thước ảnh gốc
        rx1 = int(x1 * orig_w / 640)
        ry1 = int(y1 * orig_h / 640)
        rx2 = int(x2 * orig_w / 640)
        ry2 = int(y2 * orig_h / 640)
        cv_boxes.append([rx1, ry1, rx2 - rx1, ry2 - ry1])

    # Áp dụng Non-Maximum Suppression (NMS) để loại bỏ bớt các box trùng lặp
    indices = cv2.dnn.NMSBoxes(cv_boxes, scores, score_threshold=conf_threshold, nms_threshold=0.45)

    print(f"[INFO] Tìm thấy {len(indices) if len(indices) > 0 else 0} đối tượng sau NMS.")

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = cv_boxes[i]
            score = scores[i]
            # Vẽ hình chữ nhật lên ảnh
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f"Obj: {score:.2f}", (x, max(y - 10, 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    output_img_path = "result_detected.jpg"
    cv2.imwrite(output_img_path, img)
    print(f"[SUCCESS] Đã lưu ảnh kết quả có Bounding Box tại: {output_img_path}")

if __name__ == "__main__":
    main()