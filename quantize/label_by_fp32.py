import os
from ultralytics import YOLO

# 1. Dùng model FP32 gốc để sinh file nhãn (.txt) tự động cho tập ảnh
model_pt = YOLO("C:/Users/PC/Downloads/best.pt") #thêm đường dẫn model để label

# Chạy predict và lưu kết quả dạng file txt nhãn
model_pt.predict(
    source="C:/Users/PC/Downloads/my_data_valid/images/", # Thư mục chứa ảnh
    save_txt=True,                      # Lưu nhãn .txt
    save_conf=True,                     # Lưu confidence
    project="C:/Users/PC/Downloads/my_data_valid/",
    name="labels",
    exist_ok=True
)

print("[✓] Đã tạo xong tập nhãn pseudo-labels từ mô hình FP32!")