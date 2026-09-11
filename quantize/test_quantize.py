from ultralytics import YOLO

# 1. Nạp file ONNX INT8 vừa export
model = YOLO("C:/Users/PC/Downloads/best (2)_int8.onnx")

# 2. Chạy thử suy luận trên 1 tấm ảnh (kết quả sẽ lưu trong folder runs/detect/predict)
# results = model.predict(
#     source="C:/Users/PC/Downloads/test/test/img/02194.jpg", 
#     save=True, 
#     conf=0.25
# )

# 3. Đánh giá lại mAP trên tập validation để xem có bị sụt giảm độ chính xác không
metrics = model.val(data="C:/Users/PC/Downloads/dataset_calib.yaml")
print("mAP50-95 của bản INT8: ",metrics.box.map)