- (code c không dùng lib)
- node: gồm tòa bộ định nghĩa node trong model yolo26n
- module: tổng 24 module, mỗi module gồm các node
- cơ chế: load , comput, store thông qua địa chỉ 2 DRAM
- dram.csv
- DRAM1: lưu weight (Địa chỉ phân cấp)
- DRAM2: lưu toàn bộ tensor ofm (Địa chỉ phân cấp)

Cách chạy: 
- clone github về gõ terminal : ./demo
- sau đó, code c sẽ verify toàn bộ hơn 300 node node với sai số < 0.03 và in output toàn bộ node của backbone/neck/head sang .txt
- chạy "run" print_boudingbox_c.py để hiện boudingbox kết quả:
- Node 365: mảng 1x8400x5 gồm [x1, y1, x2, y2, score]) và vẽ lên ảnh gốc image.png
![Result Detection](https://raw.githubusercontent.com/ducsuibot/yolo26n_software/main/result_detected.jpg)
