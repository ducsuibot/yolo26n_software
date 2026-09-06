- node: gồm tòa bộ định nghĩa code trong model yolo26n
- module: tổng 24 module, mỗi module gồm các node
- cơ chế: load , comput, store thông qua địa chỉ 2 DRAM
- DRAM1: lưu weight
- DRAM2: lưu toàn bộ tensor ofm

Cách chạy: 
- clone github về gõ terminal : ./demo
- sau đó, code c sẽ verify toàn bộ 382 node với sai số < 0.03 và in output toàn bộ 382node sang .txt
- chạy "run" print_boudingbox_c.py để hiện boudingbox kết quả
