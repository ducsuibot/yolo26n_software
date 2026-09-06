#ifndef DRAM_H
#define DRAM_H

// Các con trỏ bộ nhớ tổng mô phỏng kiến trúc phần cứng
extern float *input_buffer;
extern float *dram1;
extern float *dram2;

// Hàm khởi tạo và giải phóng cấp phát động
void init_dram();
void free_dram();

// Nạp dữ liệu từ file txt (.txt chứa list các số float phân tách bằng khoảng trắng/xuống dòng)
// - filename: tên file txt
// - dram_ptr: trỏ tới vùng nhớ đích (input_buffer, dram1, hoặc dram2)
// - offset: địa chỉ bắt đầu nạp (dựa theo bảng CSV)
// - size: số lượng phần tử cần nạp
void load_txt_to_dram(const char *filename, float *dram_ptr, int offset, int size);

#endif // DRAM_H