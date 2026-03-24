# Yêu Cầu & Cấu Trúc Báo Cáo Đồ Án: Nhận Diện Người Đi Bộ Đường Phố (YOLOv8)

Tài liệu này quy định cấu trúc báo cáo đồ án Thị giác máy tính và danh sách các dữ liệu/logs mà team (4 thành viên) cần lưu lại trong quá trình training và deploy để phục vụ cho việc viết báo cáo.

---

## [cite_start]Chương 1: Tổng quan về đề tài (Dự kiến 5 - 7 trang) [cite: 1]

- [cite_start]**Mục tiêu:** Nhận diện class `person` trên video đường phố thực tế[cite: 6].
- [cite_start]**Khó khăn bài toán:** Xử lý các trường hợp che khuất, kích thước nhỏ ở xa, và thay đổi ánh sáng[cite: 3].
- [cite_start]**Phương pháp đánh giá:** Định lượng qua mAP và FPS[cite: 7].

---

## Chương 2: Cơ sở lý thuyết và Kiến trúc YOLOv8

> [cite_start]**Lưu ý cho team viết báo cáo:** Cần xoáy sâu vào kỹ thuật tối ưu (Optimization Techniques) và cách Gradient Descent vượt qua local minima[cite: 10, 11].

- [cite_start]**2.1 - 2.2 Kiến trúc cơ bản:** * Bài toán bao gồm Phân loại (Classification) và Định vị (Localization) với tọa độ $(x_{center}, y_{center}, w, h)$ [cite: 15, 16].
  - [cite_start]So sánh mạng One-stage (tốc độ cao) và Two-stage (độ chính xác cao nhưng chậm do RPN) để bảo vệ lý do chọn YOLOv8[cite: 18, 19, 21].
  - [cite_start]Hàm kích hoạt SiLU:

    $$f(x)=x \cdot \sigma(x)=\frac{x}{1+e^{-x}}$$
    [cite: 25, 26].
- **2.3 - 2.4 Cấu trúc mạng:**
  - [cite_start]Backbone: Khối C2f thay thế C3, giúp kết hợp đặc trưng mức thấp và cao, cực kỳ hiệu quả để nhận diện người đi bộ kích thước nhỏ ở xa[cite: 30, 31].
  - [cite_start]Neck: Kết hợp FPN và PANet giúp không bỏ sót người đi bộ lấp ló sau xe cộ[cite: 33, 34, 35].
- **2.5 - 2.6 Head & Loss Functions (Trọng tâm Toán học):**
  - [cite_start]Cơ chế Anchor-free và Decoupled Head (tách nhánh Classification và Bounding Box Regression)[cite: 37, 39, 40, 41].
  - [cite_start]**Classification Loss:** Binary Cross-Entropy (BCE) hoặc Varifocal Loss (VFL) để xử lý mất cân bằng class[cite: 45, 46].
  - [cite_start]**Regression Loss (CIoU):** Xét độ lệch tâm, diện tích chập và tỷ lệ khung hình[cite: 48, 49].
    
    $$L_{CIoU}=1-IoU+\frac{\rho^2(b,b^{gt})}{c^2}+\alpha v$$
    [cite: 51].

    $$v=\frac{4}{\pi^2} \left(\arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h} \right)^2$$
    [cite: 51].
  - [cite_start]**Distribution Focal Loss (DFL):** Tối ưu viền Bounding Box nhòe do chuyển động[cite: 52, 53].
    
    $$DFL(S_i, S_{i+1}) = \left( (y_{i+1} - y) \log(S_i) + (y-y_i) \log(S_{i+1}) \right)$$
    [cite: 54].

---

## [cite_start]Chương 3: Dữ liệu và Thiết lập thực nghiệm (10 - 15 trang) [cite: 55]

> **Task cho team Code:** Ghi chép lại toàn bộ thông số môi trường và data.

- [cite_start]**Dữ liệu (Dataset):** Ghi rõ dùng CrowdHuman, MOT hay COCO[cite: 58]. [cite_start]Cần xuất biểu đồ phân phối dữ liệu (số lượng ảnh, tỷ lệ class `person`, số lượng ca che khuất)[cite: 59].
- [cite_start]**Tiền xử lý & Augmentation:** * Lưu script chuyển đổi nhãn sang chuẩn YOLO[cite: 60].
  - [cite_start]Ghi nhận các kỹ thuật Augmentation đã dùng: Mosaic, MixUp, Random Perspective[cite: 61].
- [cite_start]**Tham số (Hyperparameters):** Note lại cấu hình Hardware (GPU loại gì), Epochs, Batch size, Learning rate scheduler, Optimizer[cite: 62].

---

## [cite_start]Chương 4: Đánh giá kết quả thực nghiệm (15 - 18 trang) [cite: 63]

> **Task cho team Code:** Phải lưu (save) lại các metrics và hình ảnh test thực tế.

- [cite_start]**Chỉ số đánh giá:** Tính toán IoU, Precision, Recall, F1-Score, [mAP@0.5](mailto:mAP@0.5), [mAP@0.5](mailto:mAP@0.5):0.95 và FPS[cite: 66, 67, 68, 69].
- [cite_start]**Biểu đồ huấn luyện:** Chụp/lưu lại đồ thị hội tụ của Loss function qua các Epochs[cite: 70].
- [cite_start]**Phân tích Test:** * Lập bảng tổng hợp mAP[cite: 72].
  - [cite_start]**QUAN TRỌNG:** Lưu lại các frame ảnh thành công và thất bại (False Positives/False Negatives do quá tối hoặc bị che khuất) để đưa vào báo cáo[cite: 73, 74].

---

## [cite_start]Chương 5: Xây dựng Pipeline xử lý Video (7 - 10 trang) [cite: 75]

- [cite_start]**Luồng xử lý:** Sử dụng OpenCV đọc frame-by-frame[cite: 77].
- [cite_start]**Dự đoán:** Đưa qua trọng số YOLOv8 (ví dụ: YOLOv8m), áp dụng Non-Maximum Suppression (NMS) để lọc Bounding box trùng[cite: 79].
- [cite_start]**Đầu ra:** Vẽ Bounding Box, gán Confidence score và xuất file mp4[cite: 80].

---

## [cite_start]Kết luận & Hướng phát triển [cite: 81]

- [cite_start]**Hướng mở rộng:** Nghiên cứu lượng tử hóa (Quantization) để chạy trên Jetson Nano, hoặc tích hợp Tracking (DeepSORT/BoT-SORT)[cite: 83].
- [cite_start]**Tham khảo:** Chuẩn bị sẵn link Github, Dataset, Paper theo chuẩn APA/IEEE[cite: 85].

