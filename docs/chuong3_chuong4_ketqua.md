# Báo cáo kết quả thực nghiệm (Chương 3 & Chương 4)
Tài liệu tổng hợp từ các file trong `runs/`:
- `runs/human_crowd_yolov8s/args.yaml`
- `runs/human_crowd_yolov8s/results.csv`
- `runs/val_metrics.json`

## Chương 3: Dữ liệu và Thiết lập thực nghiệm

### 3.1 Dataset sử dụng
- Tên dataset: `Human Crowd - v1` (Roboflow Universe).
- Định dạng nhãn: YOLOv8.
- Tổng số ảnh: **4362** ảnh (theo `roboflow_downloads/README.roboflow.txt`).
- Class hiện có trong dataset: `head`, `person` (theo `roboflow_downloads/data.yaml`, `nc=2`).

### 3.2 Phân phối dữ liệu (theo split local đã tải)
- Train:
  - Số ảnh: **3272**
  - Số bbox class `head`: **66668**
  - Số bbox class `person`: **66668**
- Validation:
  - Số ảnh: **1090**
  - Số bbox class `head`: **22848**
  - Số bbox class `person`: **22848**

Ghi chú:
- Dữ liệu đang cân bằng giữa hai class `head` và `person`.
- Nếu mục tiêu đề tài chỉ detect người đi bộ, nên cố định chế độ infer với `classes=[1]` (class `person`) hoặc huấn luyện lại trên bộ `person-only`.

### 3.3 Tiền xử lý và augmentation
- Tiền xử lý từ Roboflow:
  - Auto-orientation theo EXIF
  - Resize về `640x640` (fit + black edges)
- Theo metadata export: không áp augmentation ở bước export của Roboflow.
- Theo cấu hình train Ultralytics (`args.yaml`), augmentation trong lúc train có:
  - `mosaic=1.0`
  - `mixup=0.0`
  - `translate=0.1`, `scale=0.5`, `fliplr=0.5`
  - `hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4`

### 3.4 Cấu hình môi trường và hyperparameters
- Framework: Ultralytics YOLO detect.
- Cấu hình run (`runs/human_crowd_yolov8s/args.yaml`):
  - `model: yolov8n.pt` (pretrained, tức fine-tune)
  - `data: /kaggle/working/Human-Crowd-1/data.yaml`
  - `epochs: 50`
  - `batch: 16`
  - `imgsz: 640`
  - `optimizer: auto`
  - `lr0: 0.01`, `lrf: 0.01`
  - `device: '0'` (GPU trên Kaggle)
  - `project: /kaggle/working/runs`
  - `name: human_crowd_yolov8s`

## Chương 4: Đánh giá kết quả thực nghiệm

### 4.1 Kết quả theo log train (`results.csv`)
Kết quả epoch cuối (epoch 50):
- Precision: **0.84765**
- Recall: **0.65275**
- mAP@0.5: **0.74177**
- mAP@0.5:0.95: **0.45638**
- F1-score (tính từ P/R): **0.73754**

Nhận xét hội tụ:
- Loss train giảm đều theo epoch:
  - `train/box_loss`: 1.66238 -> 1.22960
  - `train/cls_loss`: 1.45384 -> 0.66425
  - `train/dfl_loss`: 1.31709 -> 1.10615
- mAP tăng dần và đạt tốt nhất ở epoch cuối:
  - best mAP@0.5 tại epoch 50: **0.74177**
  - best mAP@0.5:0.95 tại epoch 50: **0.45638**

### 4.2 Kết quả validate độc lập (`val_metrics.json`)
Theo `results_dict` trong `runs/val_metrics.json`:
- Precision: **0.84847**
- Recall: **0.66339**
- mAP@0.5: **0.79074**
- mAP@0.5:0.95: **0.52569**
- Fitness: **0.52569**

Thông tin thêm từ `val_metrics.json`:
- `names: {0: 'head', 1: 'person'}`
- `nt_per_class`: xấp xỉ 22.8k bbox mỗi class trên tập val.
- `speed` (ms/image):
  - preprocess: **0.988**
  - inference: **4.322**
  - postprocess: **3.157**
  - tổng xấp xỉ: **8.467 ms/image** (~ **118 FPS** lý thuyết trong điều kiện benchmark ảnh đơn).

### 4.3 Lưu ý quan trọng về tính nhất quán metric
Hiện có độ lệch metric giữa hai nguồn:
- `results.csv` epoch 50: mAP@0.5:0.95 = **0.45638**
- `val_metrics.json`: mAP@0.5:0.95 = **0.52569**

Khuyến nghị khi viết báo cáo chính thức:
- Chọn **một nguồn chính** để tránh mâu thuẫn số liệu trong văn bản.
- Nếu chọn `val_metrics.json`, ghi rõ đây là kết quả validate riêng trên checkpoint tốt nhất.
- Nếu chọn `results.csv`, ghi rõ đây là metric theo tiến trình train tại epoch cuối.

### 4.4 Kết luận ngắn cho Chương 4
- Mô hình đã học ổn định, loss giảm đều và mAP tăng theo epoch.
- Precision cao (~0.85), nhưng Recall thấp hơn (~0.65), gợi ý mô hình còn bỏ sót một phần đối tượng trong điều kiện khó.
- Dữ liệu hiện gồm cả `head` và `person`; để đúng mục tiêu phát hiện người đi bộ, cần thống nhất chiến lược đánh giá class `person` trong toàn bộ thí nghiệm.
