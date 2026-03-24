## [cite_start]Chương 1: Tổng quan về đề tài (Khoảng 5 - 7 trang) [cite: 1]
[cite_start]**Mục tiêu:** Đặt vấn đề, nêu rõ lý do tại sao phát hiện người đi bộ trên video đường phố lại quan trọng và khó khăn. [cite: 2]

* **1.1. [cite_start]Đặt vấn đề & Lý do chọn đề tài:** Khó khăn của bài toán (che khuất, kích thước nhỏ, thay đổi ánh sáng). [cite: 3]
* **1.2. [cite_start]Mục tiêu nghiên cứu:** Xây dựng luồng xử lý nhận diện người đi bộ trên video bằng YOLOv8. [cite: 5]
* **1.3. [cite_start]Đối tượng và phạm vi nghiên cứu:** Giới hạn nhận diện class person, test trên video đường phố thực tế. [cite: 6]
* **1.4. [cite_start]Phương pháp nghiên cứu khoa học:** Mô tả phương pháp thu thập dữ liệu, phương pháp thực nghiệm và phương pháp đánh giá (định lượng qua mAP, FPS). [cite: 7]
* **1.5. [cite_start]Cấu trúc của báo cáo:** Tóm tắt nội dung các chương tiếp theo. [cite: 8]

> [cite_start]**Ghi chú:** Bạn hoàn toàn có lý khi tập trung vào Chương 2, vì với bậc Thạc sĩ, hội đồng và thầy giáo sẽ soi rất kỹ nền tảng toán học và sự hiểu biết về bản chất kiến trúc thay vì chỉ biết gọi API có sẵn. [cite: 9] [cite_start]Đặc biệt ở phần Hàm mất mát, việc vận dụng sâu các góc nhìn về kỹ thuật tối ưu (Optimization Techniques) để phân tích cách thuật toán Gradient Descent tối ưu hóa không gian tham số, vượt qua các điểm local minima thông qua đạo hàm của CIoU hay VFL sẽ cực kỳ ghi điểm. [cite: 10, 11] [cite_start]Dưới đây là bố cục chi tiết đến từng mục nhỏ cho Chương 2: Cơ sở lý thuyết và Kiến trúc mô hình YOLOv8, kèm theo các gợi ý viết và công thức toán học (bạn có thể đưa trực tiếp vào file Word). [cite: 12]

---

## [cite_start]Chương 2: Cơ sở lý thuyết và Kiến trúc mô hình YOLOv8 [cite: 13]

### 2.1. [cite_start]Tổng quan bài toán Phát hiện vật thể (Object Detection) [cite: 14]
* **2.1.1. [cite_start]Định nghĩa bài toán:** Giải thích bài toán phát hiện người đi bộ bao gồm hai nhiệm vụ song song: Phân loại (Classification - xác định đó là người) và Định vị (Localization - vẽ Bounding Box). [cite: 15] [cite_start]Tọa độ được biểu diễn dưới dạng $(x_{center}, y_{center}, w, h)$. [cite: 16]
* **2.1.2. [cite_start]Phân loại các kiến trúc mạng:** [cite: 17]
    * Mạng một giai đoạn (One-stage detectors): YOLO, SSD. Ưu điểm: Tốc độ cao (phù hợp với video đường phố). [cite: 18]
    * Mạng hai giai đoạn (Two-stage detectors): Faster R-CNN. [cite_start]Ưu điểm: Độ chính xác cao nhưng chậm do có bước sinh vùng đề xuất (Region Proposal Network - RPN). [cite: 19]
    * [cite_start]Lý do chọn YOLOv8: Lập luận bảo vệ lựa chọn của nhóm dựa trên sự cân bằng giữa độ chính xác (mAP) và tốc độ (FPS). [cite: 21]

### 2.2. [cite_start]Các thành phần cơ bản của Mạng nơ-ron tích chập (CNN) [cite: 22]
[cite_start]*(Lưu ý: Chỉ tóm tắt ngắn gọn vì đây là kiến thức nền)* [cite: 23]
* **2.2.1. [cite_start]Lớp Tích chập (Convolutional Layer):** Khái niệm Kernel, Stride, Padding và cách chúng trích xuất đặc trưng không gian của người đi bộ. [cite: 24]
* **2.2.2. [cite_start]Hàm kích hoạt (Activation Function):** Khúc này nhấn mạnh YOLOv8 sử dụng hàm SiLU (Sigmoid Linear Unit) thay vì ReLU truyền thống để đạo hàm trơn tru hơn: [cite: 25]
    
    $$f(x) = x \cdot \sigma(x) = \frac{x}{1+e^{-x}}$$
    [cite: 26]
* **2.2.3. [cite_start]Lớp Gộp (Pooling Layer):** Giảm kích thước không gian (Spatial dimension). [cite: 27]

### 2.3. [cite_start]Cấu trúc Trích xuất đặc trưng (Backbone) [cite: 28]
* **2.3.1. [cite_start]Kiến trúc CSPDarknet:** Giải thích cách mạng Cross Stage Partial (CSP) chia feature map thành hai nhánh để giảm lượng tính toán nhưng vẫn giữ được luồng gradient (gradient flow) mạnh mẽ. [cite: 29]
* **2.3.2. [cite_start]Khối C2f (Cross Stage Partial Bottleneck with 2 Convolutions):** Đây là điểm mới của YOLOv8. [cite: 30] [cite_start]Giải thích cách C2f thay thế C3 của YOLOv5 để kết hợp đặc trưng mức thấp (low-level) và mức cao (high-level) tốt hơn, rất hiệu quả khi nhận diện người đi bộ ở đằng xa (kích thước nhỏ). [cite: 31]

### 2.4. [cite_start]Cấu trúc Trộn đặc trưng (Neck) [cite: 32]
* **2.4.1. [cite_start]Mạng FPN (Feature Pyramid Network):** Khối truyền đặc trưng ngữ nghĩa từ các layer sâu (độ phân giải thấp) lên các layer nông (độ phân giải cao). [cite: 33]
* **2.4.2. [cite_start]Khối PANet (Path Aggregation Network):** Khối dẫn đường đi từ dưới lên (Bottom-up path augmentation) giúp định vị Bounding Box chính xác hơn. [cite: 34] [cite_start]Tip báo cáo: Kết hợp FPN + PANet giúp YOLOv8 không bỏ sót người đi bộ bị lấp ló sau xe cộ. [cite: 35]

### 2.5. [cite_start]Cấu trúc Dự đoán (Head) - Trọng tâm lý thuyết [cite: 36]
* **2.5.1. [cite_start]Cơ chế Anchor-free (Không dùng khung neo):** So sánh với Anchor-based: Các phiên bản YOLO cũ (như v5) dùng các anchor boxes định nghĩa sẵn. [cite: 37] [cite_start]Thầy giáo rất hay hỏi nhược điểm của cái này (là phải gom cụm k-means trước cho dataset và kém linh hoạt). [cite: 38] [cite_start]Cách Anchor-free hoạt động: YOLOv8 dự đoán trực tiếp khoảng cách từ tâm (center) của cell đến 4 cạnh của Bounding Box. [cite: 39]
* **2.5.2. [cite_start]Đầu ra tách biệt (Decoupled Head):** [cite: 40] [cite_start]Giải thích việc tách riêng nhánh Classification (phân loại) và nhánh Bounding Box Regression (hồi quy). [cite: 41] [cite_start]Việc tách này loại bỏ sự mâu thuẫn (misalignment) giữa hai tác vụ, giúp mô hình hội tụ nhanh hơn. [cite: 42]

### 2.6. [cite_start]Hàm Mất mát (Loss Functions) và Tối ưu hóa [cite: 43]
[cite_start]*(Đây là phần "show kỹ năng" để thuyết phục hội đồng về nền tảng toán)* [cite: 44]
* **2.6.1. [cite_start]Hàm mất mát phân loại (Classification Loss):** Sử dụng Binary Cross-Entropy (BCE) Loss hoặc Varifocal Loss (VFL). [cite: 45] [cite_start]VFL đặc biệt hữu ích để xử lý sự mất cân bằng giữa class có ích (positive) và class nền (negative background). [cite: 46]
* **2.6.2. [cite_start]Hàm mất mát hồi quy Bounding Box (Regression Loss):** [cite: 47] YOLOv8 loại bỏ IoU thông thường và sử dụng CIoU (Complete IoU). [cite_start]CIoU xét đến 3 yếu tố: độ lệch tâm, khoảng cách diện tích chập và tỷ lệ khung hình (Aspect Ratio). [cite: 48, 49]
    [cite_start]Công thức đưa vào Word: [cite: 50]

    $$L_{CIoU}=1-IoU+\frac{\rho^2(b,b^{gt})}{c^2}+\alpha v$$
    [cite: 51]
    [cite_start]Trong đó, $v$ đo lường sự khác biệt về tỷ lệ khung hình: [cite: 51]

    $$v=\frac{4}{\pi^2} \left(\arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h} \right)^2$$
    [cite: 51]
* **2.6.3. [cite_start]Distribution Focal Loss (DFL):** [cite: 52] [cite_start]Tối ưu hóa khả năng mô hình dự đoán các viền của Bounding Box (edges) dưới dạng một phân phối xác suất liên tục thay vì một giá trị sắc nét (điều này cực tốt khi viền người đi bộ bị nhòe do chuyển động trong video). [cite: 53]

    $$DFL(S_i, S_{i+1}) = \left( (y_{i+1} - y) \log(S_i) + (y-y_i) \log(S_{i+1}) \right)$$
    [cite: 54]

---

## [cite_start]Chương 3: Dữ liệu và Thiết lập thực nghiệm (Khoảng 10 - 15 trang) [cite: 55]
[cite_start]**Mục tiêu:** Trình bày quy trình chuẩn bị "nguyên liệu" và cấu hình huấn luyện. [cite: 56]

* **3.1. [cite_start]Tập dữ liệu (Dataset):** [cite: 57]
    * 3.1.1. Giới thiệu bộ dữ liệu (CrowdHuman / MOT/COCO). [cite: 58]
    * 3.1.2. [cite_start]Phân tích phân phối dữ liệu (Số lượng ảnh, tỷ lệ class, các trường hợp che khuất). [cite: 59]
* **3.2. [cite_start]Tiền xử lý dữ liệu và Data Augmentation:** [cite: 60]
    * 3.2.1. Chuyển đổi định dạng nhãn sang chuẩn YOLO. [cite: 60]
    * 3.2.2. [cite_start]Các kỹ thuật tăng cường dữ liệu: Mosaic, MixUp, Random Perspective... [cite: 61]
* **3.3. [cite_start]Thiết lập môi trường và tham số huấn luyện (Hyperparameters):** Hardware (GPU), số lượng Epochs, Batch size, Learning rate scheduler, Optimizer. [cite: 62]

---

## [cite_start]Chương 4: Đánh giá kết quả thực nghiệm (Khoảng 15 - 18 trang) [cite: 63]
[cite_start]**Mục tiêu:** Đưa ra các con số biết nói, biểu đồ trực quan để chứng minh hiệu năng. [cite: 64]

* **4.1. [cite_start]Các chỉ số đánh giá (Evaluation Metrics):** [cite: 65]
    * 4.1.1. Intersection over Union ($IoU$). [cite: 66]
    * 4.1.2. [cite_start]Precision, Recall, và F1-Score. [cite: 67]
    * 4.1.3. [cite_start]Mean Average Precision ($mAP@0.5$ và $mAP@0.5:0.95$). [cite: 68]
    * 4.1.4. Frames Per Second (FPS) - Chỉ số quan trọng cho video. [cite: 69]
* **4.2. [cite_start]Phân tích quá trình huấn luyện:** Đồ thị biểu diễn sự hội tụ của Loss function qua các Epochs. [cite: 70]
* **4.3. [cite_start]Kết quả đánh giá trên tập Test:** Bảng tổng hợp các chỉ số mAP. [cite: 72]
* **4.4. [cite_start]Phân tích các trường hợp thành công và thất bại (Failure cases):** Đưa ra hình ảnh thực tế mô hình nhận diện tốt và những ca dự đoán sai (False Positives/False Negatives) do quá tối hoặc bị che khuất hoàn toàn. [cite: 73, 74]

---

## [cite_start]Chương 5: Xây dựng Pipeline xử lý Video (Khoảng 7 - 10 trang) [cite: 75]
[cite_start]**Mục tiêu:** Trình bày sản phẩm thực tế. [cite: 76]

* **5.1. [cite_start]Luồng xử lý video đầu vào:** Cách dùng OpenCV đọc từng frame (frame-by-frame). [cite: 77]
* **5.2. [cite_start]Quá trình nội suy và dự đoán:** Đưa frame qua trọng số YOLOv8m đã huấn luyện, áp dụng Non-Maximum Suppression (NMS) để lọc Bounding box trùng lặp. [cite: 79]
* **5.3. [cite_start]Render video đầu ra:** Vẽ Bounding Box, gán nhãn độ tin cậy (Confidence score) và xuất ra file video mp4. [cite: 80]

---

## [cite_start]Kết luận và Hướng phát triển (Khoảng 2 trang) [cite: 81]
* [cite_start]**Kết luận:** Tóm tắt những gì nhóm đã làm được (Hiểu lý thuyết, huấn luyện thành công mô hình, xuất ra video dự đoán tốt với tốc độ FPS đáp ứng thực tế). [cite: 82]
* **Hướng phát triển:** Đề xuất hướng tối ưu mô hình nhẹ hơn (như lượng tử hóa - Quantization) để chạy trên các thiết bị nhúng (Jetson Nano) hoặc kết hợp thêm thuật toán Tracking (như DeepSORT/BoT-SORT) để theo dõi ID từng người. [cite: 83]

---

## Tài liệu tham khảo (Khoảng 2 trang) [cite: 84]
* [cite_start]Liệt kê sách, bài báo khoa học (Papers), và các links GitHub/Dataset chuẩn APA hoặc IEEE. [cite: 85]