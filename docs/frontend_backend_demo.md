# Frontend + Backend demo (upload video -> person-only)

## 1) Cài thư viện
```bash
pip install -r requirements.txt
```

## 2) Chạy backend
```bash
uvicorn backend_app:app --host 0.0.0.0 --port 8000 --reload
```

## 3) Mở frontend
Truy cập:
- [http://localhost:8000](http://localhost:8000)

Frontend sẽ:
- Upload video
- Gọi API `/api/upload`
- Nhận video output chỉ có box class `person`
- Hiển thị:
  - `max_person_per_frame` (số người lớn nhất trong 1 frame)
  - `total_person_detections` (tổng detections trên toàn video)
  - số frame và FPS xử lý

## 4) Ghi chú model
- Backend mặc định dùng `yolov8n.pt` ở root project.
- Nếu muốn dùng model đã fine-tune, sửa `WEIGHTS_PATH` trong `backend_app.py`.
