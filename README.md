# Pedestrian Detection Demo (YOLOv8 + FastAPI + Frontend)

Project demo phát hiện người (`person`) trên video, xử lý ở backend bằng YOLOv8 và hiển thị kết quả trên frontend.

## 1) Tổng quan hệ thống

- **Backend**: `FastAPI` (`backend_app.py`)
- **Frontend**: HTML tĩnh (`frontend/index.html`)
- **Model infer**: YOLOv8 (Ultralytics)
- **Luồng xử lý**:
  1. Upload video qua UI
  2. Backend infer từng frame
  3. Vẽ bounding box class `person`
  4. Xuất video output và trả về metrics

## 2) Yêu cầu môi trường

- Python 3.10+ (khuyến nghị dùng virtualenv)
- pip

## 3) Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4) Chuẩn bị dữ liệu và artifacts

### Video input

Tải video mẫu tại:

- [Google Drive - data_video folder](https://drive.google.com/drive/folders/10WpOzojaXPXtJP-oA1IJJqUhBBA0r5Hb?usp=sharing)

Sau khi tải xong, đặt vào thư mục:

- `data_video/`

### Training runs (nếu cần đối chiếu kết quả)

Tải `runs.zip` tại:

- [Google Drive - runs.zip](https://drive.google.com/file/d/1RSXUSC3nfeFnmaGe9s19gTBYxpgXnsQ8/view?usp=sharing)

Giải nén vào root project để có cấu trúc `runs/...`.

### Model đã fine-tune (`yolov8n.pt`)

Model fine-tune bạn cung cấp tại:

- [Google Drive - yolov8n.pt](https://drive.google.com/file/d/1JGQLzQ-SABQRm1xuSsUxK0cCSZIumk2d/view?usp=sharing)

Cách dùng:

- Tải file từ Drive về.
- Đặt file `yolov8n.pt` ở **thư mục gốc project** (cùng cấp với `backend_app.py`).
- Chạy backend là dùng đúng model fine-tune, không cần sửa code.

## 5) Chọn model để infer

Trong `backend_app.py` hiện tại:

```python
WEIGHTS_PATH = ROOT / "yolov8n.pt"
```

## 6) Chạy project

```bash
uvicorn backend_app:app --host 0.0.0.0 --port 8000 --reload
```

Mở trình duyệt:

- [http://localhost:8000](http://localhost:8000)

Sau đó:

1. Chọn video (`.mp4`, `.mov`, `.avi`, `.mkv`)
2. Upload để backend xử lý
3. Xem video output và metrics trả về

## 7) API chính

- `POST /api/upload`: upload video và xử lý detect person
- `GET /api/video/{filename}`: trả video output

## 8) Output tạo ra

- Video upload tạm: `uploads/`
- Video đã xử lý: `outputs/`

## 9) Ghi chú

- Class detect đang lock là `person` (`PERSON_CLASS_ID = 0`).
- Nếu model/đường dẫn weights không tồn tại, API sẽ báo lỗi.
- Đã có `.gitignore` để bỏ qua các thư mục output, cache, dataset lớn và weights.

