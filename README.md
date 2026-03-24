# Report Requirements (YOLOv8 Person Detection)

## 1. Muc tieu de tai (Tong quan de dua vao Chuong 1)
- Bai toan: phat hien nguoi di bo (class `person`) tren video duong pho thuc te.
- Huong tiep can: xay dung pipeline tu video dau vao -> infer YOLOv8 -> ve bounding box -> xuat video dau ra.
- Tieu chi danh gia (de thuc hien theo Chuong 4):
  - do chinh xac (mAP@0.5, mAP@0.5:0.95, Precision/Recall/F1)
  - toc do (FPS tren video)

## 2. Scope code & demo (can doi chieu voi `code_and_demo.pdf`)
> Luu y: Trong moi truong hien tai, `code_and_demo.pdf` khong trich duoc text (co the la scan/anh). Vi vay doan ben duoi la checklist theo template/report da co trong `ComputerVision_Report.pdf`. Neu `code_and_demo.pdf` co yeu cau cu the hon, vui long gui ban co OCR hoac noi dung text de minh chinh lai 100%.

### 2.1. Code deliverables (yeu cau toi thieu de co demo chay duoc)
- [ ] Script/command huan luyen model YOLOv8 (model backbone tuong ung: `yolov8m` hoac tuong duong) tren dataset da chuan hoa theo dinh dang YOLO.
- [ ] Script/command infer tren 1 anh/1 frame de kiem tra nhanh.
- [ ] Pipeline infer tren video:
  - doc video dau vao (OpenCV, frame-by-frame)
  - chay model YOLOv8
  - ap dung NMS (neu framework khong dam bao)
  - render bounding box + confidence score
  - xuat video `mp4`
- [ ] Log/ket qua sinh ra tu training (de dua vao Chuang 4):
  - bang metrics tren test/val (mAP, Precision/Recall, F1)
  - bieu do loss theo epoch (training convergence)

### 2.2. Demo requirements (nhung thu hoi dong can thay)
- [ ] Demo co it nhat 1 video dau vao tu duong pho thuc te va 1 video dau ra co ve bounding box `person`.
- [ ] Demo the hien duoc FPS (hoac thoi gian infer trung binh) tren video.
- [ ] Demo co vai truong hop minh hoa:
  - thanh cong (detection dung)
  - that bai (false positive / false negative do toi, che khuat)
- [ ] Neu co them: giai thich tuy chon (img size, confidence threshold, IoU threshold) dung trong infer.

## 3. Du lieu can chuan bi cho Chuang 3: Data & Experimental Setup

### 3.1. Tap du lieu (Dataset)
Ben trong bao cao, phan nay can co:
- [ ] Gioi thieu ro nguon dataset (vi du: CrowdHuman / MOT / COCO hoac tap tu tuyen chinh).
- [ ] Neu ro muc tieu label:
  - Chi lay `person` (neu dataset goc co nhieu class thi phai loc/chien luoc label).
- [ ] Thong ke dataset:
  - so luong anh/video frames
  - ti le class `person` / so doi tuong annotated
  - phan bo theo dieu kien: che khuat, kich thuoc nho (xa), anh sang thay doi, goc chuyen dong
- [ ] Dataset split:
  - `train/val/test` (ty le va ly do chon)
  - danh sach file hoac mo ta cach split (neu split theo video thi ghi ro nguyen tac)

### 3.2. Tien xu ly du lieu & Data Augmentation
Trong bao cao, phai mo ta duoc:
- [ ] Chuyen doi dinh dang nhan (labels) sang chuan YOLO:
  - YOLO label: `class x_center y_center w h` (don vi normalized [0..1])
  - cach xac dinh cac gia tri tu bounding box goc (x1,y1,x2,y2) -> (xc,yc,w,h)
- [ ] Data Augmentation (ghi ro da dung nhung gi va vi sao):
  - Mosaic
  - MixUp
  - Random Perspective
  - (Neu co) Random Flip/HSV/Jitter tu cau hinh YOLOv8
- [ ] Neu co cau hinh custom augmentation (neu dung), ghi ro tham so (bang hoac anh chup cau hinh).

### 3.3. Thiet lap moi truong & Hyperparameters (Hardware + Training)
Can liet ke day du de co the tai lap (reproducible):
- [ ] Mo ta moi truong:
  - CPU, RAM
  - GPU (model), CUDA version (neu co)
  - he dieu hanh
  - version YOLOv8/Ultralytics (neu biet)
- [ ] Hyperparameters training:
  - model: `yolov8m` (hoac bien the da chon)
  - image size (`imgsz`)
  - number of epochs
  - batch size
  - optimizer (vi du: SGD/AdamW neu YOLOv8 da set)
  - learning rate schedule (warmup/cosine/step neu co)
  - learning rate
  - confidence threshold, IoU threshold (cho inference, va neu co trong training thi ghi)
  - early stopping (neu co)
- [ ] Cac file/ket qua sinh ra:
  - weights/checkpoints (best/last)
  - file log metrics
  - bieu do loss theo epoch (de dua vao Chuang 4)

## 4. Du lieu can chuan bi cho Chuang 4: Evaluation Results & Analysis

### 4.1. Evaluation Metrics
Bao cao can giai thich va dua con so tu training:
- [ ] IoU
- [ ] Precision, Recall, F1-score
- [ ] mAP@0.5 va mAP@0.5:0.95
- [ ] FPS (chi so quan trong cho video)

### 4.2. Phan tich qua trinh huan luyen
- [ ] Cac bieu do/tap anh minh hoa su hoi tu (convergence) theo epoch:
  - training loss (va/hoac val loss neu co)
  - mAP theo epoch (neu framework cung cap)
- [ ] Nhan xet:
  - epoch nao dat diem tot (best checkpoint)
  - co vuot/dao dong khong (overfitting/underfitting) va giai thich theo du lieu

### 4.3. Ket qua danh gia tren test/val
- [ ] Bang tong hop metrics:
  - mAP@0.5
  - mAP@0.5:0.95
  - Precision, Recall, F1
  - (Neu co) so truong hop detect dung/sai tren sample video
- [ ] Neu co nhieu run/bieu do so sanh:
  - ghi ro config khac nhau giua cac run (img size, epochs, confidence, augmentation)
  - ket luan run nao tot hon va ly do

### 4.4. Failure cases (Thanh cong & that bai)
- [ ] It nhat 5-10 hinh/clip khac nhau cho:
  - false positives (nhan sai)
  - false negatives (bong bo nguoi)
- [ ] Mo ta ly do that bai duoi dang bullet trong bao cao:
  - toi (low light)
  - che khuat (occlusion)
  - nguoi qua nho/xa
  - blur do van toc/giong rung

## 5. Du lieu can chuan bi cho Chuang 5: Video Processing Pipeline

### 5.1. Luong xu ly video dau vao (Input)
- [ ] Mo ta cach doc video:
  - OpenCV `VideoCapture`
  - doc tung frame theo thu tu (frame-by-frame)
  - xu ly truong hop loi/het frame
- [ ] Neu co resize/format truoc infer, ghi ro:
  - cach scale anh
  - giu ti le (letterbox) neu dung

### 5.2. Noi suy (neu co) & du doan (Inference)
- [ ] Mo ta luong infer:
  - dua frame vao YOLOv8 da huan luyen (best checkpoint)
  - thuc hien post-processing:
    - NMS de loai bo bbox trung lap
  - chon nguong confidence
- [ ] Neu co noi suy giua cac frame (vi du: bo qua frame, interpolate), ghi ro:
  - stride/skip frame
  - cach interpolate va ly do

### 5.3. Render video dau ra (Output)
- [ ] Mo ta render:
  - ve bounding box
  - gan nhan `person`
  - gan confidence score (neu yeu cau)
  - xuat video mp4 voi FPS/codec phu hop
- [ ] Dua vao bao cao:
  - 3-6 khung hinh (frames) minh hoa truoc/sau detect
  - link hoac mo ta video demo da xuat

## 6. Ket luan & Huong phat trien
- [ ] Ket luan: tom tat nhung gi da lam duoc:
  - ly thuyet YOLOv8
  - training dat yeu cau (mAP/FPS)
  - xuat video prediction co cha luc tot
- [ ] Huong phat trien:
  - toi uu model nhe hon (quantization, pruning)
  - ket hop tracking (DeepSORT / BoT-SORT) de theo doi ID

## 7. Tai lieu tham khao (Reference)
- [ ] Liet ke:
  - Papers bai bao khoa hoc phu hop
  - Dataset paper/Trang web chinh thong
  - Repo/implementation YOLOv8 (Ultralytics) va cac link
- [ ] Dung dinh dang cite theo IEEE hoac APA (ghi ro chinh xac format)

