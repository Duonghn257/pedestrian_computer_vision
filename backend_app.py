from __future__ import annotations

import time
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "outputs"
FRONTEND_DIR = ROOT / "frontend"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# If you have trained weights, replace with your path:
# WEIGHTS_PATH = ROOT / "runs" / "human_crowd_yolov8s" / "weights" / "best.pt"
WEIGHTS_PATH = ROOT / "yolov8n.pt"

# On COCO models, class id 0 = person.
PERSON_CLASS_ID = 0

app = FastAPI(title="Pedestrian Video Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _create_compatible_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """
    Try a list of codecs and return the first working writer.
    Browsers usually play H264 (`avc1`) better than `mp4v`.
    """
    codec_candidates = ["avc1", "H264", "mp4v"]
    for codec in codec_candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError("Cannot create video writer with supported codecs (avc1/H264/mp4v).")


def process_video_person_only(input_path: Path, output_path: Path) -> dict:
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Missing weights: {WEIGHTS_PATH}")

    model = YOLO(str(WEIGHTS_PATH))

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError("Cannot open uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = _create_compatible_writer(output_path, fps, width, height)

    total_frames = 0
    total_person_detections = 0
    max_person_per_frame = 0

    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total_frames += 1

        result = model.predict(
            source=frame,
            conf=0.25,
            iou=0.6,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )[0]

        person_count_frame = 0
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for box, conf in zip(xyxy, confs):
                x1, y1, x2, y2 = map(int, box.tolist())
                person_count_frame += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.putText(
                    frame,
                    f"person {conf:.2f}",
                    (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 220, 0),
                    2,
                )

        total_person_detections += person_count_frame
        max_person_per_frame = max(max_person_per_frame, person_count_frame)

        cv2.putText(
            frame,
            f"person/frame: {person_count_frame}",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )
        writer.write(frame)

    elapsed = max(time.perf_counter() - t0, 1e-9)
    cap.release()
    writer.release()

    return {
        "frames": total_frames,
        "video_fps": fps,
        "process_fps": total_frames / elapsed,
        "total_person_detections": total_person_detections,
        "max_person_per_frame": max_person_per_frame,
    }


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv"}:
        raise HTTPException(status_code=400, detail="Unsupported video format.")

    file_id = uuid.uuid4().hex
    input_path = UPLOAD_DIR / f"{file_id}{suffix}"
    output_path = OUTPUT_DIR / f"{file_id}_person_only.mp4"

    with input_path.open("wb") as f:
        f.write(await file.read())

    try:
        metrics = process_video_person_only(input_path, output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return {
        "message": "Processed successfully",
        "video_url": f"/api/video/{output_path.name}",
        "metrics": metrics,
    }


@app.get("/api/video/{filename}")
def get_video(filename: str):
    video_path = OUTPUT_DIR / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(str(video_path), media_type="video/mp4", filename=filename)


# Mount frontend LAST so it does not override /api/* routes.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

