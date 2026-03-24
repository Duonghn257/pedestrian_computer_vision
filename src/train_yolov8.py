from __future__ import annotations

import argparse
from pathlib import Path

from config import PERSON_ONLY_DATA_YAML, RUNS_DIR


def _auto_device() -> str:
    # Ultralytics supports "mps" on Apple Silicon and "cpu" fallback.
    # We keep it simple to avoid importing torch here.
    try:
        import torch  # type: ignore

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(PERSON_ONLY_DATA_YAML))
    ap.add_argument("--model", type=str, default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--project", type=str, default=str(RUNS_DIR))
    ap.add_argument("--name", type=str, default="person_yolov8_visible")
    args = ap.parse_args()

    if not Path(args.data).exists():
        raise FileNotFoundError(f"--data not found: {args.data}")

    # Import lazily so dataset prep can run without YOLO installed.
    from ultralytics import YOLO  # type: ignore

    device = _auto_device() if args.device == "auto" else args.device

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
    )

    print(f"Training finished. Check: {RUNS_DIR / args.name}")


if __name__ == "__main__":
    main()

