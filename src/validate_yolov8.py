from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import RUNS_DIR


def _auto_device() -> str:
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
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=str, default=str(RUNS_DIR / "val_metrics.json"))
    args = ap.parse_args()

    data_path = Path(args.data)
    weights_path = Path(args.weights)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"--data not found: {data_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"--weights not found: {weights_path}")

    from ultralytics import YOLO  # type: ignore

    device = _auto_device() if args.device == "auto" else args.device

    model = YOLO(str(weights_path))
    results = model.val(
        data=str(data_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=device,
    )

    # `results` structure can vary by ultralytics version; we persist a JSON best-effort.
    payload = {
        "data": str(data_path),
        "weights": str(weights_path),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "device": device,
        "results_str": str(results),
    }
    try:
        payload["results"] = results.results_dict  # type: ignore[attr-defined]
    except Exception:
        pass

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Validation finished. Metrics saved to: {out_path}")


if __name__ == "__main__":
    main()

