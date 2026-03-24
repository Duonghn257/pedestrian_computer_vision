from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from config import (
    PERSON_ONLY_DATA_YAML,
    PERSON_ONLY_ROOT,
    RUNS_DIR,
)
from prepare_person_only_dataset import build_person_only_dataset


def _try_read_text(p: Path) -> str | None:
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8", errors="ignore")


def _count_instances_from_labels(labels_dir: Path, class_id: int = 0) -> int:
    total = 0
    for f in labels_dir.glob("*.txt"):
        for ln in f.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) != 5:
                continue
            cls = int(float(parts[0]))
            if cls == class_id:
                total += 1
    return total


def _count_images(images_dir: Path) -> int:
    # Roboflow export uses jpg; keep robust for png too.
    return len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png")))


def _extract_metrics_from_val_json(val_json_path: Path) -> dict:
    if not val_json_path.exists():
        return {}
    try:
        payload = json.loads(val_json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def update_report_requirements_md(report_md_path: Path, insert_ch3: str, insert_ch4: str) -> None:
    """
    Inserts extra bullet lines before the '---' delimiter ending Chương 3 and Chương 4.
    """
    if not report_md_path.exists():
        raise FileNotFoundError(f"Missing report file: {report_md_path}")

    text = report_md_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    def _insert_before_delim(chapter_title: str, insert_block: str) -> None:
        # Find chapter start
        start_idx = None
        for i, ln in enumerate(lines):
            if chapter_title in ln:
                start_idx = i
                break
        if start_idx is None:
            return

        # Find the next delimiter '---' after start_idx
        delim_idx = None
        for j in range(start_idx + 1, len(lines)):
            if lines[j].strip() == "---":
                delim_idx = j
                break
        if delim_idx is None:
            return

        # Insert only if not already inserted (idempotent-ish)
        if insert_block.strip() in text:
            return

        insert_lines = [f"{b}\n" if not b.endswith("\n") else b for b in insert_block.splitlines()]
        lines[delim_idx:delim_idx] = insert_lines

    _insert_before_delim("Chương 3:", insert_ch3)
    _insert_before_delim("Chương 4:", insert_ch4)

    report_md_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--limit-failure", type=int, default=50)
    ap.add_argument("--train-name", type=str, default="person_yolov8_visible")
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    # 1) Prepare dataset (person-only)
    build_person_only_dataset()

    train_images_dir = PERSON_ONLY_ROOT / "train" / "images"
    valid_images_dir = PERSON_ONLY_ROOT / "valid" / "images"
    train_labels_dir = PERSON_ONLY_ROOT / "train" / "labels"
    valid_labels_dir = PERSON_ONLY_ROOT / "valid" / "labels"

    n_train_images = _count_images(train_images_dir)
    n_valid_images = _count_images(valid_images_dir)
    n_train_instances = _count_instances_from_labels(train_labels_dir, class_id=0)
    n_valid_instances = _count_instances_from_labels(valid_labels_dir, class_id=0)

    # 2) Train
    from ultralytics import YOLO  # type: ignore
    import torch  # type: ignore

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device

    model = YOLO(args.model)
    train_run = model.train(
        data=str(PERSON_ONLY_DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(RUNS_DIR),
        name=args.train_name,
    )

    # Determine run directory
    # Ultralytics uses: runs/detect/<name> by default (since project="runs")
    # but internal structure might differ by version; we attempt best-effort.
    # We rely on "weights/best.pt" relative to RUNS_DIR/name.
    train_dir_candidates = [
        RUNS_DIR / "detect" / args.train_name,
        RUNS_DIR / args.train_name,
    ]
    train_dir = None
    for c in train_dir_candidates:
        if (c / "weights" / "best.pt").exists():
            train_dir = c
            break
    if train_dir is None:
        # Fallback: try to infer from returned object
        train_dir = train_dir_candidates[0]

    best_weights = train_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Could not find best weights: {best_weights}")

    # 3) Validate
    val_json_path = RUNS_DIR / "val_metrics.json"
    from validate_yolov8 import main as validate_main  # type: ignore

    # Call validate script logic programmatically by shelling args is painful; we re-run via YOLO here.
    model = YOLO(str(best_weights))
    results = model.val(
        data=str(PERSON_ONLY_DATA_YAML),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=device,
    )
    results_dict = None
    payload = {"device": device, "weights": str(best_weights), "results": None}
    try:
        results_dict = results.results_dict  # type: ignore[attr-defined]
        payload["results"] = results_dict
    except Exception:
        payload["results_str"] = str(results)
    val_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # 4) Failure cases (FP/FN)
    failure_out_dir = RUNS_DIR / "failure_cases"
    import subprocess
    import sys

    failure_cmd = [
        sys.executable,
        str(project_root / "src" / "test_failure_cases.py"),
        "--weights",
        str(best_weights),
        "--split",
        "valid",
        "--limit",
        str(args.limit_failure),
        "--conf",
        str(args.conf),
        "--match-iou",
        str(args.match_iou),
        "--out-dir",
        str(failure_out_dir),
    ]
    subprocess.run(failure_cmd, check=True)

    failure_summary_path = failure_out_dir / "summary.json"
    failure_summary = {}
    if failure_summary_path.exists():
        failure_summary = json.loads(failure_summary_path.read_text(encoding="utf-8"))

    # Store a markdown summary (user can paste into report if desired).
    summary_path = RUNS_DIR / "report_ch3_ch4_summary.md"
    summary = []
    summary.append("# Auto summary (Chương 3-4)\n")
    summary.append("## Dataset (person-only)\n")
    summary.append(f"- Train images: {n_train_images}\n")
    summary.append(f"- Val images: {n_valid_images}\n")
    summary.append(f"- Train person instances: {n_train_instances}\n")
    summary.append(f"- Val person instances: {n_valid_instances}\n")
    summary.append("## Training config\n")
    summary.append(f"- Model: {args.model}\n")
    summary.append(f"- Epochs: {args.epochs}\n")
    summary.append(f"- imgsz: {args.imgsz}\n")
    summary.append(f"- batch: {args.batch}\n")
    summary.append(f"- device: {device}\n")
    summary.append("\n## Validation metrics\n")
    summary.append(f"- Best weights: {best_weights}\n")
    summary.append(f"- Validation JSON: {val_json_path}\n")
    if failure_summary:
        summary.append("\n## Failure cases (FP/FN)\n")
        summary.append(f"- FP count: {failure_summary.get('fp_count')}\n")
        summary.append(f"- FN count: {failure_summary.get('fn_count')}\n")
        summary.append(f"- Estimated FPS (images): {failure_summary.get('fps_on_images_estimate')}\n")
        summary.append(f"- Failure cases folder: {failure_out_dir}\n")
    summary_path.write_text("".join(summary), encoding="utf-8")

    # 5) Update docs/report_requirements.md (best effort: dataset stats + train config + paths)
    report_md_path = Path(__file__).resolve().parents[1] / "docs" / "report_requirements.md"

    def pick_metric(results_obj: dict | None, patterns: list[str]) -> float | None:
        if not isinstance(results_obj, dict):
            return None
        for k, v in results_obj.items():
            key_str = str(k)
            for p in patterns:
                if re.search(p, key_str, flags=re.IGNORECASE):
                    try:
                        return float(v)
                    except Exception:
                        continue
        return None

    mAP50_95 = pick_metric(results_dict, [r"mAP50-95"])
    # There are multiple key spellings across ultralytics versions; we match broadly.
    mAP50 = pick_metric(results_dict, [r"mAP50([^0-9]|$)", r"mAP50\)"])
    precision = pick_metric(results_dict, [r"precision"])
    recall = pick_metric(results_dict, [r"recall"])
    f1 = pick_metric(results_dict, [r"\bf1\b"])

    insert_ch3 = (
        f"- Dataset (person-only, class `person`): train images={n_train_images}, val images={n_valid_images}; "
        f"person instances (labels) train={n_train_instances}, val={n_valid_instances}.\n"
        f"- Dataset build: lọc từ `roboflow_downloads` giữ class source_id=1 (person) và remap về id=0; tạo `roboflow_downloads_person_only`.\n"
        f"- Hyperparameters: model={args.model}, epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, device={device}.\n"
        f"- Output logs: Ultralytics run dir: {train_dir}\n"
    )

    metrics_bits = []
    if mAP50 is not None:
        metrics_bits.append(f"mAP@0.5={mAP50:.4f}")
    if mAP50_95 is not None:
        metrics_bits.append(f"mAP@0.5:0.95={mAP50_95:.4f}")
    if precision is not None:
        metrics_bits.append(f"Precision={precision:.4f}")
    if recall is not None:
        metrics_bits.append(f"Recall={recall:.4f}")
    if f1 is not None:
        metrics_bits.append(f"F1={f1:.4f}")
    metrics_str = ", ".join(metrics_bits) if metrics_bits else "See `val_metrics.json` for full metrics."

    insert_ch4 = (
        f"- Validate với `best.pt`: {best_weights}\n"
        f"- Validation metrics: {metrics_str}\n"
        f"- Metrics JSON saved to: {val_json_path}\n"
        f"- Loss convergence plots (results.*) nằm trong: {train_dir}\n"
        f"- Failure cases (FP/FN): ảnh đã sinh vào `{failure_out_dir}` (FP/FN được tính & lưu `summary.json`).\n"
    )
    update_report_requirements_md(report_md_path, insert_ch3, insert_ch4)

    print(f"Summary saved to: {summary_path}")
    print(f"Updated report: {report_md_path}")


if __name__ == "__main__":
    main()

