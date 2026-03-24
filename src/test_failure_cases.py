from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from config import PERSON_ONLY_ROOT, RUNS_DIR


@dataclass(frozen=True)
class BoxXYXY:
    x1: float
    y1: float
    x2: float
    y2: float

    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def iou_xyxy(a: BoxXYXY, b: BoxXYXY) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    denom = a.area() + b.area() - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom


def parse_yolo_label_file(labels_txt: Path, img_w: int, img_h: int) -> list[BoxXYXY]:
    boxes: list[BoxXYXY] = []
    if not labels_txt.exists():
        return boxes
    for ln in labels_txt.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) != 5:
            continue
        # person-only dataset uses cls=0 only; we ignore cls here.
        _, xc, yc, bw, bh = parts
        xc = float(xc)
        yc = float(yc)
        bw = float(bw)
        bh = float(bh)
        x1 = (xc - bw / 2.0) * img_w
        y1 = (yc - bh / 2.0) * img_h
        x2 = (xc + bw / 2.0) * img_w
        y2 = (yc + bh / 2.0) * img_h
        boxes.append(BoxXYXY(x1, y1, x2, y2))
    return boxes


def remap_xyxy_clamp(box: BoxXYXY, w: int, h: int) -> BoxXYXY:
    return BoxXYXY(
        max(0.0, min(box.x1, w - 1)),
        max(0.0, min(box.y1, h - 1)),
        max(0.0, min(box.x2, w - 1)),
        max(0.0, min(box.y2, h - 1)),
    )


def greedy_match(gt: list[BoxXYXY], preds: list[BoxXYXY], match_iou: float) -> tuple[list[bool], list[bool]]:
    """
    Returns:
      - assigned_gt: len(gt), True if matched with some prediction
      - assigned_pred: len(preds), True if matched with some gt
    """
    assigned_gt = [False] * len(gt)
    assigned_pred = [False] * len(preds)

    candidates: list[tuple[float, int, int]] = []  # (iou, pred_idx, gt_idx)
    for pi, pb in enumerate(preds):
        for gi, gb in enumerate(gt):
            ov = iou_xyxy(pb, gb)
            if ov >= match_iou:
                candidates.append((ov, pi, gi))

    candidates.sort(reverse=True, key=lambda x: x[0])
    for ov, pi, gi in candidates:
        if assigned_pred[pi] or assigned_gt[gi]:
            continue
        assigned_pred[pi] = True
        assigned_gt[gi] = True

    return assigned_gt, assigned_pred


def draw_boxes(
    img_bgr,
    gt_boxes: list[BoxXYXY],
    pred_boxes: list[BoxXYXY],
    assigned_gt: list[bool],
    assigned_pred: list[bool],
    out_path: Path,
) -> None:
    import cv2  # type: ignore

    # Colors:
    # - GT matched: green
    # - GT unmatched (FN): red
    # - Pred matched: blue
    # - Pred unmatched (FP): orange
    for i, b in enumerate(gt_boxes):
        c = (0, 255, 0) if assigned_gt[i] else (0, 0, 255)
        cv2.rectangle(img_bgr, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), c, 2)

    for i, b in enumerate(pred_boxes):
        c = (255, 0, 0) if assigned_pred[i] else (0, 165, 255)
        cv2.rectangle(img_bgr, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), c, 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img_bgr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--split", type=str, default="valid", choices=["train", "valid"])
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--out-dir", type=str, default=str(RUNS_DIR / "failure_cases"))
    args = ap.parse_args()

    from ultralytics import YOLO  # type: ignore
    import cv2  # type: ignore

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"--weights not found: {weights_path}")

    img_dir = PERSON_ONLY_ROOT / args.split / "images"
    label_dir = PERSON_ONLY_ROOT / args.split / "labels"
    out_dir = Path(args.out_dir)

    fp_dir = out_dir / "fp"
    fn_dir = out_dir / "fn"
    ok_dir = out_dir / "success"

    model = YOLO(str(weights_path))

    img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    img_paths = img_paths[: args.limit]

    fp_count = 0
    fn_count = 0
    processed = 0
    total_time = 0.0

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        gt_boxes = parse_yolo_label_file(label_dir / f"{img_path.stem}.txt", w, h)

        t0 = time.perf_counter()
        pred = model.predict(
            source=str(img_path),
            conf=args.conf,
            iou=0.6,
            classes=[0],  # person-only
            verbose=False,
        )[0]
        total_time += time.perf_counter() - t0

        processed += 1

        pred_boxes: list[BoxXYXY] = []
        if getattr(pred, "boxes", None) is not None and len(pred.boxes) > 0:
            # pred.boxes.xyxy is Nx4
            for b in pred.boxes.xyxy.tolist():
                pred_boxes.append(remap_xyxy_clamp(BoxXYXY(*b), w, h))

        assigned_gt, assigned_pred = greedy_match(gt_boxes, pred_boxes, match_iou=args.match_iou)

        fn_local = sum(1 for x in assigned_gt if not x)
        fp_local = sum(1 for x in assigned_pred if not x)

        if fn_local > 0 and fp_local > 0:
            # Save into both folders for easy selection in report.
            fn_count += fn_local
            fp_count += fp_local
            draw_boxes(img.copy(), gt_boxes, pred_boxes, assigned_gt, assigned_pred, fn_dir / f"{img_path.stem}.jpg")
            draw_boxes(
                img.copy(),
                gt_boxes,
                pred_boxes,
                assigned_gt,
                assigned_pred,
                fp_dir / f"{img_path.stem}.jpg",
            )
        elif fn_local > 0:
            fn_count += fn_local
            draw_boxes(img, gt_boxes, pred_boxes, assigned_gt, assigned_pred, fn_dir / f"{img_path.stem}.jpg")
        elif fp_local > 0:
            fp_count += fp_local
            draw_boxes(img, gt_boxes, pred_boxes, assigned_gt, assigned_pred, fp_dir / f"{img_path.stem}.jpg")
        else:
            draw_boxes(img, gt_boxes, pred_boxes, assigned_gt, assigned_pred, ok_dir / f"{img_path.stem}.jpg")

    avg_time = total_time / max(1, processed)
    fps = 1.0 / avg_time if avg_time > 0 else None

    summary = {
        "weights": str(weights_path),
        "split": args.split,
        "limit": args.limit,
        "processed_images": processed,
        "fp_count": fp_count,
        "fn_count": fn_count,
        "avg_infer_time_sec": avg_time,
        "fps_on_images_estimate": fps,
        "out_dir": str(out_dir),
        "match_iou": args.match_iou,
        "conf": args.conf,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Done. Saved failure cases to: {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

