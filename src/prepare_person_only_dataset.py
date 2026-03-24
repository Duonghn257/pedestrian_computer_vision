from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from config import (
    PERSON_CLASS_ID_IN_SOURCE,
    PERSON_ONLY_DATA_YAML,
    PERSON_ONLY_ROOT,
    ROBofLOW_DATASET_ROOT,
)


def _safe_symlink_or_copy(src: Path, dst: Path) -> None:
    """
    Prefer symlinks to avoid duplicating images; fall back to copytree if symlink fails.
    """
    if dst.exists():
        return
    try:
        os.symlink(src, dst, target_is_directory=True)
    except OSError:
        shutil.copytree(src, dst)


def _filter_yolo_labels(
    src_labels_dir: Path,
    dst_labels_dir: Path,
    keep_class_id: int,
) -> None:
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    for lbl_path in src_labels_dir.glob("*.txt"):
        out_path = dst_labels_dir / lbl_path.name
        kept_lines: list[str] = []
        for ln in lbl_path.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) != 5:
                continue
            cls = int(float(parts[0]))
            if cls == keep_class_id:
                # Re-map to single-class id = 0 for YOLO training.
                parts[0] = "0"
                kept_lines.append(" ".join(parts))

        # Write even if empty so YOLO sees "no objects" consistently.
        out_path.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""))


def build_person_only_dataset(
    keep_class_id: int = PERSON_CLASS_ID_IN_SOURCE,
) -> Path:
    """
    Creates `roboflow_downloads_person_only/` with train/valid splits containing only `person` labels.
    """
    if not ROBofLOW_DATASET_ROOT.exists():
        raise FileNotFoundError(f"Missing source dataset root: {ROBofLOW_DATASET_ROOT}")

    if PERSON_ONLY_DATA_YAML.exists():
        return PERSON_ONLY_DATA_YAML

    for split in ["train", "valid"]:
        src_split_images = ROBofLOW_DATASET_ROOT / split / "images"
        src_split_labels = ROBofLOW_DATASET_ROOT / split / "labels"

        dst_split_images = PERSON_ONLY_ROOT / split / "images"
        dst_split_labels = PERSON_ONLY_ROOT / split / "labels"

        if not src_split_images.exists() or not src_split_labels.exists():
            raise FileNotFoundError(
                f"Missing split data: {split}. "
                f"Expected {src_split_images} and {src_split_labels}."
            )

        _safe_symlink_or_copy(src_split_images, dst_split_images)
        _filter_yolo_labels(src_split_labels, dst_split_labels, keep_class_id=keep_class_id)

    PERSON_ONLY_ROOT.mkdir(parents=True, exist_ok=True)

    # Create YOLO data.yaml for Ultralytics.
    # Note: we omit `test` because the exported dataset only provides train/valid.
    data_yaml = f"""\
names:
  - person
nc: 1
train: {PERSON_ONLY_ROOT / "train" / "images"}
val: {PERSON_ONLY_ROOT / "valid" / "images"}
"""
    PERSON_ONLY_DATA_YAML.write_text(data_yaml)
    return PERSON_ONLY_DATA_YAML


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-class-id", type=int, default=PERSON_CLASS_ID_IN_SOURCE)
    args = ap.parse_args()

    out_yaml = build_person_only_dataset(keep_class_id=args.keep_class_id)
    print(f"Built person-only dataset: {out_yaml}")


if __name__ == "__main__":
    main()

