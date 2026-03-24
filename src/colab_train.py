from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def ensure_requirements(requirements_path: Path) -> None:
    run_cmd([sys.executable, "-m", "pip", "install", "-U", "pip"])
    run_cmd([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)])


def detect_device() -> str:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Colab-friendly full pipeline: download -> prepare person-only -> train -> val -> failure cases."
    )
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Persistent output root, e.g. /content/drive/MyDrive/cv_pedestrian",
    )
    parser.add_argument("--api-key", type=str, default=os.getenv("ROBOFLOW_API_KEY", ""))
    parser.add_argument("--workspace", type=str, default="new-workspace-5uval")
    parser.add_argument("--project", type=str, default="human-crowd-vbdc9")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--model", type=str, default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--limit-failure", type=int, default=100)
    parser.add_argument("--train-name", type=str, default="person_yolov8_visible_colab")
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_install:
        ensure_requirements(project_root / "requirements.txt")

    # Keep all artifacts in Drive, not Colab temp storage.
    data_dir = output_root / "roboflow_downloads"
    person_only_dir = output_root / "roboflow_downloads_person_only"
    runs_dir = output_root / "runs"
    data_dir.mkdir(parents=True, exist_ok=True)
    person_only_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # 1) Download dataset from Roboflow into output_dir
    prepare_cmd = [
        sys.executable,
        str(project_root / "prepare_data.py"),
        "--api-key",
        args.api_key,
        "--workspace",
        args.workspace,
        "--project",
        args.project,
        "--version",
        str(args.version),
        "--format",
        "yolov8",
        "--output-dir",
        str(data_dir),
    ]
    run_cmd(prepare_cmd)

    # 2) Link persistent folders into project paths expected by existing scripts.
    local_download_link = project_root / "roboflow_downloads"
    if local_download_link.exists() or local_download_link.is_symlink():
        if local_download_link.is_symlink() or local_download_link.is_file():
            local_download_link.unlink()
        else:
            raise RuntimeError(
                f"{local_download_link} already exists as a real directory. "
                "Please remove/backup it first, then re-run."
            )
    if not local_download_link.exists():
        os.symlink(str(data_dir), str(local_download_link), target_is_directory=True)

    local_person_only = project_root / "roboflow_downloads_person_only"
    if local_person_only.exists() or local_person_only.is_symlink():
        if local_person_only.is_symlink() or local_person_only.is_file():
            local_person_only.unlink()
        else:
            raise RuntimeError(
                f"{local_person_only} already exists as a real directory. "
                "Please remove/backup it first, then re-run."
            )
    if not local_person_only.exists():
        os.symlink(str(person_only_dir), str(local_person_only), target_is_directory=True)

    local_runs = project_root / "runs"
    if local_runs.exists() or local_runs.is_symlink():
        if local_runs.is_symlink() or local_runs.is_file():
            local_runs.unlink()
        else:
            raise RuntimeError(
                f"{local_runs} already exists as a real directory. "
                "Please remove/backup it first, then re-run."
            )
    if not local_runs.exists():
        os.symlink(str(runs_dir), str(local_runs), target_is_directory=True)

    # 3) Build person-only dataset (labels filtered)
    run_cmd([sys.executable, str(project_root / "src" / "prepare_person_only_dataset.py")])

    # 4) Train + Val + Failure cases with runs persisted to Drive
    device = detect_device()
    run_cmd(
        [
            sys.executable,
            str(project_root / "src" / "run_full_pipeline.py"),
            "--model",
            args.model,
            "--epochs",
            str(args.epochs),
            "--imgsz",
            str(args.imgsz),
            "--batch",
            str(args.batch),
            "--conf",
            str(args.conf),
            "--iou",
            str(args.iou),
            "--match-iou",
            str(args.match_iou),
            "--limit-failure",
            str(args.limit_failure),
            "--train-name",
            args.train_name,
            "--device",
            device,
        ]
    )

    summary = {
        "system": platform.platform(),
        "python": sys.version,
        "device": device,
        "output_root": str(output_root),
        "runs_dir": str(local_runs),
        "persistent_runs_dir": str(runs_dir),
    }
    (output_root / "colab_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
