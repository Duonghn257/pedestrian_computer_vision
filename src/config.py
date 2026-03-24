from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Source dataset exported from Roboflow (already downloaded by the user).
ROBofLOW_DATASET_ROOT = PROJECT_ROOT / "roboflow_downloads"

# Output dataset we will build for training (only class "person", re-mapped to id=0).
PERSON_ONLY_ROOT = PROJECT_ROOT / "roboflow_downloads_person_only"
PERSON_ONLY_DATA_YAML = PERSON_ONLY_ROOT / "data_person_only.yaml"

# Training runs created by Ultralytics.
RUNS_DIR = PROJECT_ROOT / "runs"

# In the current Roboflow export there are 2 classes in data.yaml:
#   names: [head, person]
# Based on bbox size statistics (person boxes are much larger), we map:
#   head -> 0
#   person -> 1
PERSON_CLASS_ID_IN_SOURCE = 1

