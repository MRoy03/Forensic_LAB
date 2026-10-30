"""Shared helpers for the Colab/Kaggle training notebooks.

Import this after `sys.path.insert(0, project_root)` from within a notebook,
e.g.:
    import sys
    sys.path.insert(0, "/content/ForensicAI_Lab")  # after cloning/uploading
    from notebooks.utils.colab_common import mount_drive, install_requirements, get_project_root
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def is_colab() -> bool:
    return os.path.exists("/content")


def mount_drive(mount_point: str = "/content/drive") -> None:
    if not is_colab():
        print("Not running in Colab; skipping Drive mount.")
        return
    from google.colab import drive  # type: ignore

    drive.mount(mount_point)


def install_requirements(project_root: str, requirements_file: str = "requirements-train.txt") -> None:
    req_path = Path(project_root) / requirements_file
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(req_path)])


def get_project_root() -> Path:
    """Best-effort project root detection: the parent of notebooks/, whether
    running locally, in Colab (after cloning/uploading), or in Kaggle."""
    here = Path(__file__).resolve()
    return here.parent.parent.parent


def set_dataset_root(drive_path: str, local_path: str = "./data") -> Path:
    """Mirrors config.loader.resolve_dataset_root for use before the project
    package is necessarily importable (e.g. very first setup cell)."""
    drive_root = Path(drive_path)
    if is_colab() and drive_root.exists():
        return drive_root
    local_root = Path(local_path)
    local_root.mkdir(parents=True, exist_ok=True)
    return local_root
