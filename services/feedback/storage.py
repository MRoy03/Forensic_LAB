"""Append feedback records to a CSV log and optionally save the submitted image."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.loader import resolve_path

_COLUMNS = [
    "feedback_id",
    "timestamp",
    "image_path",
    "feedback_type",
    "predicted_label",
    "predicted_manipulation_type",
    "is_correct",
    "correct_label",
    "correct_manipulation_type",
    "comments",
]


def save_feedback(record: dict, image_bytes: bytes | None, log_csv: str, images_dir: str) -> None:
    log_path = resolve_path(log_csv)
    images_path = resolve_path(images_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(parents=True, exist_ok=True)

    image_rel_path = ""
    if image_bytes is not None:
        image_file = images_path / f"{record['feedback_id']}.png"
        image_file.write_bytes(image_bytes)
        image_rel_path = str(image_file)

    row = {col: record.get(col, "") for col in _COLUMNS}
    row["image_path"] = image_rel_path

    df_row = pd.DataFrame([row], columns=_COLUMNS)
    write_header = not log_path.exists()
    df_row.to_csv(log_path, mode="a", header=write_header, index=False)
