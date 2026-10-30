"""Folds corrected user feedback into the training manifest (data/labels.csv).

Only merges the manifest row -- it does not trigger any actual training. Real
retraining happens in notebooks/05_stage5_add_feedback_data.ipynb, which
reads the updated labels.csv produced here.
"""

from __future__ import annotations

import pandas as pd

from config.loader import resolve_path

_MERGED_IDS_SIDECAR = "data/feedback_data/_merged_feedback_ids.txt"
_LABELS_COLUMNS = ["filename", "label", "type"]


def _read_merged_ids(sidecar_path) -> set[str]:
    if not sidecar_path.exists():
        return set()
    return set(sidecar_path.read_text(encoding="utf-8").splitlines())


def append_to_training_manifest(
    feedback_csv: str = "data/feedback_data/feedback_log.csv",
    labels_csv: str = "data/labels.csv",
) -> int:
    feedback_path = resolve_path(feedback_csv)
    labels_path = resolve_path(labels_csv)
    sidecar_path = resolve_path(_MERGED_IDS_SIDECAR)

    if not feedback_path.exists():
        return 0

    feedback_df = pd.read_csv(feedback_path, dtype=str).fillna("")
    merged_ids = _read_merged_ids(sidecar_path)

    corrected = feedback_df[
        (feedback_df["correct_label"] != "") & (~feedback_df["feedback_id"].isin(merged_ids))
    ]
    if corrected.empty:
        return 0

    new_rows = []
    for _, row in corrected.iterrows():
        label = row["correct_label"]
        manip_type = row["correct_manipulation_type"] or ("none" if label == "real" else "unknown")
        new_rows.append(
            {
                "filename": row["image_path"] or f"feedback/{row['feedback_id']}.png",
                "label": label,
                "type": manip_type,
            }
        )

    new_df = pd.DataFrame(new_rows, columns=_LABELS_COLUMNS)

    if labels_path.exists():
        existing_df = pd.read_csv(labels_path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        combined = new_df

    combined.to_csv(labels_path, index=False)

    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sidecar_path, "a", encoding="utf-8") as f:
        for feedback_id in corrected["feedback_id"]:
            f.write(f"{feedback_id}\n")

    return len(new_rows)
