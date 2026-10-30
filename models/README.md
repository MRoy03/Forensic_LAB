# models/

This directory holds exported/trained model artifacts. None are committed to
git (see `.gitignore`) since they're large binaries.

## Expected files

| File | Produced by | Used by |
|---|---|---|
| `image_model.onnx` | `services/detection/export_onnx.py` (training checkpoint, or random weights if none yet) | The live app (`services/detection/loader.py`), via ONNX Runtime |
| `image_model.pth` | `notebooks/01_stage1_train_image_datasets.ipynb` | `export_onnx.py` (training-side only) |
| `video_model.pth` / `video_model.onnx` | Stage 2-3 notebooks | Reserved for a dedicated video-specific checkpoint; the current scaffold reuses `image_model.onnx` for per-frame video inference |
| `final_model.pth` / `final_model.onnx` | `notebooks/06_stage6_final_export_and_onnx.ipynb` | The fully-trained, production checkpoint after all 6 training stages |

Note: `image_model.onnx` may be paired with an `image_model.onnx.data`
file (PyTorch's ONNX exporter stores weights externally by default on some
versions). Keep both files together -- `onnxruntime` loads the `.data` file
automatically by reference from the `.onnx` file. Both are gitignored.

## Getting started without any trained weights

The app ships with `image_model.onnx` already exported from a
**randomly-initialized** `ForensicMultiHeadNet` (see `services/detection/model.py`),
so `/predict/image`, `/predict/video`, and the Streamlit UI all work
out of the box. Predictions will be low-quality/noisy until real weights are
trained -- that's expected, not a bug.

## Dropping in trained weights later

After training in Colab/Kaggle (see `notebooks/`), copy the resulting
`.onnx` file over the matching path above (or update the path in
`config/settings.yaml`). No serving code changes are required -- the ONNX
Runtime session is loaded by path alone.
