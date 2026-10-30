# ForensicAI Lab

A modular AI-powered image and video forgery detection platform:

1. Real/Fake Image Classification
2. Manipulation Type Detection
3. Tampered Region Localization
4. Explainable AI (Eigen-CAM -- see [Explainability](#explainability-eigen-cam) below)
5. Video Forgery Analysis
6. Synthetic Manipulation Generation (Playground)
7. Educational Quiz Mode
8. User Feedback Collection
9. Continuous Dataset Expansion
10. Future CCTV/Anomaly Module Integration (not built yet, see [Future Roadmap](#future-roadmap))

## Architecture

**Pattern:** Modular monolith -- one FastAPI application, cleanly separated
into `services/*` packages so individual pieces can be split into
microservices later without a rewrite.

**Current state:** every module is scaffolded and runnable end-to-end today,
using a **randomly-initialized model** (no training has happened yet). See
[Current Limitations](#current-limitations) and [Training Roadmap](#training-roadmap--notebooks).

## Resource & Cost Constraints

This is a personal project built to run at **$0 cost** and on a **low-spec
local machine**. Two deliberate architecture decisions follow from that:

- **The served app never requires PyTorch.** Inference runs entirely on
  **ONNX Runtime** (`onnxruntime`, CPU execution provider) -- a much lighter
  install and lower memory footprint than torch+timm. PyTorch is confined to
  `services/detection/model.py` (architecture definition) and
  `services/detection/export_onnx.py` / the Colab notebooks (training and
  export). `requirements.txt` (what you install locally to run the app) is
  torch-free; `requirements-train.txt` (torch, torchvision, timm) is only
  installed inside Colab/Kaggle notebooks, or once to (re-)export a model.
- **Explainability is Eigen-CAM, not gradient-based Grad-CAM.** The usual
  `pytorch-grad-cam` library needs a PyTorch backward pass, which doesn't
  work against an ONNX Runtime session. `services/explainability/gradcam.py`
  instead implements **Eigen-CAM** (Muhammad & Yeasin, 2020): it takes the
  model's last feature-map output (already produced by the same forward
  pass used for classification) and computes its dominant principal
  component via `numpy.linalg.svd` -- a gradient-free saliency heatmap with
  no extra compute. If you ever run this on a machine with PyTorch
  available at serving time, swap that file for `pytorch_grad_cam.GradCAM`
  or `ScoreCAM` for class-discriminative heatmaps instead.
- Training happens exclusively on **Colab/Kaggle free tiers** -- never
  locally, never on paid compute. See `notebooks/`.

## Tech Stack

| Layer | Tools |
|---|---|
| Serving inference | ONNX Runtime |
| Training (notebooks only) | PyTorch, torchvision, timm |
| Backend | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Image/video processing | OpenCV, NumPy, scikit-image |
| Data | Pandas |
| Training environments | Google Colab, Kaggle (free tiers) |

## Directory Structure

```
ICNNP/
├── app/                   FastAPI app: main.py, routes.py, schemas.py
├── services/
│   ├── preprocessing/     ELA, FFT, RGB+ELA+FFT stacking pipeline
│   ├── detection/         model (torch, training-side), loader/classifier/segmentor/predictor (onnxruntime, serving-side), export_onnx.py
│   ├── explainability/    Eigen-CAM
│   ├── video/             frame extraction, sampling, per-frame inference, aggregation
│   ├── playground/        synthetic manipulation generators + dispatch controller
│   ├── feedback/          feedback collection, CSV storage, retrain-manifest merging
│   └── postprocessing/    mask/heatmap overlays, base64 encoding
├── frontend/streamlit_app.py
├── data/                  images/ masks/ labels.csv synthetic_data/ feedback_data/
├── models/                exported .onnx (serving) + .pth (training) checkpoints
├── notebooks/             Colab/Kaggle training notebooks, Stage 1-6
├── config/                settings.yaml + typed loader
├── tests/                 pytest suite (torch-free)
├── scripts/smoke_test.py  end-to-end HTTP smoke test
├── requirements.txt       serving deps (no torch)
└── requirements-train.txt training deps (torch, notebooks + export only)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

This installs only serving dependencies -- no PyTorch, no large downloads.
The app ships with `models/image_model.onnx` already exported from a
randomly-initialized model, so it runs immediately.

## Configuration

All paths, thresholds, and labels live in `config/settings.yaml`, parsed by
`config/loader.py` into typed Pydantic settings (`get_settings()`). Notable
settings:

- `inference.*` -- ONNX Runtime execution provider, image size, input/output
  tensor names.
- `models.*` -- `.onnx` paths (serving) and `.pth` paths (training).
- `video.fake_frame_ratio_threshold` -- the 0.30 decision-rule threshold.
- `data.root_local` / `data.root_drive` -- dataset root resolves to the
  Drive-mounted path automatically when running in Colab, else falls back to
  local.

## Running the App

```bash
uvicorn app.main:app --reload
streamlit run frontend/streamlit_app.py   # in a second terminal
```

## API Reference

| Endpoint | Description |
|---|---|
| `GET /health` | `{"status": "ok", "model_loaded": true, "runtime": "onnxruntime"}` |
| `POST /predict/image` | multipart `file` -> classification + mask + Eigen-CAM heatmap |
| `POST /predict/video` | multipart `file` -> per-frame results + aggregated verdict |
| `POST /playground/generate` | multipart `file` + form `manipulation_type` (+ optional `params` JSON) -> synthetic manipulated image |
| `POST /feedback` | JSON body -> stores a feedback record |

Example:

```bash
curl -F "file=@photo.jpg" http://localhost:8000/predict/image
```

```json
{
  "label": "fake",
  "confidence": 0.94,
  "manipulation_type": "splicing",
  "mask": "<base64 PNG>",
  "gradcam": "<base64 PNG>",
  "explanation": "Edge and lighting inconsistencies detected at a region boundary, consistent with splicing."
}
```

## Model Architecture

`services/detection/model.py::ForensicMultiHeadNet`: a shared EfficientNet-B0
encoder (via `timm`, 5 input channels) feeding three parallel heads:

```
Input (5, 256, 256)  [RGB + ELA + FFT]
   -> EfficientNet-B0 encoder
   -> shared pooled features ----> Classification Head (real/fake)
                          \-------> Manipulation-Type Head (6 classes)
   -> multi-scale features -------> Segmentation Decoder (tampered-area mask)
```

A fourth output, `feature_map` (the encoder's last stage), is exported
alongside the three prediction heads specifically to feed Eigen-CAM at
serving time.

## Preprocessing Pipeline

```
Input image -> Resize 256x256 -> Normalize
            -> Compute ELA   (services/preprocessing/ela.py)
            -> Compute FFT   (services/preprocessing/fft.py)
            -> Stack [R, G, B, ELA, FFT] -> (5, 256, 256) array
```

Pure NumPy/OpenCV, shared identically between the live app and the training
notebooks.

## Explainability (Eigen-CAM)

See [Resource & Cost Constraints](#resource--cost-constraints) above for why
this is Eigen-CAM rather than gradient-based Grad-CAM. **On a
randomly-initialized model, the heatmap will look like generic texture/edge
saliency, not a trained "why this is fake" signal -- this is expected, not a
bug**, and will sharpen once a trained model is exported (see
[Training Roadmap](#training-roadmap--notebooks)).

## Video Pipeline & Decision Rule

```
Video -> extract frames -> sample at ~2 FPS (capped at max_frames_to_process)
      -> classify each sampled frame -> aggregate

fake_frames / total_frames_sampled > 0.30  =>  video = fake
otherwise                                  =>  video = real
```

## Playground Module

| Type | Implementation |
|---|---|
| `copy_move` | Real: random patch duplication within the same image |
| `splicing` | Real: `cv2.seamlessClone` Poisson blending |
| `compression` | Real: aggressive JPEG re-encode |
| `blur` | Real: Gaussian blur, whole-image or region |
| `beautify` | Real: bilateral filter + tone adjustment |
| `upscale` | **Placeholder**: `cv2.resize` with `INTER_CUBIC`, standing in for a future super-resolution model |

Used for the Quiz Mode, manual testing, and Stage 4 synthetic dataset
expansion.

## Quiz Mode

In the Streamlit app: generates a random (or unmanipulated) image via the
playground, collects your real/fake + manipulation-type guess, then calls
`/predict/image` on the same image to reveal ground truth vs. your guess vs.
the AI's prediction, heatmap, and mask.

## Feedback & Retraining

`POST /feedback` stores records (image, prediction, correction, timestamp)
in `data/feedback_data/feedback_log.csv`. `services/feedback/retrain.py`
folds corrected rows into `data/labels.csv` for the next training run --
it does **not** trigger training itself; that happens in
`notebooks/05_stage5_add_feedback_data.ipynb`.

## Training Roadmap & Notebooks

All training runs on **Colab/Kaggle free tiers** using `requirements-train.txt`.
Each notebook under `notebooks/` reuses the exact same
`services/preprocessing/pipeline.py`, `services/detection/model.py`, and
`services/playground/*` code as the live app, so training and serving stay
consistent.

| Stage | Notebook | Run on | Produces |
|---|---|---|---|
| 0 | `00_colab_setup_and_drive_mount.ipynb` | Colab | Drive mounted, deps installed |
| 0b | `00b_prepare_local_datasets.ipynb` | Colab | Unified `labels.csv` + `images/`/`masks/` from CASIA2.0, CoMoFoD, Coverage, IMD2020, FaceForensics++ (c23) |
| 1 | `01_stage1_train_image_datasets.ipynb` | Colab | `models/image_model.pth` |
| 2 | `02_stage2_dfdc_frame_extraction_and_train.ipynb` | **Kaggle** (DFDC attached via "Add Input") | `models/video_model.pth` |
| 3 | `03_stage3_merge_image_video_datasets.ipynb` | Colab | Merged manifest + checkpoint |
| 4 | `04_stage4_add_synthetic_playground_data.ipynb` | Colab | Synthetic data + checkpoint |
| 5 | `05_stage5_add_feedback_data.ipynb` | Colab | Feedback-augmented checkpoint |
| 6 | `06_stage6_final_export_and_onnx.ipynb` | Colab | `models/final_model.pth` + `.onnx`, verified |

Stages 0/0b/1/3/4/5/6 assume a Drive-backed `dataset_root`; Stage 2 runs on
Kaggle instead (DFDC is a Kaggle dataset, attachable without downloading it
yourself) and its checkpoint needs to be manually copied to Drive afterward
for Stage 3 to pick up -- see the notebook's last cell. Many public DFDC
re-uploads ship as ZIP archives rather than pre-extracted files; Stage 2
extracts one zip part at a time into scratch space, processes it, and deletes
it before moving to the next, so attaching ~100GB across many parts doesn't
exceed Kaggle's working-disk quota. Stage 2 can run
before or after Stage 1; it warm-starts from `image_model.pth` if present,
otherwise trains from random weights.

After Stage 6, copy `models/final_model.onnx` over `models/image_model.onnx`
(or update `config/settings.yaml`) to make the live app use it -- no serving
code changes required.

## Dataset Registry

Not bundled -- download these yourself and point `config/settings.yaml`'s
`data.root_local` / `data.root_drive` at them:

- **Images:** CASIA 2.0, CoMoFoD, Coverage, IMD2020
- **Video:** DFDC (subset, via Kaggle), FaceForensics++ (c23, via a Kaggle mirror --
  note the official FF++ release is distributed by TUM, not Meta, and normally
  requires a request-form approval; Kaggle re-uploads are a faster alternative)

`labels.csv` format: `filename,label,type` (see `data/labels.csv` for
examples).

## Current Limitations

- No trained weights yet -- `models/image_model.onnx` was exported from a
  **randomly-initialized** architecture, purely so the app is runnable out
  of the box. Prediction quality will be near-random until trained via the
  notebooks above.
- Eigen-CAM heatmaps are class-agnostic saliency, not true class-discriminative
  Grad-CAM (see [Explainability](#explainability-eigen-cam)).
- `upscale` in the Playground is a plain resize placeholder, not a real
  super-resolution model.
- The video pipeline reuses the single image model per-frame; there's no
  dedicated temporal/video-specific architecture yet.

## Future Roadmap

- Phase 2: YOLO-based object detection
- Phase 3: CCTV anomaly detection
- Phase 4: Live stream analysis
- Phase 5: Full microservice deployment

## Testing

```bash
pytest -q                                   # unit/integration tests (torch-free)
python scripts/smoke_test.py                # end-to-end HTTP smoke test (requires a running server)
```
