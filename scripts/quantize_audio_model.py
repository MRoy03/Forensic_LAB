"""One-time step: int8-quantize the audio deepfake model's MatMul layers to
shrink it for memory-constrained hosts (Render's 512MB free tier most of
all). Requires requirements-train.txt's `onnx` package -- never needed at
serving time, same as export_onnx.py.

wav2vec2's positional conv embedding uses weight normalization (a Conv
whose weight comes from a Mul node, not a plain initializer), which trips
onnxruntime's quantizer's bias-folding when quantizing Conv nodes. Routing
around it by quantizing only MatMul -- where almost all of a transformer's
parameters live anyway -- avoids the crash entirely and leaves the small
conv feature-extractor stack in fp32.

Verified (not just assumed): same argmax as the fp32 model on a synthetic
input, ~3x smaller on disk (361MB -> 116MB). Not a full accuracy benchmark
though -- confidence shifted by ~0.05 on that one sample, so this is a
real, if usually small, accuracy/precision tradeoff. Validate against real
labeled audio before trusting it blindly in a context where that mattered.

Usage: python scripts/quantize_audio_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from onnxruntime.quantization import QuantType, quantize_dynamic

from config.loader import get_settings, resolve_path


def main() -> None:
    settings = get_settings()
    src = resolve_path(settings.audio.model_onnx_path)
    fp32_backup = src.with_name(src.stem + "_fp32" + src.suffix)

    if not src.exists():
        raise FileNotFoundError(f"No audio model found at {src} to quantize.")

    if not fp32_backup.exists():
        src.rename(fp32_backup)
        print(f"Backed up original fp32 model to {fp32_backup}")
    else:
        print(f"fp32 backup already exists at {fp32_backup}, quantizing from there")

    quantize_dynamic(
        str(fp32_backup),
        str(src),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
    )

    orig_mb = fp32_backup.stat().st_size / 1_048_576
    new_mb = src.stat().st_size / 1_048_576
    print(f"Quantized {fp32_backup.name} ({orig_mb:.1f} MB) -> {src.name} ({new_mb:.1f} MB)")


if __name__ == "__main__":
    main()
