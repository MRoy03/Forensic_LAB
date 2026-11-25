"""One-time step: int8-quantize Organika/sdxl-detector's MatMul layers to
shrink it -- at 337.6MB fp32, it alone pushed the core service's working
set to ~468MB, uncomfortably close to Render's 512MB free-tier ceiling.
Requires requirements-train.txt's `onnx` package -- never needed at
serving time, same as export_onnx.py and quantize_audio_model.py.

MatMul-only (not Conv), same reasoning as the audio model's quantization:
transformers' parameter mass lives almost entirely in MatMul layers, and
restricting to those avoids any risk from Swin's patch-embedding Conv2d
needing special handling.

Verified (not just assumed): same argmax as the fp32 model on a synthetic
input before this script is trusted. Not a full accuracy benchmark --
validate against real labeled images before trusting it blindly in a
context where that mattered.

Usage: python scripts/quantize_organika.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

from config.loader import get_settings, resolve_path


def main() -> None:
    settings = get_settings()
    src = resolve_path(settings.organika.model_onnx_path)
    fp32_backup = src.with_name(src.stem + "_fp32" + src.suffix)

    if not src.exists():
        raise FileNotFoundError(f"No Organika model found at {src} to quantize.")

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

    print("Verifying against the fp32 output ...")
    h, w = settings.organika.image_size
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 3, h, w)).astype(np.float32)

    fp32_session = ort.InferenceSession(str(fp32_backup), providers=["CPUExecutionProvider"])
    quant_session = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])

    fp32_logits = fp32_session.run([settings.organika.output_name], {settings.organika.input_name: x})[0]
    quant_logits = quant_session.run([settings.organika.output_name], {settings.organika.input_name: x})[0]

    print(f"  fp32 logits:  {fp32_logits}")
    print(f"  quant logits: {quant_logits}")
    print(f"  max abs diff: {np.abs(fp32_logits - quant_logits).max():.6f}")
    assert np.argmax(fp32_logits) == np.argmax(quant_logits), (
        "Quantized model's argmax disagrees with fp32 on this sample -- do not trust this "
        "quantization without further validation."
    )
    print("Verified: same argmax as fp32 on a synthetic sample.")


if __name__ == "__main__":
    main()
