"""One-time export of Organika/sdxl-detector (raw PyTorch safetensors
weights, parked at models/pending/organika_sdxl_detector.safetensors since
Phase 1 -- that HF repo ships no ONNX export of its own) to a self-contained
ONNX file for serving. Requires requirements-train.txt's torch stack plus
transformers/safetensors -- never needed at serving time, same discipline
as services/detection/export_onnx.py and scripts/quantize_audio_model.py.

Architecture/config (SwinForImageClassification, num_labels=2, image_size
224) and preprocessing (ImageNet normalize, 224x224) are taken directly
from the HF repo's config.json/preprocessor_config.json, not guessed --
fetched and cross-checked during Phase 1's model-prerequisites research.

Usage: python scripts/convert_organika_to_onnx.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import SwinConfig, SwinForImageClassification

from config.loader import resolve_path

SAFETENSORS_PATH = "models/pending/organika_sdxl_detector.safetensors"
OUTPUT_PATH = "models/ensemble/organika_sdxl_detector.onnx"
INPUT_SIZE = 224

# From Organika/sdxl-detector's config.json on Hugging Face.
SWIN_CONFIG = SwinConfig(
    image_size=224,
    patch_size=4,
    num_channels=3,
    embed_dim=128,
    depths=[2, 2, 18, 2],
    num_heads=[4, 8, 16, 32],
    window_size=7,
    mlp_ratio=4.0,
    qkv_bias=True,
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0,
    drop_path_rate=0.1,
    hidden_act="gelu",
    use_absolute_embeddings=False,
    initializer_range=0.02,
    layer_norm_eps=1e-5,
    encoder_stride=32,
    num_labels=2,
    id2label={0: "artificial", 1: "human"},
    label2id={"artificial": 0, "human": 1},
)


class _LogitsOnlyWrapper(nn.Module):
    """SwinForImageClassification returns a SequenceClassifierOutput
    dataclass -- torch.onnx.export traces a plain tensor output far more
    reliably, same reasoning as export_onnx.py's _TupleOutputWrapper."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.base_model(pixel_values=pixel_values).logits


def main() -> None:
    ckpt_path = resolve_path(SAFETENSORS_PATH)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

    print(f"Loading {ckpt_path} ...")
    state_dict = load_file(str(ckpt_path))
    model = SwinForImageClassification(SWIN_CONFIG)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"  missing keys ({len(missing)}): {missing}")
        print(f"  unexpected keys ({len(unexpected)}): {unexpected}")
        raise RuntimeError(
            "State dict does not match the model architecture exactly -- refusing to "
            "export. This exact failure mode (attention layers silently staying at "
            "random init while everything else loads fine) is what happened when a "
            "newer transformers version's Swin implementation had renamed attention "
            "parameters vs. what this checkpoint was saved with. Install the "
            "transformers version in the HF repo's config.json (transformers_version) "
            "instead of the latest, and retry."
        )
    model.eval()

    wrapped = _LogitsOnlyWrapper(model)
    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    with torch.no_grad():
        torch_logits = wrapped(dummy_input).numpy()

    out_path = resolve_path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    export_kwargs = dict(
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    try:
        torch.onnx.export(
            wrapped, dummy_input, str(out_path), dynamo=False, external_data=False, **export_kwargs
        )
    except TypeError:
        torch.onnx.export(wrapped, dummy_input, str(out_path), **export_kwargs)

    # Same consolidation as export_onnx.py/quantize_audio_model.py -- belt
    # and suspenders against an external-data sidecar split.
    onnx_model = onnx.load(str(out_path), load_external_data=True)
    onnx.save_model(onnx_model, str(out_path), save_as_external_data=False)
    sidecar = out_path.with_name(out_path.name + ".data")
    if sidecar.exists():
        sidecar.unlink()

    size_mb = out_path.stat().st_size / 1_048_576
    print(f"Exported -> {out_path} ({size_mb:.1f} MB)")

    print("Verifying against the PyTorch output ...")
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(["logits"], {"pixel_values": dummy_input.numpy()})[0]

    max_diff = np.abs(torch_logits - onnx_logits).max()
    print(f"  torch logits:  {torch_logits}")
    print(f"  onnx  logits:  {onnx_logits}")
    print(f"  max abs diff:  {max_diff:.6f}")
    assert max_diff < 1e-3, "ONNX output diverges from PyTorch output -- do not trust this export"
    print("Verified: ONNX output matches PyTorch within tolerance.")


if __name__ == "__main__":
    main()
