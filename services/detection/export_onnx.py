"""Export ForensicMultiHeadNet to ONNX for torch-free serving.

Usable standalone:
    python services/detection/export_onnx.py --output models/image_model.onnx
    python services/detection/export_onnx.py --checkpoint models/image_model.pth --output models/image_model.onnx

Also called from the Stage 6 training notebook after a real training run.
If --checkpoint is omitted or the file doesn't exist, exports the
randomly-initialized architecture -- this is expected and intentional until
a real checkpoint exists; the served app only cares that the .onnx file's
input/output shapes and names are correct.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn

from config.loader import get_settings, resolve_path
from services.detection.model import ForensicMultiHeadNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_NAMES = ["cls_logits", "manip_logits", "mask_logits", "feature_map"]


class _TupleOutputWrapper(nn.Module):
    """torch.onnx.export traces tuple outputs far more reliably than dict
    outputs across torch versions, so wrap the dict-returning model."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor):
        out = self.base_model(x)
        return tuple(out[name] for name in OUTPUT_NAMES)


def build_model(checkpoint_path: str | None = None) -> nn.Module:
    settings = get_settings()
    model = ForensicMultiHeadNet(
        backbone=settings.models.backbone,
        in_chans=settings.inference.input_channels,
        num_manip_classes=len(settings.labels.manipulation_types),
        pretrained=settings.models.pretrained_encoder,
    )

    if checkpoint_path:
        resolved = resolve_path(checkpoint_path)
        if resolved.exists():
            state_dict = torch.load(resolved, map_location="cpu")
            model.load_state_dict(state_dict)
            logger.info("Loaded checkpoint from %s", resolved)
        else:
            logger.warning(
                "Checkpoint %s not found; exporting randomly-initialized weights", resolved
            )
    else:
        logger.warning("No checkpoint given; exporting randomly-initialized weights")

    model.eval()
    return model


def export_to_onnx(
    output_path: str,
    checkpoint_path: str | None = None,
    input_shape: tuple[int, int, int, int] | None = None,
) -> Path:
    settings = get_settings()
    h, w = settings.inference.image_size
    c = settings.inference.input_channels
    input_shape = input_shape or (1, c, h, w)

    model = build_model(checkpoint_path)
    wrapped = _TupleOutputWrapper(model)

    dummy_input = torch.randn(*input_shape)
    out_path = resolve_path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapped,
        dummy_input,
        str(out_path),
        input_names=[settings.inference.input_name],
        output_names=OUTPUT_NAMES,
        dynamic_axes={
            settings.inference.input_name: {0: "batch"},
            **{name: {0: "batch"} for name in OUTPUT_NAMES},
        },
        opset_version=17,
    )
    logger.info("Exported ONNX model to %s", out_path)
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ForensicMultiHeadNet to ONNX")
    parser.add_argument("--checkpoint", default=None, help="Path to a .pth checkpoint (optional)")
    parser.add_argument("--output", default="models/image_model.onnx", help="Output .onnx path")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    export_to_onnx(output_path=args.output, checkpoint_path=args.checkpoint)
