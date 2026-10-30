"""PyTorch model architecture: EfficientNet-B0 encoder + three parallel heads.

Training/export side only -- imported by export_onnx.py and the training
notebooks, never by app/ or the serving path (services/detection/loader.py
loads the exported .onnx file with onnxruntime instead). Keeping torch
confined to this file and export_onnx.py is what lets the served app run
without installing PyTorch locally.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationDecoder(nn.Module):
    """Light upsampling decoder that turns multi-scale encoder feature maps
    back into a single-channel tampered-area mask at the input resolution."""

    def __init__(self, encoder_channels: list[int], decoder_channels: int = 64):
        super().__init__()
        # Use the last three stages (coarsest -> finer) for a small U-Net-style decoder.
        c3, c2, c1 = encoder_channels[-1], encoder_channels[-2], encoder_channels[-3]

        self.up1 = nn.ConvTranspose2d(c3, decoder_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(decoder_channels + c2, decoder_channels, kernel_size=3, padding=1)

        self.up2 = nn.ConvTranspose2d(decoder_channels, decoder_channels, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(decoder_channels + c1, decoder_channels, kernel_size=3, padding=1)

        self.head = nn.Conv2d(decoder_channels, 1, kernel_size=1)

    def forward(self, feats: list[torch.Tensor], output_size: tuple[int, int]) -> torch.Tensor:
        f1, f2, f3 = feats[-3], feats[-2], feats[-1]

        x = self.up1(f3)
        x = F.relu(x)
        x = torch.cat([x, F.interpolate(f2, size=x.shape[-2:], mode="nearest")], dim=1)
        x = F.relu(self.conv1(x))

        x = self.up2(x)
        x = F.relu(x)
        x = torch.cat([x, F.interpolate(f1, size=x.shape[-2:], mode="nearest")], dim=1)
        x = F.relu(self.conv2(x))

        mask_logits = self.head(x)
        return F.interpolate(mask_logits, size=output_size, mode="bilinear", align_corners=False)


class ForensicMultiHeadNet(nn.Module):
    """Shared EfficientNet-B0 encoder feeding:
    - Head 1: real/fake binary classification
    - Head 2: manipulation-type classification
    - Head 3: tampered-area segmentation
    Also exposes the last encoder feature map as a fourth output, used by
    Eigen-CAM for gradient-free explainability at serving time.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        in_chans: int = 5,
        num_manip_classes: int = 6,
        pretrained: bool = False,
    ):
        super().__init__()
        self.encoder = timm.create_model(
            backbone, features_only=True, in_chans=in_chans, pretrained=pretrained
        )
        channels = self.encoder.feature_info.channels()

        self.pool = nn.AdaptiveAvgPool2d(1)
        last_channels = channels[-1]

        self.cls_head = nn.Sequential(
            nn.Linear(last_channels, 128), nn.ReLU(), nn.Linear(128, 2)
        )
        self.manip_head = nn.Sequential(
            nn.Linear(last_channels, 128), nn.ReLU(), nn.Linear(128, num_manip_classes)
        )
        self.seg_head = SegmentationDecoder(encoder_channels=channels)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.encoder(x)
        last_feat = feats[-1]

        pooled = self.pool(last_feat).flatten(1)

        cls_logits = self.cls_head(pooled)
        manip_logits = self.manip_head(pooled)
        mask_logits = self.seg_head(feats, output_size=x.shape[-2:])

        return {
            "cls_logits": cls_logits,
            "manip_logits": manip_logits,
            "mask_logits": mask_logits,
            "feature_map": last_feat,
        }
