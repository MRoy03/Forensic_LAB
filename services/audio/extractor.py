"""Decode uploaded audio bytes into a 16kHz mono float32 waveform.

Handles WAV/FLAC/OGG directly via soundfile (libsndfile), with no ffmpeg
dependency -- extracting audio from a video container would need the
ffmpeg binary as a separate system-level install, which this module
deliberately does not assume.
"""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf
from scipy.signal import resample


def decode_audio_bytes(data: bytes, target_sample_rate: int) -> np.ndarray:
    """Raw uploaded bytes -> mono float32 waveform at target_sample_rate."""
    try:
        waveform, source_sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:
        raise ValueError(
            "Could not decode audio bytes (unsupported or corrupt format; "
            "try WAV, FLAC, or OGG)"
        ) from exc

    mono = waveform.mean(axis=1)

    if source_sample_rate != target_sample_rate:
        num_target_samples = round(len(mono) * target_sample_rate / source_sample_rate)
        mono = resample(mono, num_target_samples)

    return mono.astype(np.float32)
