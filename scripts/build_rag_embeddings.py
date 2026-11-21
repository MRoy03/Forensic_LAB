"""One-time offline step: embed every passage in
data/knowledge_base/forgery_techniques.json with the MiniLM ONNX model and
commit the resulting small embeddings.npy. No torch needed -- the embedder
is already ONNX, so this runs with the same requirements.txt as the served
app. Rerun after editing the knowledge base.

Usage: python scripts/build_rag_embeddings.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config.loader import get_settings, resolve_path
from services.explainability.rag import embed_text, load_knowledge_base


def main() -> None:
    settings = get_settings()
    passages = load_knowledge_base()

    embeddings = np.stack([embed_text(p["text"]) for p in passages])

    out_path = resolve_path(settings.rag.embeddings_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings)

    print(f"Embedded {len(passages)} passages -> {out_path} (shape {embeddings.shape})")


if __name__ == "__main__":
    main()
