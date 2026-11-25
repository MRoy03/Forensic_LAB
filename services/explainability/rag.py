"""Retrieval-only RAG explanations (onnx-community/all-MiniLM-L6-v2-ONNX).

Pure retrieval, no generative LLM call -- zero hallucination risk, and keeps
serving torch-free. Replaces predictor.py's old hardcoded _EXPLANATIONS dict
with a cosine-similarity match against a small curated knowledge base
(data/knowledge_base/forgery_techniques.json), using precomputed embeddings
(data/knowledge_base/embeddings.npy, built offline by
scripts/build_rag_embeddings.py since this ONNX model has no torch
dependency even at build time).
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from config.loader import get_settings, is_cached, resolve_path

logger = logging.getLogger(__name__)


@lru_cache
def get_rag_session() -> ort.InferenceSession:
    settings = get_settings()
    path = resolve_path(settings.rag.embedder_onnx_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No RAG embedder ONNX model found at {path}. Download "
            "onnx-community/all-MiniLM-L6-v2-ONNX's model.onnx (and its "
            "external-data sidecar, named model.onnx_data) and place it there."
        )
    session = ort.InferenceSession(str(path), providers=[settings.inference.execution_provider])
    logger.info("Loaded RAG embedder ONNX model from %s", path)
    return session


def is_rag_loaded() -> bool:
    """Passive check: has this already been loaded, without triggering a
    load as a side effect -- only ever called from /health. Worth knowing:
    explain() (used by every image/audio prediction) does a direct
    knowledge-base dict lookup and never actually touches this embedder at
    all -- only retrieve() does, which nothing currently calls in the
    serving path -- so before this fix, /health was loading an 86MB model
    that normal serving never even uses, just by being polled."""
    return is_cached(get_rag_session)


@lru_cache
def get_tokenizer() -> Tokenizer:
    settings = get_settings()
    path = resolve_path(settings.rag.tokenizer_path)
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.enable_truncation(max_length=settings.rag.max_seq_length)
    return tokenizer


def _mean_pool(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """token_embeddings: (1, seq_len, hidden). attention_mask: (1, seq_len)."""
    mask = attention_mask[:, :, np.newaxis].astype(np.float32)
    summed = (token_embeddings * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    return summed / counts


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, 1e-9, None)


def embed_text(text: str) -> np.ndarray:
    """Single sentence/query -> L2-normalized (hidden_size,) embedding."""
    session = get_rag_session()
    tokenizer = get_tokenizer()

    encoding = tokenizer.encode(text)
    input_ids = np.array([encoding.ids], dtype=np.int64)
    attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
    token_type_ids = np.array([encoding.type_ids], dtype=np.int64)

    outputs = session.run(
        ["last_hidden_state"],
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )
    pooled = _mean_pool(outputs[0], attention_mask)
    return _l2_normalize(pooled)[0]


@lru_cache
def load_knowledge_base() -> list[dict]:
    settings = get_settings()
    path = resolve_path(settings.rag.knowledge_base_path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache
def load_passage_embeddings() -> np.ndarray:
    settings = get_settings()
    path = resolve_path(settings.rag.embeddings_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No precomputed RAG embeddings found at {path}. Run "
            "`python scripts/build_rag_embeddings.py` once to generate them."
        )
    passages = load_knowledge_base()
    embeddings = np.load(path)
    if embeddings.shape[0] != len(passages):
        raise ValueError(
            f"{path} has {embeddings.shape[0]} rows but the knowledge base has "
            f"{len(passages)} passages -- rerun scripts/build_rag_embeddings.py "
            "after editing the knowledge base."
        )
    return embeddings


def retrieve(query: str, modality: str | None = None, top_k: int = 1) -> list[dict]:
    """Cosine-similarity top_k passages for a free-text query, optionally
    restricted to one modality ("image" or "audio"). General-purpose
    semantic search over the knowledge base -- e.g. for a future "explain
    this in more detail" / natural-language question feature. Not used by
    explain() below, which has an exact category id and doesn't need
    similarity search to find it."""
    passages = load_knowledge_base()
    embeddings = load_passage_embeddings()

    if modality is not None:
        keep = [i for i, p in enumerate(passages) if p["modality"] == modality]
        passages = [passages[i] for i in keep]
        embeddings = embeddings[keep]

    query_vec = embed_text(query)
    scores = embeddings @ query_vec
    top_indices = np.argsort(-scores)[:top_k]

    return [{**passages[i], "score": float(scores[i])} for i in top_indices]


def explain(category: str, modality: str = "image") -> str:
    """Replacement for the old hardcoded _EXPLANATIONS dict. category is
    already an exact known id (a manipulation_type for image, or
    "real"/"fake" for audio) coming straight from the model's own
    classification head, so this is a direct lookup by id -- not a
    similarity search, which would only add a chance of matching the wrong
    passage for data we already have an exact key for."""
    for passage in load_knowledge_base():
        if passage["category"] == category and passage["modality"] == modality:
            return passage["text"]
    return "Manipulation artifacts detected."
