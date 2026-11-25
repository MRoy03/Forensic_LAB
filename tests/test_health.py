def test_health_basics(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True  # primary model is eager-loaded at startup
    assert body["runtime"] == "onnxruntime"


def test_health_reports_lazy_models_only_after_use(client, sample_image_bytes, sample_audio_bytes):
    """The optional/heavy models (ensemble, Organika, audio, RAG) must stay
    lazy -- /health should never force-load them just by being polled (that
    was a real bug: it used to call the same function that triggers
    loading). Triggers real predictions first so this test is self-
    contained regardless of what other tests already ran in this session."""
    client.post(
        "/predict/image",
        files={"file": ("sample.jpg", sample_image_bytes, "image/jpeg")},
    )
    client.post(
        "/predict/audio",
        files={"file": ("sample.wav", sample_audio_bytes, "audio/wav")},
    )

    body = client.get("/health").json()
    assert body["ensemble_model_loaded"] is True
    assert body["organika_model_loaded"] is True
    assert body["audio_model_loaded"] is True
    # RAG's embedder is never actually touched by explain()'s direct
    # dict lookup (see services/explainability/rag.py) -- correctly stays
    # unloaded even after real predictions.
    assert body["rag_model_loaded"] is False
