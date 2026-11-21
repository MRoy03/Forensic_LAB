def test_predict_audio(client, sample_audio_bytes):
    response = client.post(
        "/predict/audio",
        files={"file": ("sample.wav", sample_audio_bytes, "audio/wav")},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["label"] in ("real", "fake")
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["explanation"], str) and len(body["explanation"]) > 0


def test_predict_audio_rejects_garbage(client):
    response = client.post(
        "/predict/audio",
        files={"file": ("not_audio.txt", b"this is not audio", "text/plain")},
    )
    assert response.status_code == 400
