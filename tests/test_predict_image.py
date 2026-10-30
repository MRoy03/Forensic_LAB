def test_predict_image(client, sample_image_bytes):
    response = client.post(
        "/predict/image",
        files={"file": ("sample.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["label"] in ("real", "fake")
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["manipulation_type"] in (
        "copy_move", "splicing", "inpainting", "gan", "compression", "beautify", "none",
    )
    assert isinstance(body["gradcam"], str) and len(body["gradcam"]) > 0
    assert isinstance(body["mask"], str)
    assert isinstance(body["explanation"], str) and len(body["explanation"]) > 0


def test_predict_image_rejects_garbage(client):
    response = client.post(
        "/predict/image",
        files={"file": ("not_an_image.txt", b"this is not an image", "text/plain")},
    )
    assert response.status_code == 400
