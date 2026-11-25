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

    second_opinions = body.get("second_opinions")
    assert second_opinions and len(second_opinions) == 2  # detectra-v1 + Organika/sdxl-detector
    model_names = {o["model_name"] for o in second_opinions}
    assert model_names == {"TensorFold/detectra-v1", "Organika/sdxl-detector"}
    for opinion in second_opinions:
        assert opinion["label"] in ("ai_generated", "real")
        assert 0.0 <= opinion["confidence"] <= 1.0


def test_predict_image_rejects_garbage(client):
    response = client.post(
        "/predict/image",
        files={"file": ("not_an_image.txt", b"this is not an image", "text/plain")},
    )
    assert response.status_code == 400
