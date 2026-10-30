import pytest


@pytest.mark.parametrize(
    "manipulation_type", ["copy_move", "splicing", "compression", "blur", "beautify", "upscale"]
)
def test_playground_generate(client, sample_image_bytes, manipulation_type):
    response = client.post(
        "/playground/generate",
        data={"manipulation_type": manipulation_type},
        files={"file": ("sample.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "fake"
    assert body["manipulation_type"] == manipulation_type
    assert isinstance(body["image"], str) and len(body["image"]) > 0


def test_playground_generate_none(client, sample_image_bytes):
    response = client.post(
        "/playground/generate",
        data={"manipulation_type": "none"},
        files={"file": ("sample.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "real"
    assert body["mask"] is None


def test_playground_generate_unknown_type(client, sample_image_bytes):
    response = client.post(
        "/playground/generate",
        data={"manipulation_type": "not_a_real_type"},
        files={"file": ("sample.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 400
