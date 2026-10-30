def test_predict_video(client, sample_video_bytes):
    response = client.post(
        "/predict/video",
        files={"file": ("sample.mp4", sample_video_bytes, "video/mp4")},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["label"] in ("real", "fake")
    assert body["total_frames_sampled"] > 0
    assert len(body["frame_results"]) == body["total_frames_sampled"]
    assert 0.0 <= body["fake_frame_ratio"] <= 1.0
