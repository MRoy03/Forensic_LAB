def test_feedback_simple(client):
    response = client.post(
        "/feedback",
        json={
            "feedback_type": "simple",
            "predicted_label": "fake",
            "predicted_manipulation_type": "splicing",
            "is_correct": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert len(body["feedback_id"]) > 0


def test_feedback_advanced(client):
    response = client.post(
        "/feedback",
        json={
            "feedback_type": "advanced",
            "predicted_label": "fake",
            "predicted_manipulation_type": "splicing",
            "correct_label": "real",
            "correct_manipulation_type": "none",
            "comments": "This was actually an authentic photo.",
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
