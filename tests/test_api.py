"""Unit and integration tests for the NeuroQA FastAPI inference server."""

from __future__ import annotations

import json
import numpy as np
import pytest
from fastapi.testclient import TestClient

from neuroqa.api.app import app


@pytest.fixture
def client() -> TestClient:
    """Provide a TestClient with startup and shutdown lifespan execution."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient) -> None:
    """Verify health endpoint returns status ok, model_loaded true, and version."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["version"] == "1.0.0"


def test_metrics_endpoint(client: TestClient) -> None:
    """Verify metrics tracking endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "average_latency_ms" in data


def test_predict_success(client: TestClient) -> None:
    """Verify successful prediction on valid (19, 512) EEG segment."""
    eeg_segment = np.random.randn(19, 512).astype(float).tolist()
    response = client.post("/predict", json={"eeg_segment": eeg_segment})
    assert response.status_code == 200
    data = response.json()

    assert data["prediction"] in {"clean", "artifact"}
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["top_time_windows"]) == 3
    assert data["processing_time_ms"] >= 0.0
    assert data["model_version"] == "1.0.0"


def test_predict_invalid_channel_count(client: TestClient) -> None:
    """Verify validation error when segment does not have 19 channels."""
    bad_segment = np.random.randn(18, 512).astype(float).tolist()
    response = client.post("/predict", json={"eeg_segment": bad_segment})
    assert response.status_code == 422


def test_predict_invalid_sample_count(client: TestClient) -> None:
    """Verify validation error when any channel does not have 512 samples."""
    bad_segment = np.random.randn(19, 500).astype(float).tolist()
    response = client.post("/predict", json={"eeg_segment": bad_segment})
    assert response.status_code == 422


def test_predict_nan_inf_detection(client: TestClient) -> None:
    """Verify HTTP 422 with message when NaN or Inf are in the signal."""
    base_json = json.dumps({"eeg_segment": [[0.0] * 512] * 19})

    # Test NaN
    nan_json = base_json.replace("0.0", "NaN", 1)
    response_nan = client.post(
        "/predict",
        content=nan_json,
        headers={"Content-Type": "application/json"},
    )
    assert response_nan.status_code == 422
    assert response_nan.json()["detail"] == "NaN or Inf detected in input signal"

    # Test Inf
    inf_json = base_json.replace("0.0", "Infinity", 1)
    response_inf = client.post(
        "/predict",
        content=inf_json,
        headers={"Content-Type": "application/json"},
    )
    assert response_inf.status_code == 422
    assert response_inf.json()["detail"] == "NaN or Inf detected in input signal"
