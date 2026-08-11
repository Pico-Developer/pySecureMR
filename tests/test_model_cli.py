import json
from pathlib import Path

import pytest

from pyspatialml import model_cli

REPO_ROOT = Path(__file__).resolve().parents[1]
FACE_MODEL = REPO_ROOT / "tests" / "data" / "face_mediapipe_package" / "model" / "face_detector.tflite"


def test_model_info_real_face_model(capsys):
    assert model_cli.model_info(FACE_MODEL) == 0

    captured = capsys.readouterr()
    assert f"Model: {FACE_MODEL}" in captured.out
    assert "Signature: <placeholder signature>" in captured.out
    assert "Inputs:" in captured.out
    assert "image: shape=(1, 256, 256, 3) dtype=float32 index=0" in captured.out
    assert "Outputs:" in captured.out
    assert "box_coords_1: shape=(1, 512, 16) dtype=float32" in captured.out
    assert "box_scores_2: shape=(1, 384, 1) dtype=float32" in captured.out


def test_model_info_json_real_face_model(capsys):
    assert model_cli.model_info(FACE_MODEL, as_json=True) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["signature_key"] == "<placeholder signature>"
    assert payload["inputs"][0]["name"] == "image"
    assert payload["inputs"][0]["shape"] == [1, 256, 256, 3]
    assert {item["name"] for item in payload["outputs"]} == {
        "box_coords_1",
        "box_coords_2",
        "box_scores_1",
        "box_scores_2",
    }


def test_model_info_rejects_missing_model(tmp_path):
    with pytest.raises(model_cli.ModelCliError, match="Model file not found"):
        model_cli.model_info(tmp_path / "missing.tflite")
