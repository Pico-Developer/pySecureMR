import json

import pytest

from securemr.pipeline_zoo import (
    PipelinePackageEntry,
    PipelineZooPackageSpec,
    configure_litert_inference_operator,
    create_litert_model_spec,
    load_pipeline_zoo_manifest,
    validate_pipeline_zoo_manifest,
    write_pipeline_zoo_package,
)


def test_create_litert_model_spec_defaults_to_pipeline_zoo_schema():
    model_spec = create_litert_model_spec(
        "model/face_detector.tflite",
        "face_detector",
        input_tensors=[{"name": "input", "shape": [1, 128, 128, 3]}],
        output_tensors=[{"name": "scores", "shape": [1, 896, 1]}],
    )

    assert model_spec["bin_path"] == "model/face_detector.tflite"
    assert model_spec["model_name"] == "face_detector"
    assert model_spec["model_type"] == "tflite"
    assert model_spec["model_target"] == "npu"
    assert model_spec["cpu_target_num_threads"] == 1
    assert "path_to_zoo" not in model_spec
    assert "specific_config" not in model_spec
    assert model_spec["input"][0]["name"] == "input"
    assert model_spec["output"][0]["name"] == "scores"


def test_write_pipeline_zoo_package_writes_manifest_and_assets(tmp_path):
    model_file = tmp_path / "source.tflite"
    model_file.write_bytes(b"model")
    package = PipelineZooPackageSpec(
        package_id="face",
        display_name="Face",
        task="face_detection",
        supported_modes=["xr", "spatial", "xr"],
        pipelines=[PipelinePackageEntry("detection", "pipeline/face_detection_pipeline.json")],
        labels=["face"],
        runtime={"detection_tensor": "detections"},
    )
    pipeline = {
        "tensors": {},
        "operators": [
            configure_litert_inference_operator(
                {"type": "RUN_MODEL_INFERENCE", "inputs": [], "outputs": []},
                model_path="model/face_detector.tflite",
                model_name="face_detector",
            )
        ],
        "inputs": [],
        "outputs": [],
    }

    manifest = write_pipeline_zoo_package(
        tmp_path / "pkg",
        package,
        pipelines={"detection": pipeline},
        assets={"model/face_detector.tflite": model_file},
    )

    assert manifest["schema_version"] == "2"
    assert manifest["runtime"]["supported_modes"] == ["xr", "spatial"]
    assert "model" not in manifest
    assert "models" not in manifest
    assert (tmp_path / "pkg" / "manifest.json").exists()
    assert (tmp_path / "pkg" / "pipeline" / "face_detection_pipeline.json").exists()
    assert not (tmp_path / "pkg" / "model" / "model.json").exists()
    assert (tmp_path / "pkg" / "model" / "face_detector.tflite").read_bytes() == b"model"

    loaded_manifest = load_pipeline_zoo_manifest(tmp_path / "pkg")
    assert loaded_manifest["runtime"]["detection_tensor"] == "detections"
    with open(tmp_path / "pkg" / "pipeline" / "face_detection_pipeline.json", encoding="utf-8") as file:
        loaded_pipeline = json.load(file)
    assert loaded_pipeline["operators"][0]["model_type"] == "tflite"
    assert loaded_pipeline["operators"][0]["model"]["bin_path"] == "model/face_detector.tflite"
    assert "model_file" not in loaded_pipeline["operators"][0]


def test_configure_litert_inference_operator_requires_inline_model():
    by_model_path = configure_litert_inference_operator(
        {"type": "RUN_MODEL_INFERENCE", "inputs": [], "outputs": []},
        model_path="model/detector.tflite",
        model_name="detector",
    )
    by_inline_model = configure_litert_inference_operator(
        {"type": "RUN_MODEL_INFERENCE", "inputs": [], "outputs": []},
        model={"bin_path": "model/inline.tflite", "model_name": "inline", "model_type": "tflite"},
    )

    assert by_model_path["model"]["bin_path"] == "model/detector.tflite"
    assert by_inline_model["model"]["bin_path"] == "model/inline.tflite"
    assert by_inline_model["model_type"] == "tflite"

    with pytest.raises(ValueError, match="requires inline model metadata"):
        configure_litert_inference_operator({"type": "RUN_MODEL_INFERENCE", "inputs": [], "outputs": []})

    with pytest.raises(ValueError, match="either model or model_path"):
        configure_litert_inference_operator(
            {"type": "RUN_MODEL_INFERENCE", "inputs": [], "outputs": []},
            model={"bin_path": "model/inline.tflite"},
            model_path="model/detector.tflite",
        )


def test_validate_pipeline_zoo_manifest_rejects_path_traversal():
    with pytest.raises(ValueError, match="Invalid package-relative path"):
        validate_pipeline_zoo_manifest(
            {"schema_version": "2", "id": "bad", "pipelines": [{"id": "p", "path": "../pipeline.json"}]}
        )


def test_validate_pipeline_zoo_manifest_requires_schema_v2():
    with pytest.raises(ValueError, match="schema_version must be 2"):
        validate_pipeline_zoo_manifest({"schema_version": "1.0", "id": "bad", "pipelines": []})


def test_validate_pipeline_zoo_manifest_rejects_unknown_execution_mode():
    with pytest.raises(ValueError, match="Unsupported execution mode"):
        validate_pipeline_zoo_manifest(
            {
                "schema_version": "2",
                "id": "bad",
                "pipelines": [{"id": "p", "path": "pipeline.json"}],
                "runtime": {"supported_modes": ["desktop"]},
            }
        )

    with pytest.raises(ValueError, match="supported_modes must be a list"):
        validate_pipeline_zoo_manifest(
            {
                "schema_version": "2",
                "id": "bad",
                "pipelines": [{"id": "p", "path": "pipeline.json"}],
                "runtime": {"supported_modes": "xr"},
            }
        )
