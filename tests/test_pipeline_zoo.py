import json

import pytest

from securemr.pipeline_zoo import (
    ModelPackageSpec,
    PipelinePackageEntry,
    PipelineZooPackageSpec,
    configure_litert_inference_operator,
    create_litert_model_json,
    load_pipeline_zoo_manifest,
    validate_pipeline_zoo_manifest,
    write_pipeline_zoo_package,
)


def test_create_litert_model_json_defaults_to_pipeline_zoo_schema():
    model_json = create_litert_model_json(
        "model/face_detector.tflite",
        "face_detector",
        input_tensors=[{"name": "input", "shape": [1, 128, 128, 3]}],
        output_tensors=[{"name": "scores", "shape": [1, 896, 1]}],
    )

    assert model_json["model_name"] == "face_detector"
    assert model_json["path_to_zoo"] == "model/face_detector.tflite"
    assert model_json["engine_type"] == "litert"
    assert model_json["model_target"] == "npu"
    assert model_json["specific_config"]["cpu_target_num_threads"] == 1
    assert model_json["input"][0]["name"] == "input"
    assert model_json["output"][0]["name"] == "scores"


def test_write_pipeline_zoo_package_writes_manifest_and_assets(tmp_path):
    model_file = tmp_path / "source.tflite"
    model_file.write_bytes(b"model")
    package = PipelineZooPackageSpec(
        package_id="face",
        display_name="Face",
        task="face_detection",
        supported_modes=["xr", "spatial", "xr"],
        pipelines=[PipelinePackageEntry("detection", "pipeline/face_detection_pipeline.json")],
        model=ModelPackageSpec(
            bin_path="model/face_detector.tflite",
            json_path="model/model.json",
            extra_json_path="model/anchors.json",
            model_id="default",
        ),
        models=[ModelPackageSpec("model/landmarks.tflite", "model/landmarks.json", model_id="landmarks")],
        labels=["face"],
        runtime={"detection_tensor": "detections"},
    )
    pipeline = {
        "tensors": {},
        "operators": [
            configure_litert_inference_operator(
                {"type": "RUN_MODEL_INFERENCE", "inputs": [], "outputs": []},
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
        model_json=create_litert_model_json("model/face_detector.tflite", "face_detector"),
        assets={"model/face_detector.tflite": model_file},
    )

    assert manifest["schema_version"] == "1.0"
    assert manifest["runtime"]["supported_modes"] == ["xr", "spatial"]
    assert manifest["model"]["id"] == "default"
    assert manifest["models"][0]["id"] == "landmarks"
    assert manifest["model"]["bin_path"] == "model/face_detector.tflite"
    assert manifest["model"]["extra_json_path"] == "model/anchors.json"
    assert (tmp_path / "pkg" / "manifest.json").exists()
    assert (tmp_path / "pkg" / "pipeline" / "face_detection_pipeline.json").exists()
    assert (tmp_path / "pkg" / "model" / "model.json").exists()
    assert (tmp_path / "pkg" / "model" / "face_detector.tflite").read_bytes() == b"model"

    loaded_manifest = load_pipeline_zoo_manifest(tmp_path / "pkg")
    assert loaded_manifest["runtime"]["detection_tensor"] == "detections"
    with open(tmp_path / "pkg" / "pipeline" / "face_detection_pipeline.json", encoding="utf-8") as file:
        loaded_pipeline = json.load(file)
    assert loaded_pipeline["operators"][0]["model_type"] == "litert"
    assert "model_file" not in loaded_pipeline["operators"][0]


def test_configure_litert_inference_operator_supports_model_selectors():
    by_model_id = configure_litert_inference_operator(
        {"type": "RUN_MODEL_INFERENCE", "inputs": [], "outputs": []},
        model_id="landmarks",
    )
    by_model_field = configure_litert_inference_operator(
        {"type": "RUN_MODEL_INFERENCE", "inputs": [], "outputs": []},
        model="detector",
    )
    by_inline_model = configure_litert_inference_operator(
        {"type": "RUN_MODEL_INFERENCE", "inputs": [], "outputs": []},
        model={"bin_path": "model/inline.tflite", "model_name": "inline"},
    )

    assert by_model_id["model_id"] == "landmarks"
    assert by_model_field["model"] == "detector"
    assert by_inline_model["model"]["bin_path"] == "model/inline.tflite"
    assert by_inline_model["model_type"] == "litert"

    with pytest.raises(ValueError, match="either model or model_id"):
        configure_litert_inference_operator(
            {"type": "RUN_MODEL_INFERENCE", "inputs": [], "outputs": []},
            model="detector",
            model_id="landmarks",
        )


def test_validate_pipeline_zoo_manifest_rejects_path_traversal():
    with pytest.raises(ValueError, match="Invalid package-relative path"):
        validate_pipeline_zoo_manifest(
            {"id": "bad", "pipelines": [{"id": "p", "path": "../pipeline.json"}], "model": {}}
        )


def test_validate_pipeline_zoo_manifest_rejects_unknown_execution_mode():
    with pytest.raises(ValueError, match="Unsupported execution mode"):
        validate_pipeline_zoo_manifest(
            {
                "id": "bad",
                "pipelines": [{"id": "p", "path": "pipeline.json"}],
                "model": {"bin_path": "model/model.tflite", "json_path": "model/model.json"},
                "runtime": {"supported_modes": ["desktop"]},
            }
        )

    with pytest.raises(ValueError, match="supported_modes must be a list"):
        validate_pipeline_zoo_manifest(
            {
                "id": "bad",
                "pipelines": [{"id": "p", "path": "pipeline.json"}],
                "model": {"bin_path": "model/model.tflite", "json_path": "model/model.json"},
                "runtime": {"supported_modes": "xr"},
            }
        )
