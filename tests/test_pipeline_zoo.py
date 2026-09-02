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


@pytest.mark.parametrize(
    "model_path",
    [
        "/tmp/model.tflite",
        "C:/tmp/model.tflite",
        r"C:\tmp\model.tflite",
        r"\\server\share\model.tflite",
    ],
)
def test_create_litert_model_spec_rejects_absolute_paths(model_path):
    with pytest.raises(ValueError, match="Package paths must be relative"):
        create_litert_model_spec(model_path, "model")


def test_write_pipeline_zoo_package_writes_manifest_and_assets(tmp_path):
    model_file = tmp_path / "source.tflite"
    model_file.write_bytes(b"model")
    package = PipelineZooPackageSpec(
        package_id="face",
        supported_modes=["xr", "spatial", "xr"],
        pipelines=[PipelinePackageEntry("detection", "pipeline/face_detection_pipeline.json")],
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
    assert "display_name" not in manifest
    assert "task" not in manifest
    assert "labels" not in manifest
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
    assert by_inline_model["model_type"] == by_inline_model["model"]["model_type"]

    by_gpu_model = configure_litert_inference_operator(
        {"type": "RUN_MODEL_INFERENCE", "inputs": [], "outputs": []},
        model={"bin_path": "model/gpu.tflite", "model_target": "gpu",
               "cpu_target_num_threads": 4},
    )
    assert by_gpu_model["model_target"] == "gpu"
    assert by_gpu_model["model_target"] == by_gpu_model["model"]["model_target"]
    assert by_gpu_model["cpu_target_num_threads"] == 4

    with pytest.raises(ValueError, match="requires inline model metadata"):
        configure_litert_inference_operator({"type": "RUN_MODEL_INFERENCE", "inputs": [], "outputs": []})

    with pytest.raises(ValueError, match="either model or model_path"):
        configure_litert_inference_operator(
            {"type": "RUN_MODEL_INFERENCE", "inputs": [], "outputs": []},
            model={"bin_path": "model/inline.tflite"},
            model_path="model/detector.tflite",
        )


def test_validate_pipeline_zoo_rejects_path_traversal():
    with pytest.raises(ValueError, match="Invalid package-relative path"):
        validate_pipeline_zoo_manifest(
            {"schema_version": "2", "id": "bad", "pipelines": [{"id": "p", "path": "../pipeline.json"}]}
        )


def test_validate_pipeline_zoo_rejects_absolute_paths():
    with pytest.raises(ValueError, match="Package paths must be relative"):
        validate_pipeline_zoo_manifest(
            {"schema_version": "2", "id": "bad",
             "pipelines": [{"id": "p", "path": "/tmp/pipeline.json"}]}
        )


def test_pipeline_zoo_rejects_duplicate_pipeline_ids():
    package = PipelineZooPackageSpec(
        package_id="bad",
        pipelines=[
            PipelinePackageEntry("same", "pipeline/one.json"),
            PipelinePackageEntry("same", "pipeline/two.json"),
        ],
    )
    with pytest.raises(ValueError, match="Duplicate pipeline id: same"):
        package.to_manifest_dict()

    with pytest.raises(ValueError, match="Duplicate pipeline id: same"):
        validate_pipeline_zoo_manifest(
            {
                "schema_version": "2",
                "id": "bad",
                "pipelines": [
                    {"id": "same", "path": "pipeline/one.json"},
                    {"id": "same", "path": "pipeline/two.json"},
                ],
            }
        )


def test_validate_pipeline_zoo_requires_schema_v2():
    with pytest.raises(ValueError, match="schema_version must be 2"):
        validate_pipeline_zoo_manifest({"schema_version": "1.0", "id": "bad", "pipelines": []})


def test_validate_pipeline_zoo_rejects_unknown_execution_mode():
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
