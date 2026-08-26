import json
import importlib.util
import zipfile
from pathlib import Path

import numpy as np
import pytest

from pyspatialml import device_runner_base, package_cli, run_cli

REPO_ROOT = Path(__file__).resolve().parents[1]
FACE_FIXTURE = REPO_ROOT / "tests" / "data" / "face_mediapipe_package"
XR_RUNNER_SCRIPT = REPO_ROOT / "pyspatialml" / "xr_pipeline_runner" / "scripts" / "run_xr_pipeline.py"


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _package_pipeline(tmp_path, pipeline, *, package_id="demo", pipeline_id="main"):
    package = tmp_path / f"{package_id}-package"
    package_cli.create_package(
        package_id=package_id,
        pipelines=[f"{pipeline_id}={pipeline}"],
        output=package,
    )
    return package


def _package_output(output_dir, pipeline_id="main"):
    return output_dir / pipeline_id


def _write_device_package_manifest(package, *, pipeline_id="main", supported_modes=None):
    package.mkdir()
    pipeline_path = package / "pipeline" / f"{pipeline_id}.json"
    _write_json(pipeline_path, _simple_pipeline())
    manifest = {"id": "demo", "pipelines": [{"id": pipeline_id, "path": f"pipeline/{pipeline_id}.json"}]}
    if supported_modes is not None:
        manifest["runtime"] = {"supported_modes": supported_modes}
    _write_json(package / "manifest.json", manifest)


def _simple_pipeline():
    return {
        "tensors": {
            "x": {
                "dimensions": [2, 2],
                "channels": 1,
                "data_type": 6,
                "is_placeholder": True,
                "usage": 6,
            },
            "y": {
                "dimensions": [2, 2],
                "channels": 1,
                "data_type": 6,
                "is_placeholder": True,
                "usage": 6,
            },
        },
        "operators": [
            {
                "type": "XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO",
                "inputs": ["x"],
                "outputs": ["y"],
                "expression": "{0} * 2.0",
            }
        ],
        "inputs": ["x"],
        "outputs": ["y"],
    }


def _second_pipeline():
    return {
        "tensors": {
            "y": {
                "dimensions": [2, 2],
                "channels": 1,
                "data_type": 6,
                "is_placeholder": True,
                "usage": 6,
            },
            "z": {
                "dimensions": [2, 2],
                "channels": 1,
                "data_type": 6,
                "is_placeholder": True,
                "usage": 6,
            },
        },
        "operators": [
            {
                "type": "XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO",
                "inputs": ["y"],
                "outputs": ["z"],
                "expression": "{0} + 3.0",
            }
        ],
        "inputs": ["y"],
        "outputs": ["z"],
    }


def _load_xr_runner_script():
    spec = importlib.util.spec_from_file_location("run_xr_pipeline", XR_RUNNER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_host_pipeline_json_prints_summary_and_saves_outputs(capsys, tmp_path):
    pipeline = tmp_path / "pipeline.json"
    sample = tmp_path / "x.npy"
    output_dir = tmp_path / "outputs"
    _write_json(pipeline, _simple_pipeline())
    np.save(sample, np.ones((2, 2), dtype=np.float32))

    assert run_cli.run_host(
        _package_pipeline(tmp_path, pipeline),
        inputs=[f"x={sample}"],
        output_dir=output_dir,
    ) == 0

    captured = capsys.readouterr()
    output = np.load(_package_output(output_dir) / "y.npy")
    np.testing.assert_allclose(output, np.ones((2, 2), dtype=np.float32) * 2.0)
    assert "Outputs: 1" in captured.out
    assert "y: shape=(2, 2) dtype=float32" in captured.out
    assert "mean=2" in captured.out
    assert "preview=[2, 2, 2, 2]" in captured.out
    assert "Host Run Summary" in captured.out
    assert "main:" in captured.out
    assert "Total host run time:" in captured.out


def test_run_host_rejects_non_positive_duration(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    _write_json(pipeline, _simple_pipeline())

    with pytest.raises(run_cli.RunCliError, match="--duration"):
        run_cli.run_host(_package_pipeline(tmp_path, pipeline), duration=0)


def test_run_host_summary_marks_all_zero_and_truncated_preview(capsys, tmp_path):
    pipeline = tmp_path / "pipeline.json"
    sample = tmp_path / "x.npy"
    _write_json(
        pipeline,
        {
            "tensors": {
                "x": {
                    "dimensions": [1, 10],
                    "channels": 1,
                    "data_type": 6,
                    "is_placeholder": True,
                    "usage": 6,
                },
                "y": {
                    "dimensions": [1, 10],
                    "channels": 1,
                    "data_type": 6,
                    "is_placeholder": True,
                    "usage": 6,
                },
            },
            "operators": [
                {
                    "type": "XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO",
                    "inputs": ["x"],
                    "outputs": ["y"],
                }
            ],
            "inputs": ["x"],
            "outputs": ["y"],
        },
    )
    np.save(sample, np.zeros((1, 10), dtype=np.float32))

    assert run_cli.run_host(_package_pipeline(tmp_path, pipeline), inputs=[f"x={sample}"]) == 0

    captured = capsys.readouterr()
    assert "all_zero=true" in captured.out
    assert "preview=[0, 0, 0, 0, 0, 0, 0, 0, ...]" in captured.out


def test_run_host_accepts_schema_v2_generic_elementwise(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    a = tmp_path / "a.npy"
    b = tmp_path / "b.npy"
    output_dir = tmp_path / "outputs"
    _write_json(
        pipeline,
        {
            "tensors": {
                "a": {"dimensions": [2, 2], "channels": 1, "data_type": 6, "is_placeholder": True, "usage": 6},
                "b": {"dimensions": [2, 2], "channels": 1, "data_type": 6, "is_placeholder": True, "usage": 6},
                "y": {"dimensions": [2, 2], "channels": 1, "data_type": 6, "is_placeholder": True, "usage": 6},
            },
            "operators": [{"type": "elementwise", "op": "multiply", "inputs": ["a", "b"], "outputs": ["y"]}],
            "inputs": ["a", "b"],
            "outputs": ["y"],
        },
    )
    np.save(a, np.ones((2, 2), dtype=np.float32) * 2)
    np.save(b, np.ones((2, 2), dtype=np.float32) * 3)

    assert run_cli.run_host(_package_pipeline(tmp_path, pipeline), inputs=[f"a={a}", f"b={b}"], output_dir=output_dir) == 0

    np.testing.assert_allclose(np.load(_package_output(output_dir) / "y.npy"), np.ones((2, 2), dtype=np.float32) * 6)


def test_run_host_preserves_supplied_rectified_vst_outputs(tmp_path):
    pipeline = tmp_path / "vst.json"
    left = tmp_path / "left.npy"
    right = tmp_path / "right.npy"
    output_dir = tmp_path / "outputs"
    _write_json(
        pipeline,
        {
            "tensors": {
                "vst_right_image": {
                    "dimensions": [2, 2],
                    "channels": 3,
                    "data_type": 1,
                    "is_placeholder": True,
                    "usage": 6,
                },
                "vst_left_image": {
                    "dimensions": [2, 2],
                    "channels": 3,
                    "data_type": 1,
                    "is_placeholder": True,
                    "usage": 6,
                },
                "vst_timestamp": {
                    "tensor_type": "timestamp",
                    "is_placeholder": True,
                },
                "vst_camera_matrix": {
                    "dimensions": [3, 3],
                    "channels": 1,
                    "data_type": 6,
                    "is_placeholder": True,
                    "usage": 6,
                },
            },
            "operators": [
                {
                    "type": "XR_SECURE_MR_OPERATOR_TYPE_RECTIFIED_VST_ACCESS_PICO",
                    "inputs": [],
                    "outputs": [
                        "vst_right_image",
                        "vst_left_image",
                        "vst_timestamp",
                        "vst_camera_matrix",
                    ],
                }
            ],
            "inputs": [],
            "outputs": [
                "vst_right_image",
                "vst_left_image",
                "vst_timestamp",
                "vst_camera_matrix",
            ],
        },
    )
    left_value = np.ones((2, 2, 3), dtype=np.uint8) * 7
    right_value = np.ones((2, 2, 3), dtype=np.uint8) * 9
    np.save(left, left_value)
    np.save(right, right_value)

    assert run_cli.run_host(
        _package_pipeline(tmp_path, pipeline),
        inputs=[f"vst_left_image={left}", f"vst_right_image={right}"],
        output_dir=output_dir,
    ) == 0

    np.testing.assert_array_equal(np.load(_package_output(output_dir) / "vst_left_image.npy"), left_value)
    np.testing.assert_array_equal(np.load(_package_output(output_dir) / "vst_right_image.npy"), right_value)
    assert (_package_output(output_dir) / "vst_timestamp.npy").is_file()
    assert (_package_output(output_dir) / "vst_camera_matrix.npy").is_file()


def test_run_host_resizes_supplied_image_input_to_tensor_shape(tmp_path):
    pipeline = tmp_path / "vst.json"
    left = tmp_path / "left.npy"
    output_dir = tmp_path / "outputs"
    _write_json(
        pipeline,
        {
            "tensors": {
                "vst_left_image": {
                    "dimensions": [4, 3],
                    "channels": 3,
                    "data_type": 1,
                    "is_placeholder": True,
                    "usage": 6,
                }
            },
            "operators": [],
            "inputs": ["vst_left_image"],
            "outputs": ["vst_left_image"],
        },
    )
    np.save(left, np.ones((30, 40, 3), dtype=np.uint8) * 11)

    assert run_cli.run_host(_package_pipeline(tmp_path, pipeline), inputs=[f"vst_left_image={left}"], output_dir=output_dir) == 0

    output = np.load(_package_output(output_dir) / "vst_left_image.npy")
    assert output.shape == (3, 4, 3)
    assert output.dtype == np.uint8


def test_run_host_bare_image_input_feeds_rectified_vst(tmp_path):
    cv2 = pytest.importorskip("cv2")
    pipeline = tmp_path / "vst.json"
    image = tmp_path / "face.jpg"
    output_dir = tmp_path / "outputs"
    _write_json(
        pipeline,
        {
            "tensors": {
                "vst_right_image": {"dimensions": [3, 2], "channels": 3, "data_type": 1, "is_placeholder": True, "usage": 6},
                "vst_left_image": {"dimensions": [3, 2], "channels": 3, "data_type": 1, "is_placeholder": True, "usage": 6},
                "vst_timestamp": {"dimensions": [1, 1], "channels": 4, "data_type": 5, "is_placeholder": True, "usage": 6},
                "vst_camera_matrix": {"dimensions": [3, 3], "channels": 1, "data_type": 6, "is_placeholder": True, "usage": 6},
            },
            "operators": [
                {
                    "type": "XR_SECURE_MR_OPERATOR_TYPE_RECTIFIED_VST_ACCESS_PICO",
                    "inputs": [],
                    "outputs": ["vst_right_image", "vst_left_image", "vst_timestamp", "vst_camera_matrix"],
                }
            ],
            "inputs": [],
            "outputs": ["vst_left_image", "vst_right_image"],
        },
    )
    cv2.imwrite(str(image), np.ones((4, 5, 3), dtype=np.uint8) * 23)

    assert run_cli.run_host(_package_pipeline(tmp_path, pipeline), inputs=[str(image)], output_dir=output_dir) == 0

    left = np.load(_package_output(output_dir) / "vst_left_image.npy")
    right = np.load(_package_output(output_dir) / "vst_right_image.npy")
    assert left.shape == (2, 3, 3)
    np.testing.assert_array_equal(left, right)
    assert np.max(left) > 0


def test_run_host_bare_image_input_binds_vst_image_tensors_without_operator(tmp_path):
    cv2 = pytest.importorskip("cv2")
    image_path = tmp_path / "face.png"
    pipeline = tmp_path / "vst_tensors.json"
    output_dir = tmp_path / "outputs"
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[:, :, 1] = 128
    cv2.imwrite(str(image_path), image)
    _write_json(
        pipeline,
        {
            "tensors": {
                "vst_left_image": {
                    "dimensions": [3, 2],
                    "channels": 3,
                    "data_type": 1,
                    "is_placeholder": True,
                    "usage": 6,
                },
                "vst_right_image": {
                    "dimensions": [3, 2],
                    "channels": 3,
                    "data_type": 1,
                    "is_placeholder": True,
                    "usage": 6,
                },
            },
            "operators": [],
            "inputs": ["vst_left_image", "vst_right_image"],
            "outputs": ["vst_left_image", "vst_right_image"],
        },
    )

    assert run_cli.run_host(_package_pipeline(tmp_path, pipeline), inputs=[str(image_path)], output_dir=output_dir) == 0

    np.testing.assert_array_equal(np.load(_package_output(output_dir) / "vst_left_image.npy"), image)
    np.testing.assert_array_equal(np.load(_package_output(output_dir) / "vst_right_image.npy"), image)


def test_run_host_bare_input_feeds_declared_input_tensor(tmp_path):
    pipeline = tmp_path / "simple.json"
    sample = tmp_path / "x.npy"
    output_dir = tmp_path / "outputs"
    _write_json(
        pipeline,
        {
            "tensors": {
                "x": {"dimensions": [2, 2], "channels": 1, "data_type": 6, "is_placeholder": True, "usage": 6},
                "y": {"dimensions": [2, 2], "channels": 1, "data_type": 6, "is_placeholder": True, "usage": 6},
            },
            "operators": [
                {
                    "type": "XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO",
                    "inputs": ["x"],
                    "outputs": ["y"],
                    "expression": "{0} * 2.0",
                }
            ],
            "inputs": ["x"],
            "outputs": ["y"],
        },
    )
    np.save(sample, np.ones((2, 2), dtype=np.float32))

    assert run_cli.run_host(_package_pipeline(tmp_path, pipeline), inputs=[str(sample)], output_dir=output_dir) == 0

    np.testing.assert_allclose(np.load(_package_output(output_dir) / "y.npy"), np.ones((2, 2), dtype=np.float32) * 2.0)


def test_run_host_model_operator_uses_litert_runner(monkeypatch, capsys, tmp_path):
    pipeline = tmp_path / "pipeline.json"
    model = tmp_path / "model" / "demo.tflite"
    sample = tmp_path / "x.npy"
    output_dir = tmp_path / "outputs"
    model.parent.mkdir()
    model.write_bytes(b"model")
    _write_json(
        pipeline,
        {
            "tensors": {
                "x": {
                    "dimensions": [2, 2],
                    "channels": 1,
                    "data_type": 6,
                    "is_placeholder": True,
                    "usage": 6,
                },
                "scores": {
                    "dimensions": [2, 2],
                    "channels": 1,
                    "data_type": 6,
                    "is_placeholder": True,
                    "usage": 6,
                },
            },
            "operators": [
                {
                    "type": "XR_SECURE_MR_OPERATOR_TYPE_RUN_MODEL_INFERENCE_PICO",
                    "inputs": [{"name": "input", "tensor": "x"}],
                    "outputs": [{"name": "scores", "tensor": "scores"}],
                    "model": {
                        "bin_path": "model/demo.tflite",
                        "model_name": "demo",
                        "model_type": "tflite",
                        "model_target": "npu",
                    },
                }
            ],
            "inputs": ["x"],
            "outputs": ["scores"],
        },
    )
    np.save(sample, np.ones((2, 2), dtype=np.float32))
    calls = []

    monkeypatch.setattr(run_cli, "resolve_litert_cli", lambda ensure=True: object())

    def _run_litert_runtime_model(**kwargs):
        calls.append(kwargs)
        return {"scores": kwargs["inputs"]["input"] * 3.0}

    monkeypatch.setattr(run_cli, "_run_litert_runtime_model", _run_litert_runtime_model)

    package = _package_pipeline(tmp_path, pipeline)
    assert run_cli.run_host(package, inputs=[f"x={sample}"], output_dir=output_dir) == 0

    captured = capsys.readouterr()
    np.testing.assert_allclose(np.load(_package_output(output_dir) / "scores.npy"), np.ones((2, 2), dtype=np.float32) * 3.0)
    assert "RUN_MODEL_INFERENCE demo:" in captured.out
    assert "target=cpu" in captured.out
    assert calls
    assert calls[0]["model_path"] == (package / "model" / "demo.tflite").resolve()
    assert list(calls[0]["inputs"]) == ["input"]
    assert calls[0]["output_names"] == ["scores"]


def test_run_host_normalizes_model_target_to_cpu():
    spec = {
        "operators": [
            {
                "type": "XR_SECURE_MR_OPERATOR_TYPE_RUN_MODEL_INFERENCE_PICO",
                "model_target": "npu",
                "model": {"bin_path": "demo.tflite", "model_target": "npu"},
            }
        ]
    }

    normalized = run_cli._normalize_run_pipeline_spec(spec, manifest_model=None)

    assert normalized["operators"][0]["model_target"] == "cpu"
    assert normalized["operators"][0]["model"]["model_target"] == "cpu"


def test_run_host_model_operator_reports_litert_failure(monkeypatch, tmp_path):
    pipeline = tmp_path / "pipeline.json"
    model = tmp_path / "demo.tflite"
    sample = tmp_path / "x.npy"
    model.write_bytes(b"model")
    _write_json(
        pipeline,
        {
            "tensors": {
                "x": {"dimensions": [1, 1], "channels": 1, "data_type": 6, "is_placeholder": True, "usage": 6},
                "scores": {"dimensions": [1, 1], "channels": 1, "data_type": 6, "is_placeholder": True, "usage": 6},
            },
            "operators": [
                {
                    "type": "XR_SECURE_MR_OPERATOR_TYPE_RUN_MODEL_INFERENCE_PICO",
                    "inputs": [{"name": "input", "tensor": "x"}],
                    "outputs": [{"name": "scores", "tensor": "scores"}],
                    "model": {"bin_path": "demo.tflite", "model_name": "demo", "model_type": "tflite"},
                }
            ],
            "inputs": ["x"],
            "outputs": ["scores"],
        },
    )
    np.save(sample, np.ones((1, 1), dtype=np.float32))

    monkeypatch.setattr(run_cli, "resolve_litert_cli", lambda ensure=True: object())

    def _fail_model(**_kwargs):
        raise RuntimeError("bad model")

    monkeypatch.setattr(run_cli, "_run_litert_runtime_model", _fail_model)

    with pytest.raises(run_cli.RunCliError, match="LiteRT model run failed"):
        run_cli.run_host(_package_pipeline(tmp_path, pipeline), inputs=[f"x={sample}"])


def test_run_host_face_fixture_runs_real_litert_model(tmp_path):
    output_dir = tmp_path / "outputs"

    assert run_cli.run_host(
        FACE_FIXTURE,
        pipeline_ids=["detection"],
        inputs=[
            f"vst_left_image={FACE_FIXTURE / 'inputs' / 'face.jpg'}",
            f"vst_right_image={FACE_FIXTURE / 'inputs' / 'face.jpg'}",
        ],
        output_dir=output_dir,
    ) == 0

    post_det = np.load(output_dir / "detection" / "post_det.npy")
    decoded = json.loads((output_dir / "detection" / "post_det.json").read_text(encoding="utf-8"))
    assert post_det.shape == (1, 21)
    assert not np.allclose(post_det, 0.0), "fixture image should produce a non-zero face detection"
    assert post_det[0, 4] > 0.25
    assert decoded["score"] > 0.25
    bbox_width = decoded["bbox"]["x2"] - decoded["bbox"]["x1"]
    bbox_height = decoded["bbox"]["y2"] - decoded["bbox"]["y1"]
    assert bbox_width > 1.0
    assert bbox_height > 1.0
    assert bbox_width * bbox_height > 100.0
    assert any(keypoint["x"] != 0.0 or keypoint["y"] != 0.0 for keypoint in decoded["keypoints"])
    assert all(keypoint["score"] > 0.25 for keypoint in decoded["keypoints"])


def test_run_host_dumps_selected_tensor(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    sample = tmp_path / "x.npy"
    output_dir = tmp_path / "outputs"
    _write_json(pipeline, _simple_pipeline())
    np.save(sample, np.ones((2, 2), dtype=np.float32))

    assert run_cli.run_host(
        _package_pipeline(tmp_path, pipeline),
        inputs=[f"x={sample}"],
        output_dir=output_dir,
        dumps=["x"],
    ) == 0

    np.testing.assert_allclose(np.load(_package_output(output_dir) / "dumped" / "x.npy"), np.ones((2, 2), dtype=np.float32))
    assert not (_package_output(output_dir) / "dumped" / "y.npy").exists()


def test_run_host_dump_all_saves_inputs_intermediates_and_outputs(capsys, tmp_path):
    pipeline = tmp_path / "pipeline.json"
    sample = tmp_path / "x.npy"
    output_dir = tmp_path / "outputs"
    _write_json(pipeline, _simple_pipeline())
    np.save(sample, np.ones((2, 2), dtype=np.float32))

    assert run_cli.run_host(
        _package_pipeline(tmp_path, pipeline),
        inputs=[f"x={sample}"],
        output_dir=output_dir,
        dumps=["all"],
    ) == 0

    np.testing.assert_allclose(np.load(_package_output(output_dir) / "all_tensors" / "x.npy"), np.ones((2, 2), dtype=np.float32))
    np.testing.assert_allclose(np.load(_package_output(output_dir) / "all_tensors" / "y.npy"), np.ones((2, 2), dtype=np.float32) * 2.0)
    captured = capsys.readouterr()
    assert "Dumped tensors:" not in captured.out


def test_run_host_errors_for_missing_dump_tensor(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    sample = tmp_path / "x.npy"
    _write_json(pipeline, _simple_pipeline())
    np.save(sample, np.ones((2, 2), dtype=np.float32))

    with pytest.raises(run_cli.RunCliError, match="Requested dump tensor not found"):
        run_cli.run_host(_package_pipeline(tmp_path, pipeline), inputs=[f"x={sample}"], dumps=["missing"])


def test_run_host_writes_display_summary_for_pose_and_gltf(capsys, tmp_path):
    pipeline = tmp_path / "display.json"
    sample = tmp_path / "pose.npy"
    output_dir = tmp_path / "outputs"
    asset = tmp_path / "gltf" / "frame.gltf"
    asset.parent.mkdir()
    asset.write_text("{}", encoding="utf-8")
    _write_json(
        pipeline,
        {
            "tensors": {
                "pose_in": {
                    "dimensions": [4, 4],
                    "channels": 1,
                    "data_type": 6,
                    "is_placeholder": True,
                    "usage": 6,
                },
                "frame_pose": {
                    "dimensions": [4, 4],
                    "channels": 1,
                    "data_type": 6,
                    "is_placeholder": True,
                    "usage": 6,
                },
                "frame_gltf": {
                    "tensor_type": "gltf",
                    "asset": "gltf/frame.gltf",
                    "is_placeholder": True,
                },
            },
            "operators": [
                {
                    "type": "XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO",
                    "inputs": ["pose_in"],
                    "outputs": ["frame_pose"],
                }
            ],
            "inputs": ["pose_in"],
            "outputs": ["frame_pose", "frame_gltf"],
        },
    )
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    np.save(sample, pose)

    assert run_cli.run_host(_package_pipeline(tmp_path, pipeline), inputs=[f"pose_in={sample}"], output_dir=output_dir) == 0

    captured = capsys.readouterr()
    summary = json.loads((_package_output(output_dir) / "display_summary.json").read_text(encoding="utf-8"))
    assert "Outputs: 2" in captured.out
    assert "Tensor outputs: 1" in captured.out
    assert "Host note: spatial display outputs are not rendered on host." in captured.out
    assert "frame_pose: translation=[1.0, 2.0, 3.0]" in captured.out
    assert "frame_gltf: asset reference gltf/frame.gltf exists=yes" in captured.out
    assert summary["host_note"] == "Host mode does not render spatial glTF output."
    assert summary["outputs"][0]["name"] == "frame_pose"
    assert summary["outputs"][0]["translation"] == [1.0, 2.0, 3.0]
    assert summary["outputs"][1]["name"] == "frame_gltf"
    assert summary["outputs"][1]["asset"] == "gltf/frame.gltf"
    np.testing.assert_allclose(np.load(_package_output(output_dir) / "frame_pose.npy"), pose)


def test_run_host_decodes_post_det_output(capsys, tmp_path):
    pipeline = tmp_path / "detection.json"
    sample = tmp_path / "post_det_input.npy"
    output_dir = tmp_path / "outputs"
    assert run_cli._POST_DET_TENSOR_SIZE == 21
    values = np.array(
        [
            10.0,
            20.0,
            110.0,
            220.0,
            0.9,
            0.0,
            11.0,
            21.0,
            0.9,
            12.0,
            22.0,
            0.8,
            13.0,
            23.0,
            0.7,
            14.0,
            24.0,
            0.6,
            15.0,
            25.0,
            0.5,
        ],
        dtype=np.float32,
    ).reshape(1, 21)
    _write_json(
        pipeline,
        {
            "tensors": {
                "post_det_input": {
                    "dimensions": [1, 21],
                    "channels": 1,
                    "data_type": 6,
                    "is_placeholder": True,
                    "usage": 6,
                },
                "post_det": {
                    "dimensions": [1, 21],
                    "channels": 1,
                    "data_type": 6,
                    "is_placeholder": True,
                    "usage": 6,
                },
            },
            "operators": [
                {
                    "type": "XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO",
                    "inputs": ["post_det_input"],
                    "outputs": ["post_det"],
                }
            ],
            "inputs": ["post_det_input"],
            "outputs": ["post_det"],
        },
    )
    np.save(sample, values)

    assert run_cli.run_host(
        _package_pipeline(tmp_path, pipeline),
        inputs=[f"post_det_input={sample}"],
        output_dir=output_dir,
    ) == 0

    captured = capsys.readouterr()
    decoded = json.loads((_package_output(output_dir) / "post_det.json").read_text(encoding="utf-8"))
    assert "post_det decoded:" in captured.out
    assert "x1: 10" in captured.out
    assert "score: 0.9" in captured.out
    assert decoded["bbox"] == {"x1": 10.0, "y1": 20.0, "x2": 110.0, "y2": 220.0}
    assert decoded["score"] == pytest.approx(0.9)
    assert decoded["keypoints"][4] == {"index": 4, "x": 15.0, "y": 25.0, "score": 0.5}
    np.testing.assert_allclose(np.load(_package_output(output_dir) / "post_det.npy"), values)


def test_run_host_package_uses_pipeline_id(capsys, tmp_path):
    source = tmp_path / "source"
    pipeline = source / "main.json"
    sample = tmp_path / "x.npy"
    _write_json(pipeline, _simple_pipeline())
    np.save(sample, np.ones((2, 2), dtype=np.float32))
    package = tmp_path / "pkg"
    package_cli.create_package(
        package_id="demo",
        pipelines=[f"main={pipeline}"],
        output=package,
    )

    assert run_cli.run_host(package, pipeline_ids=["main"], inputs=[f"x={sample}"]) == 0

    captured = capsys.readouterr()
    assert "pipeline/main.json" in captured.out
    assert "Outputs: 1" in captured.out


def test_run_host_package_runs_all_pipelines_by_default(capsys, tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    sample = tmp_path / "x.npy"
    output_dir = tmp_path / "outputs"
    _write_json(first, _simple_pipeline())
    _write_json(second, _second_pipeline())
    np.save(sample, np.ones((2, 2), dtype=np.float32))
    package = tmp_path / "pkg"
    package_cli.create_package(
        package_id="demo",
        pipelines=[f"first={first}", f"second={second}"],
        output=package,
    )

    assert run_cli.run_host(package, inputs=[f"x={sample}"], output_dir=output_dir) == 0

    captured = capsys.readouterr()
    assert "Pipelines: first, second" in captured.out
    assert "Pipeline: first" in captured.out
    assert "Pipeline: second" in captured.out
    np.testing.assert_allclose(np.load(output_dir / "first" / "y.npy"), np.ones((2, 2), dtype=np.float32) * 2.0)
    np.testing.assert_allclose(np.load(output_dir / "second" / "z.npy"), np.ones((2, 2), dtype=np.float32) * 5.0)


def test_run_host_package_runs_requested_pipeline_order(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    sample = tmp_path / "x.npy"
    output_dir = tmp_path / "outputs"
    _write_json(first, _simple_pipeline())
    _write_json(second, _second_pipeline())
    np.save(sample, np.ones((2, 2), dtype=np.float32))
    package = tmp_path / "pkg"
    package_cli.create_package(
        package_id="demo",
        pipelines=[f"first={first}", f"second={second}"],
        output=package,
    )

    assert run_cli.run_host(
        package,
        pipeline_ids=["first", "second"],
        inputs=[f"x={sample}"],
        output_dir=output_dir,
    ) == 0

    np.testing.assert_allclose(np.load(output_dir / "second" / "z.npy"), np.ones((2, 2), dtype=np.float32) * 5.0)


def test_find_package_root_ignores_macos_metadata(tmp_path):
    root = tmp_path / "extract"
    package = root / "face-mediapipe-pipeline"
    macos = root / "__MACOSX"
    package.mkdir(parents=True)
    macos.mkdir(parents=True)
    (package / "manifest.json").write_text('{"id":"demo","pipelines":[]}', encoding="utf-8")
    (macos / "._face-mediapipe-pipeline").write_text("metadata", encoding="utf-8")

    assert run_cli._find_package_root(root) == package


def test_run_host_rejects_zip_path_traversal(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as package_zip:
        package_zip.writestr("../evil.txt", "bad")
        package_zip.writestr("manifest.json", '{"schema_version":"2","id":"bad","pipelines":[]}')

    with pytest.raises(run_cli.RunCliError, match="Unsafe zip entry path"):
        run_cli.run_host(archive)
    assert not (tmp_path.parent / "evil.txt").exists()


def test_run_host_package_rejects_duplicate_pipeline_ids(tmp_path):
    first = tmp_path / "first.json"
    sample = tmp_path / "x.npy"
    _write_json(first, _simple_pipeline())
    np.save(sample, np.ones((2, 2), dtype=np.float32))
    package = tmp_path / "pkg"
    package_cli.create_package(
        package_id="demo",
        pipelines=[f"first={first}"],
        output=package,
    )

    with pytest.raises(run_cli.RunCliError, match="Duplicate pipeline id"):
        run_cli.run_host(package, pipeline_ids=["first", "first"], inputs=[f"x={sample}"])


def test_run_host_rejects_pipeline_id_for_raw_pipeline(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    _write_json(pipeline, _simple_pipeline())

    with pytest.raises(run_cli.RunCliError, match="Raw pipeline JSON is not a valid run target"):
        run_cli.run_host(pipeline, pipeline_ids=["main"])


def test_run_host_rejects_bad_input_format(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    _write_json(pipeline, _simple_pipeline())

    with pytest.raises(run_cli.RunCliError, match="Input file not found"):
        run_cli.run_host(_package_pipeline(tmp_path, pipeline), inputs=["bad-input"])


def test_run_device_invokes_xr_runner_script_when_mode_is_explicit(monkeypatch, tmp_path):
    package = tmp_path / "pkg"
    _write_device_package_manifest(package)
    image = tmp_path / "face.jpg"
    image.write_bytes(b"jpg")
    output_dir = tmp_path / "outputs"
    apk = tmp_path / "runner.apk"
    apk.write_bytes(b"apk")
    calls = []

    script = tmp_path / "run_xr_pipeline.py"
    script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    monkeypatch.setattr(run_cli, "_xr_runner_script", lambda: script)

    def _run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        class Result:
            returncode = 7
        return Result()

    monkeypatch.setattr(run_cli.subprocess, "run", _run)

    assert run_cli.run_device(
        package,
        mode="xr",
        inputs=[str(image), f"vst_right_image={image}"],
        pipeline_ids=["detection", "display"],
        output_dir=output_dir,
        dumps=["all"],
        duration=3.5,
        loop=True,
        keep_running=True,
        use_vst=True,
        backend="cpu",
        interval_ms=25,
        apk=apk,
        device="serial-1",
    ) == 7

    cmd, kwargs = calls[0]
    assert "PYTHONPATH" in kwargs["env"]
    assert cmd[0] == run_cli.sys.executable
    assert cmd[1] == str(script)
    assert cmd[2] == str(package)
    assert "--input" in cmd and str(image) in cmd
    assert cmd.count("--input") == 2
    assert f"vst_right_image={image}" in cmd
    assert cmd.count("--pipeline") == 2
    assert "--output-dir" in cmd and str(output_dir) in cmd
    assert "--dump" in cmd and "all" in cmd
    assert "--duration" in cmd and "3.5" in cmd
    assert "--loop" in cmd
    assert "--keep-running" in cmd
    assert "--use-vst" in cmd
    assert "--backend" in cmd and "cpu" in cmd
    assert "--interval-ms" in cmd and "25" in cmd
    assert "--apk" in cmd and str(apk) in cmd
    assert "--device" in cmd and "serial-1" in cmd


def test_run_device_invokes_spatial_runner_script(monkeypatch, tmp_path):
    package = tmp_path / "pkg"
    _write_device_package_manifest(package, supported_modes=["spatial"])
    script = tmp_path / "run_spatial_pipeline.py"
    script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(run_cli, "_spatial_runner_script", lambda: script)

    def _run(cmd, **_kwargs):
        calls.append(cmd)

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(run_cli.subprocess, "run", _run)

    assert run_cli.run_device(package, mode="spatial") == 0

    cmd = calls[0]
    assert cmd[0] == run_cli.sys.executable
    assert cmd[1] == str(script)
    assert cmd[2] == str(package)
    assert "--duration" in cmd and "15.0" in cmd


def test_run_device_auto_selects_spatial_runner_by_default(monkeypatch, tmp_path):
    package = tmp_path / "pkg"
    _write_device_package_manifest(package)
    script = tmp_path / "run_spatial_pipeline.py"
    script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(run_cli, "_spatial_runner_script", lambda: script)

    def _run(cmd, **_kwargs):
        calls.append(cmd)
        class Result:
            returncode = 0
        return Result()

    monkeypatch.setattr(run_cli.subprocess, "run", _run)

    assert run_cli.run_device(package) == 0
    assert calls[0][1] == str(script)
    assert "--duration" in calls[0] and "15.0" in calls[0]
    assert "--timeout" not in calls[0]


def test_run_device_auto_selects_xr_for_xr_only_manifest(monkeypatch, tmp_path):
    package = tmp_path / "pkg"
    _write_device_package_manifest(package, supported_modes=["xr"])
    script = tmp_path / "run_xr_pipeline.py"
    script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(run_cli, "_xr_runner_script", lambda: script)

    def _run(cmd, **_kwargs):
        calls.append(cmd)

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(run_cli.subprocess, "run", _run)

    assert run_cli.run_device(package) == 0
    assert calls[0][1] == str(script)


def test_run_device_auto_prefers_spatial_when_both_modes_are_supported(monkeypatch, tmp_path):
    package = tmp_path / "pkg"
    _write_device_package_manifest(package, supported_modes=["xr", "spatial"])
    script = tmp_path / "run_spatial_pipeline.py"
    script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(run_cli, "_spatial_runner_script", lambda: script)

    def _run(cmd, **_kwargs):
        calls.append(cmd)

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(run_cli.subprocess, "run", _run)

    assert run_cli.run_device(package) == 0
    assert calls[0][1] == str(script)


def test_run_device_json_captures_runner_output(monkeypatch, capsys, tmp_path):
    package = tmp_path / "pkg"
    _write_device_package_manifest(package)
    script = tmp_path / "run_spatial_pipeline.py"
    script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    monkeypatch.setattr(run_cli, "_spatial_runner_script", lambda: script)

    def _run(cmd, **_kwargs):
        class Result:
            returncode = 0
            stdout = "runner stdout"
            stderr = "runner stderr"

        return Result()

    monkeypatch.setattr(run_cli.subprocess, "run", _run)

    assert run_cli.run_device(package, as_json=True) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["command"] == "run.device"
    assert payload["mode"] == "spatial"
    assert payload["target"] == str(package)
    assert payload["stdout"] == "runner stdout"
    assert payload["stderr"] == "runner stderr"


def test_run_device_spatial_requires_spatial_supported_mode(tmp_path):
    package = tmp_path / "pkg"
    _write_device_package_manifest(package, supported_modes=["xr"])

    with pytest.raises(run_cli.RunCliError, match="does not include 'spatial'"):
        run_cli.run_device(package, mode="spatial")


def test_xr_runner_parse_status_ignores_empty_and_invalid_json():
    runner = _load_xr_runner_script()

    assert runner.parse_status("") == {}
    assert runner.parse_status("not-json") == {}
    assert runner.parse_status("[1, 2]") == {}
    assert runner.parse_status('{"state":"complete","outputs_written":6}') == {
        "state": "complete",
        "outputs_written": 6,
    }


def test_xr_runner_prepare_package_rejects_zip_path_traversal(tmp_path):
    runner = _load_xr_runner_script()
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as package_zip:
        package_zip.writestr("../evil.txt", "bad")
        package_zip.writestr("manifest.json", '{"schema_version":"2","id":"bad","pipelines":[]}')

    with pytest.raises(SystemExit, match="Unsafe zip entry path"):
        runner.prepare_package(archive, tmp_path / "extract")
    assert not (tmp_path.parent / "evil.txt").exists()


def test_xr_runner_wait_for_outputs_waits_for_complete_status(monkeypatch):
    runner = _load_xr_runner_script()
    statuses = iter(
        [
            "",
            '{"state":"submitted","outputs_written":1}',
            '{"state":"complete","outputs_written":6}',
        ]
    )
    sleeps = []

    monkeypatch.setattr(runner, "run_as_capture", lambda *_args, **_kwargs: next(statuses))
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: sleeps.append(seconds))

    runner.wait_for_outputs(["adb"], duration=5.0)

    assert sleeps == [0.25, 0.25]


def test_xr_runner_override_model_backend_patches_staged_package(tmp_path):
    runner = _load_xr_runner_script()
    package = tmp_path / "pkg"
    pipeline_dir = package / "pipeline"
    pipeline_dir.mkdir(parents=True)
    _write_json(
        package / "manifest.json",
        {
            "id": "demo",
            "pipelines": [{"id": "main", "path": "pipeline/main.json"}],
        },
    )
    _write_json(
        pipeline_dir / "main.json",
        {
            "operators": [
                {
                    "type": "XR_SECURE_MR_OPERATOR_TYPE_RUN_MODEL_INFERENCE_PICO",
                    "model_target": "npu",
                    "model": {"model_target": "npu"},
                },
                {"type": "XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO"},
            ]
        },
    )

    result = runner.override_model_backend(package, "gpu", tmp_path)

    spec = json.loads((result / "pipeline" / "main.json").read_text(encoding="utf-8"))
    assert spec["operators"][0]["model_target"] == "gpu"
    assert spec["operators"][0]["model"]["model_target"] == "gpu"
    assert "model_target" not in spec["operators"][1]


def test_spatial_runner_compat_fields_are_staged_without_mutating_source(tmp_path):
    package = tmp_path / "pkg"
    pipeline_dir = package / "pipeline"
    pipeline_dir.mkdir(parents=True)
    _write_json(
        package / "manifest.json",
        {
            "schema_version": "2",
            "id": "demo",
            "pipelines": [{"id": "main", "path": "pipeline/main.json"}],
            "runtime": {"supported_modes": ["spatial"]},
        },
    )
    _write_json(
        pipeline_dir / "main.json",
        {
            "operators": [
                {
                    "type": "XR_SECURE_MR_OPERATOR_TYPE_RUN_MODEL_INFERENCE_PICO",
                    "model": {
                        "bin_path": "model/main.tflite",
                        "model_name": "main",
                        "model_type": "tflite",
                    },
                }
            ]
        },
    )
    original_manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))

    result = device_runner_base.add_spatial_runner_compat_fields(package, tmp_path / "stage")

    staged_manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
    assert result != package
    assert json.loads((package / "manifest.json").read_text(encoding="utf-8")) == original_manifest
    assert staged_manifest["package_type"] == "spatial_pipeline"
    assert staged_manifest["format_version"] == 1
    assert staged_manifest["model"] == {"bin_path": "model/main.tflite", "model_name": "main"}


def test_device_runner_collects_gltf_output_metadata(tmp_path):
    package = tmp_path / "pkg"
    pipeline_dir = package / "pipeline"
    gltf_dir = package / "gltf"
    pipeline_dir.mkdir(parents=True)
    gltf_dir.mkdir()
    (gltf_dir / "frame.gltf").write_text("{}", encoding="utf-8")
    _write_json(
        package / "manifest.json",
        {
            "id": "demo",
            "pipelines": [{"id": "display", "path": "pipeline/display.json"}],
        },
    )
    _write_json(
        pipeline_dir / "display.json",
        {
            "tensors": {
                "frame_pose": {"dimensions": [4, 4], "channels": 1, "data_type": 6},
                "frame_gltf": {"tensor_type": "gltf", "asset": "gltf/frame.gltf"},
            },
            "outputs": ["frame_pose", "frame_gltf"],
        },
    )

    assert device_runner_base.collect_asset_output_metadata(package) == [
        {
            "pipeline": "display",
            "tensor": "frame_gltf",
            "kind": "asset",
            "is_output": True,
            "written": False,
            "reason": "asset_reference",
            "asset": "gltf/frame.gltf",
            "exists": True,
        }
    ]


def test_device_runner_stops_xr_and_spatial_apps(monkeypatch):
    commands = []

    def _run(cmd, **_kwargs):
        commands.append(cmd)

    monkeypatch.setattr(device_runner_base, "run", _run)

    device_runner_base.stop_runner_apps(["adb"])

    assert commands == [
        ["adb", "shell", "am", "force-stop", device_runner_base.XR_CONFIG.package_name],
        ["adb", "shell", "am", "force-stop", device_runner_base.SPATIAL_CONFIG.package_name],
    ]


def test_xr_runner_parse_input_args_supports_bare_and_named(tmp_path):
    runner = _load_xr_runner_script()
    image = tmp_path / "face.jpg"
    left = tmp_path / "left.jpg"
    image.write_bytes(b"image")
    left.write_bytes(b"left")

    defaults, named = runner.parse_input_args([str(image), f"vst_left_image={left}"])

    assert defaults == [image]
    assert named == [("vst_left_image", left)]


def test_xr_runner_dump_all_rejects_named_device_dumps():
    runner = _load_xr_runner_script()

    assert runner.dump_all(["all"]) is True
    assert runner.dump_all([]) is False
    with pytest.raises(SystemExit, match="only supports --dump all"):
        runner.dump_all(["tensor"])


def test_xr_runner_pull_app_outputs_routes_dump_only_tensors(monkeypatch, tmp_path):
    runner = _load_xr_runner_script()
    files = [
        "files/outputs/status.json",
        "files/outputs/detection_post_det_1.bin",
        "files/outputs/display_post_det_1.bin",
    ]
    status = {
        "outputs_metadata": [
            {"file": "detection_post_det_1.bin", "is_output": True},
            {"file": "display_post_det_1.bin", "is_output": False},
        ]
    }

    monkeypatch.setattr(runner, "run_as_capture", lambda *_args, **_kwargs: "\n".join(files))

    def _run(cmd, **_kwargs):
        class Result:
            stdout = b""

        result = Result()
        remote_path = cmd[-1]
        if remote_path.endswith("status.json"):
            result.stdout = json.dumps(status).encode()
        elif remote_path.endswith(".bin"):
            result.stdout = b"tensor"
        return result

    monkeypatch.setattr(runner.subprocess, "run", _run)

    local_outputs = runner.pull_app_outputs(["adb"], tmp_path)

    assert local_outputs == tmp_path
    assert (tmp_path / "status.json").is_file()
    assert (tmp_path / "detection" / "detection_post_det_1.bin").is_file()
    assert (tmp_path / "display" / "all_tensors" / "display_post_det_1.bin").is_file()


def test_xr_runner_device_summary_prints_runtime_modes(monkeypatch, capsys, tmp_path):
    runner = _load_xr_runner_script()
    output_root = tmp_path / "outputs"
    detection_dir = output_root / "detection"
    detection_dir.mkdir(parents=True)
    (output_root / "status.json").write_text(
        json.dumps(
            {
                "runtime_modes": ["xr"],
                "pipelines": ["detection"],
                "total_elapsed_ms": 42,
                "outputs_metadata": [
                    {"file": "detection_post_det_1.bin", "pipeline": "detection", "is_output": True}
                ],
            }
        ),
        encoding="utf-8",
    )
    (detection_dir / "detection_post_det_1.bin").write_bytes(b"\0" * 84)
    monkeypatch.setattr(runner, "collect_relevant_logs", lambda _adb: [])

    runner.print_device_summary(["adb"], output_root)

    captured = capsys.readouterr()
    assert "Runtime modes: xr" in captured.out
    assert "Pipelines: detection" in captured.out
    assert "Total time: 42 ms" in captured.out
    assert "Submit time:" not in captured.out
    assert "Pipeline time:" not in captured.out


def test_device_runner_summary_prints_asset_references_without_dump_section(monkeypatch, capsys, tmp_path):
    output_root = tmp_path / "outputs"
    display_dir = output_root / "display"
    dump_dir = display_dir / "all_tensors"
    dump_dir.mkdir(parents=True)
    (display_dir / "display_frame_pose_1.bin").write_bytes(np.eye(4, dtype=np.float32).tobytes())
    (dump_dir / "display_post_det_1.bin").write_bytes(np.zeros((1, 21), dtype=np.float32).tobytes())
    (output_root / "status.json").write_text(
        json.dumps(
            {
                "runtime_modes": ["spatial"],
                "pipelines": ["display"],
                "outputs_metadata": [
                    {
                        "file": "display_frame_pose_1.bin",
                        "pipeline": "display",
                        "tensor": "frame_pose",
                        "dtype": "float32",
                        "data_type": 6,
                        "channels": 1,
                        "bytes": 64,
                        "is_output": True,
                        "shape": [4, 4],
                    },
                    {
                        "pipeline": "display",
                        "tensor": "frame_gltf",
                        "kind": "asset",
                        "asset": "gltf/frame.gltf",
                        "is_output": True,
                        "written": False,
                        "reason": "asset_reference",
                        "exists": True,
                    },
                    {
                        "file": "display_post_det_1.bin",
                        "pipeline": "display",
                        "tensor": "post_det",
                        "dtype": "float32",
                        "data_type": 6,
                        "channels": 1,
                        "bytes": 84,
                        "is_output": False,
                        "shape": [1, 21],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(device_runner_base, "collect_relevant_logs", lambda _config, _adb: [])

    device_runner_base.print_device_summary(device_runner_base.SPATIAL_CONFIG, ["adb"], output_root)

    captured = capsys.readouterr()
    assert "Outputs: 2" in captured.out
    assert "display_frame_pose_1.bin: 64 bytes shape=(4, 4) dtype=float32" in captured.out
    assert "frame_gltf: asset reference gltf/frame.gltf exists=yes" in captured.out
    assert "Dumped tensors:" not in captured.out
    assert "display_post_det_1.bin" not in captured.out


def test_xr_runner_wait_for_outputs_prints_logs_on_timeout(monkeypatch, capsys):
    runner = _load_xr_runner_script()

    monkeypatch.setattr(runner, "run_as_capture", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(runner.time, "monotonic", iter([0.0, 1.0]).__next__)
    monkeypatch.setattr(runner, "collect_relevant_logs", lambda _adb: ["runner log"])

    with pytest.raises(SystemExit, match="Timed out after 0.5s"):
        runner.wait_for_outputs(["adb"], duration=0.5)

    captured = capsys.readouterr()
    assert "Device Logs" in captured.out
    assert "runner log" in captured.out


def test_xr_runner_device_tensor_summary_uses_metadata_preview(tmp_path):
    runner = _load_xr_runner_script()
    output = tmp_path / "tensor.bin"
    np.arange(10, dtype=np.float32).tofile(output)

    summary = runner.device_tensor_summary(
        output,
        {
            "shape": [1, 10],
            "channels": 1,
            "dtype": "float32",
            "data_type": 6,
            "bytes": 40,
        },
    )

    assert "tensor.bin: 40 bytes shape=(1, 10) dtype=float32" in summary
    assert "min=0 max=9 mean=4.5" in summary
    assert "preview=[0, 1, 2, 3, 4, 5, 6, 7, ...]" in summary


def test_xr_runner_groups_outputs_by_pipeline(tmp_path):
    runner = _load_xr_runner_script()
    detection = tmp_path / "detection_post_det_1.bin"
    display = tmp_path / "display_frame_pose_1.bin"
    untagged = tmp_path / "custom_tensor_1.bin"
    for path in (detection, display, untagged):
        path.write_bytes(b"\0")

    grouped = runner.group_outputs_by_pipeline(
        [detection, display, untagged],
        {
            detection.name: {"pipeline": "detection"},
            display.name: {"pipeline": "display"},
        },
    )

    assert grouped == [
        ("detection", [detection]),
        ("display", [display]),
        ("custom", [untagged]),
    ]


def test_xr_runner_collect_relevant_logs_prioritizes_securemr_and_litert(monkeypatch):
    runner = _load_xr_runner_script()
    calls = []
    runner_line = f"01-01 I testbench: {runner.RUNNER_LOG_SAMPLE}: wrote readback"
    secure_line = f"01-01 I {runner.SECUREMR_LOG_TAG_SAMPLE}: inference timing"

    class Result:
        stdout = "\n".join(
            [
                "01-01 I unrelated: ignore",
                runner_line,
                secure_line,
                "01-01 E backend: ackReadbackTensorContent failed: [INVALID PARAMETER]; tensor has no shared memory",
            ]
        ).encode()
        stderr = b""

    def _run(cmd, **_kwargs):
        calls.append(cmd)
        return Result()

    monkeypatch.setattr(runner.subprocess, "run", _run)

    logs = runner.collect_relevant_logs(["adb"])
    joined = "\n".join(logs)

    assert "-b" in calls[0] and runner.LOGCAT_BUFFERS in calls[0]
    assert secure_line in joined
    assert runner_line in joined
    assert "unrelated" not in joined
    assert "ackReadbackTensorContent" not in joined


def test_xr_runner_benign_readback_filter_matches_message_variants():
    runner = _load_xr_runner_script()

    assert runner.is_benign_readback_log(
        "ackReadbackTensorContent >>> [INVALID PARAMETER]: no shared memory associated with this tensor"
    )
    assert runner.is_benign_readback_log(
        "E backend: ackReadbackTensorContent failed: [invalid parameter]; tensor has no shared memory"
    )
    assert not runner.is_benign_readback_log(
        "ackReadbackTensorContent failed: permission denied"
    )


def test_run_device_rejects_missing_target(tmp_path):
    with pytest.raises(run_cli.RunCliError, match="Target package not found"):
        run_cli.run_device(tmp_path / "missing.zip")


def test_run_device_validates_inputs_before_subprocess(monkeypatch, tmp_path):
    package = tmp_path / "pkg"
    _write_device_package_manifest(package)
    apk = tmp_path / "missing.apk"
    calls = []
    monkeypatch.setattr(run_cli.subprocess, "run", lambda cmd, **_kwargs: calls.append(cmd))

    with pytest.raises(run_cli.RunCliError, match="--duration"):
        run_cli.run_device(package, duration=0)
    with pytest.raises(run_cli.RunCliError, match="--interval-ms"):
        run_cli.run_device(package, interval_ms=0)
    with pytest.raises(run_cli.RunCliError, match="only supports --dump all"):
        run_cli.run_device(package, dumps=["tensor"])
    with pytest.raises(run_cli.RunCliError, match="Runner APK not found"):
        run_cli.run_device(package, apk=apk)
    with pytest.raises(run_cli.RunCliError, match="Input file not found"):
        run_cli.run_device(package, inputs=[str(tmp_path / "missing.jpg")])

    assert calls == []
