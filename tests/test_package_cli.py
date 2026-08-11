import json
import zipfile

import pytest

from pyspatialml import package_cli


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _pipeline_with_model(model_path="face.tflite"):
    return {
        "tensors": {},
        "operators": [
            {
                "type": "XR_SECURE_MR_OPERATOR_TYPE_RUN_MODEL_INFERENCE_PICO",
                "inputs": [],
                "outputs": [],
                "model_type": "tflite",
                "model": {
                    "bin_path": model_path,
                    "model_name": "face",
                    "model_type": "tflite",
                },
            }
        ],
        "inputs": [],
        "outputs": [],
    }


def test_create_package_normalizes_pipeline_and_model_paths(tmp_path):
    source_dir = tmp_path / "src"
    pipeline = source_dir / "detection.json"
    model = source_dir / "face.tflite"
    _write_json(pipeline, _pipeline_with_model("face.tflite"))
    model.write_bytes(b"model")

    output = tmp_path / "pkg"
    package_cli.create_package(
        package_id="face-demo",
        pipelines=[f"detection={pipeline}"],
        output=output,
        supported_modes=["spatial"],
    )

    manifest = _read_json(output / "manifest.json")
    packaged_pipeline = _read_json(output / "pipeline" / "detection.json")
    assert manifest["schema_version"] == "2"
    assert manifest["id"] == "face-demo"
    assert manifest["pipelines"] == [{"id": "detection", "path": "pipeline/detection.json"}]
    assert manifest["runtime"]["supported_modes"] == ["spatial"]
    assert (output / "model" / "face.tflite").read_bytes() == b"model"
    assert packaged_pipeline["operators"][0]["model"]["bin_path"] == "model/face.tflite"
    assert "model_file" not in packaged_pipeline["operators"][0]
    assert "model_asset" not in packaged_pipeline["operators"][0]
    assert "model_id" not in packaged_pipeline["operators"][0]


def test_create_package_resolves_model_from_asset_root(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    asset_root = tmp_path / "assets"
    model = asset_root / "model" / "face.tflite"
    _write_json(pipeline, _pipeline_with_model("model/face.tflite"))
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model")

    output = tmp_path / "pkg"
    package_cli.create_package(
        package_id="face-demo",
        pipelines=[f"main={pipeline}"],
        output=output,
        asset_roots=[asset_root],
    )

    assert (output / "model" / "face.tflite").read_bytes() == b"model"
    assert _read_json(output / "pipeline" / "main.json")["operators"][0]["model"]["bin_path"] == "model/face.tflite"


def test_create_package_normalizes_gltf_tensor_assets(tmp_path):
    source_dir = tmp_path / "src"
    pipeline = source_dir / "display.json"
    gltf = source_dir / "frame.gltf"
    _write_json(
        pipeline,
        {
            "tensors": {
                "scene": {
                    "tensor_type": "gltf",
                    "asset": "frame.gltf",
                    "is_placeholder": True,
                }
            },
            "operators": [],
            "inputs": [],
            "outputs": [],
        },
    )
    gltf.write_text("{}", encoding="utf-8")

    output = tmp_path / "pkg"
    package_cli.create_package(
        package_id="display-demo",
        pipelines=[f"display={pipeline}"],
        output=output,
    )

    packaged_pipeline = _read_json(output / "pipeline" / "display.json")
    assert (output / "gltf" / "frame.gltf").read_text(encoding="utf-8") == "{}"
    assert packaged_pipeline["tensors"]["scene"]["asset"] == "gltf/frame.gltf"


def test_create_package_errors_for_missing_asset(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    _write_json(pipeline, _pipeline_with_model("missing.tflite"))

    with pytest.raises(package_cli.PackageCliError, match="Referenced asset not found"):
        package_cli.create_package(
            package_id="bad",
            pipelines=[f"main={pipeline}"],
            output=tmp_path / "pkg",
        )


def test_create_package_errors_on_model_name_collision(tmp_path):
    first = tmp_path / "a" / "pipeline.json"
    second = tmp_path / "b" / "pipeline.json"
    first_model = tmp_path / "a" / "shared.tflite"
    second_model = tmp_path / "b" / "shared.tflite"
    _write_json(first, _pipeline_with_model("shared.tflite"))
    _write_json(second, _pipeline_with_model("shared.tflite"))
    first_model.write_bytes(b"one")
    second_model.write_bytes(b"two")

    with pytest.raises(package_cli.PackageCliError, match="Asset name collision"):
        package_cli.create_package(
            package_id="bad",
            pipelines=[f"first={first}", f"second={second}"],
            output=tmp_path / "pkg",
        )


def test_create_package_zip_output_and_validate(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    model = tmp_path / "face.tflite"
    _write_json(pipeline, _pipeline_with_model("face.tflite"))
    model.write_bytes(b"model")

    archive = tmp_path / "pkg.zip"
    package_cli.create_package(
        package_id="face-demo",
        pipelines=[f"main={pipeline}"],
        output=archive,
        zip_output=True,
    )

    assert archive.is_file()
    with zipfile.ZipFile(archive) as package_zip:
        names = set(package_zip.namelist())
    assert "manifest.json" in names
    assert "pipeline/main.json" in names
    assert "model/face.tflite" in names
    assert package_cli.validate_package(archive) == 0


def test_validate_package_rejects_zip_path_traversal(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as package_zip:
        package_zip.writestr("../evil.txt", "bad")
        package_zip.writestr("manifest.json", '{"schema_version":"2","id":"bad","pipelines":[]}')

    with pytest.raises(package_cli.PackageCliError, match="Unsafe zip entry path"):
        package_cli.validate_package(archive)
    assert not (tmp_path.parent / "evil.txt").exists()


def test_validate_package_accepts_windows_relative_zip_paths(tmp_path):
    archive = tmp_path / "windows.zip"
    with zipfile.ZipFile(archive, "w") as package_zip:
        package_zip.writestr(
            "pkg\\manifest.json",
            json.dumps(
                {
                    "schema_version": "2",
                    "id": "demo",
                    "pipelines": [{"id": "main", "path": "pipeline/main.json"}],
                }
            ),
        )
        package_zip.writestr(
            "pkg\\pipeline\\main.json",
            json.dumps({"tensors": {}, "operators": [], "inputs": [], "outputs": []}),
        )

    assert package_cli.validate_package(archive) == 0


def test_create_package_existing_output_requires_confirmation(tmp_path, monkeypatch):
    pipeline = tmp_path / "pipeline.json"
    model = tmp_path / "face.tflite"
    output = tmp_path / "pkg"
    _write_json(pipeline, _pipeline_with_model("face.tflite"))
    model.write_bytes(b"model")
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")

    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    with pytest.raises(package_cli.PackageCliError, match="already exists"):
        package_cli.create_package(
            package_id="face-demo",
            pipelines=[f"main={pipeline}"],
            output=output,
        )
    assert (output / "old.txt").read_text(encoding="utf-8") == "old"

    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")
    assert package_cli.create_package(
        package_id="face-demo",
        pipelines=[f"main={pipeline}"],
        output=output,
    ) == 0
    assert not (output / "old.txt").exists()
    assert (output / "manifest.json").is_file()


def test_create_package_force_bypasses_confirmation(tmp_path, monkeypatch):
    pipeline = tmp_path / "pipeline.json"
    model = tmp_path / "face.tflite"
    output = tmp_path / "pkg"
    _write_json(pipeline, _pipeline_with_model("face.tflite"))
    model.write_bytes(b"model")
    output.mkdir()

    def _fail_input(_prompt):
        raise AssertionError("input should not be called when force=True")

    monkeypatch.setattr("builtins.input", _fail_input)
    assert package_cli.create_package(
        package_id="face-demo",
        pipelines=[f"main={pipeline}"],
        output=output,
        force=True,
    ) == 0


def test_inspect_package_prints_summary(capsys, tmp_path):
    pipeline = tmp_path / "pipeline.json"
    model = tmp_path / "face.tflite"
    _write_json(pipeline, _pipeline_with_model("face.tflite"))
    model.write_bytes(b"model")
    output = tmp_path / "pkg"
    package_cli.create_package(
        package_id="face-demo",
        pipelines=[f"main={pipeline}"],
        output=output,
        supported_modes=["xr"],
    )

    package_cli.inspect_package(output)

    captured = capsys.readouterr()
    assert "Package: face-demo" in captured.out
    assert "Pipelines: 1" in captured.out
    assert "main -> pipeline/main.json" in captured.out
    assert "Supported modes: xr" in captured.out
    assert "model/face.tflite" in captured.out
