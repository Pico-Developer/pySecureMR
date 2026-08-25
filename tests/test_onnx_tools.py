import subprocess
from pathlib import Path

import pytest

from pyspatialml import onnx_tools


def _managed_spec():
    return "\n".join(["onnx2tf", *onnx_tools.DEFAULT_ONNX2TF_DEPENDENCIES]) + "\n"


def _write_executable(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_resolve_onnx2tf_uses_healthy_managed_cache(tmp_path, monkeypatch):
    managed = _write_executable(onnx_tools.managed_onnx2tf_bin(tmp_path))
    onnx_tools.managed_onnx2tf_spec_file(tmp_path).write_text(_managed_spec(), encoding="utf-8")
    monkeypatch.setattr(onnx_tools.shutil, "which", lambda _name: None)
    monkeypatch.setattr(onnx_tools, "_is_healthy", lambda path: path == managed)

    resolved = onnx_tools.resolve_onnx2tf(cache_dir=tmp_path)

    assert resolved.path == managed
    assert resolved.managed


def test_resolve_onnx2tf_prefers_managed_when_ensuring(tmp_path, monkeypatch):
    managed = _write_executable(onnx_tools.managed_onnx2tf_bin(tmp_path))
    system = _write_executable(tmp_path / "system" / "onnx2tf")
    onnx_tools.managed_onnx2tf_spec_file(tmp_path).write_text(_managed_spec(), encoding="utf-8")
    monkeypatch.setattr(onnx_tools.shutil, "which", lambda _name: str(system))
    monkeypatch.setattr(onnx_tools, "_is_healthy", lambda path: path in {managed, system})

    resolved = onnx_tools.resolve_onnx2tf(ensure=True, cache_dir=tmp_path)

    assert resolved.path == managed
    assert resolved.managed


def test_resolve_onnx2tf_repairs_corrupt_managed_cache(tmp_path, monkeypatch):
    managed = _write_executable(onnx_tools.managed_onnx2tf_bin(tmp_path))
    onnx_tools.managed_onnx2tf_spec_file(tmp_path).write_text("onnx2tf\n", encoding="utf-8")
    monkeypatch.setattr(onnx_tools.shutil, "which", lambda _name: None)
    monkeypatch.setattr(onnx_tools, "_is_healthy", lambda _path: False)
    calls = []

    def _install_onnx2tf(*, cache_dir=None, package="", version="", force=False, verbose=False):
        calls.append((cache_dir, package, version, force))
        return onnx_tools.Onnx2TfCli(path=managed, managed=True)

    monkeypatch.setattr(onnx_tools, "install_onnx2tf", _install_onnx2tf)

    resolved = onnx_tools.resolve_onnx2tf(ensure=True, cache_dir=tmp_path)

    assert resolved.path == managed
    assert resolved.managed
    assert calls == [(tmp_path, "onnx2tf", "", True)]


def test_install_onnx2tf_wraps_pip_failures(tmp_path, monkeypatch):
    python = tmp_path / "onnx2tf" / "bin" / "python"
    _write_executable(python)

    def _run(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, ["pip"], stderr="broken install")

    monkeypatch.setattr(onnx_tools.subprocess, "run", _run)

    with pytest.raises(onnx_tools.OnnxToolError, match="Failed to install onnx2tf: broken install"):
        onnx_tools.install_onnx2tf(cache_dir=tmp_path)


def test_install_onnx2tf_installs_runtime_dependencies(tmp_path, monkeypatch):
    python = tmp_path / "onnx2tf" / "bin" / "python"
    cli = _write_executable(onnx_tools.managed_onnx2tf_bin(tmp_path))
    _write_executable(python)
    calls = []

    def _run(argv, **_kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(onnx_tools.subprocess, "run", _run)
    monkeypatch.setattr(onnx_tools, "_is_healthy", lambda path: path == cli)

    result = onnx_tools.install_onnx2tf(cache_dir=tmp_path)

    assert result.path == cli
    assert result.managed
    assert calls == [
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "onnx2tf",
            *onnx_tools.DEFAULT_ONNX2TF_DEPENDENCIES,
        ]
    ]
    assert onnx_tools.managed_onnx2tf_spec_file(tmp_path).read_text(encoding="utf-8") == _managed_spec()


def test_install_onnx2tf_wraps_venv_creation_failures(tmp_path, monkeypatch):
    class _FailingEnvBuilder:
        def __init__(self, **_kwargs):
            pass

        def create(self, _venv_dir):
            raise OSError("no permission")

    monkeypatch.setattr(onnx_tools.venv, "EnvBuilder", _FailingEnvBuilder)

    with pytest.raises(onnx_tools.OnnxToolError, match="Failed to create managed onnx2tf environment"):
        onnx_tools.install_onnx2tf(cache_dir=tmp_path)


def test_convert_onnx_to_tflite_runs_managed_converter(tmp_path, monkeypatch):
    model = tmp_path / "model.onnx"
    output = tmp_path / "out"
    tflite = output / "model.tflite"
    model.write_bytes(b"onnx")
    tool = onnx_tools.Onnx2TfCli(path=tmp_path / "bin" / "onnx2tf", managed=True)
    calls = []

    monkeypatch.setattr(onnx_tools, "resolve_onnx2tf", lambda ensure=False, cache_dir=None, verbose=False: tool)

    def _run(argv, **kwargs):
        calls.append((argv, kwargs))
        cwd = Path(kwargs["cwd"])
        assert cwd
        assert (cwd / "calibration_image_sample_data_20x128x128x3_float32.npy").is_file()
        tflite.parent.mkdir(parents=True, exist_ok=True)
        tflite.write_bytes(b"tflite")
        return subprocess.CompletedProcess(argv, 0, stdout="stdout", stderr="stderr")

    monkeypatch.setattr(onnx_tools.subprocess, "run", _run)

    result = onnx_tools.convert_onnx_to_tflite(
        model=model,
        output=output,
        extra_args=["--verbosity", "debug"],
        cache_dir=tmp_path / "cache",
    )

    assert result.tool == tool
    assert result.output == output
    assert result.tflite_models == [tflite]
    assert result.stdout == "stdout"
    assert result.stderr == "stderr"
    assert calls[0][0] == [str(tool.path), "-i", str(model), "-o", str(output), "--verbosity", "debug"]


def test_convert_onnx_to_tflite_resolves_relative_output_directory(tmp_path, monkeypatch):
    model = tmp_path / "model.onnx"
    output = Path("relative-out")
    model.write_bytes(b"onnx")
    tool = onnx_tools.Onnx2TfCli(path=tmp_path / "bin" / "onnx2tf", managed=True)
    calls = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(onnx_tools, "resolve_onnx2tf", lambda ensure=False, cache_dir=None, verbose=False: tool)

    def _run(argv, **kwargs):
        calls.append((argv, kwargs))
        conversion_output = Path(argv[argv.index("-o") + 1])
        assert conversion_output.is_absolute()
        tflite = conversion_output / "model.tflite"
        tflite.parent.mkdir(parents=True, exist_ok=True)
        tflite.write_bytes(b"tflite")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(onnx_tools.subprocess, "run", _run)

    result = onnx_tools.convert_onnx_to_tflite(model=model, output=output)

    assert result.output == tmp_path / output
    assert result.tflite_models == [tmp_path / output / "model.tflite"]


def test_convert_onnx_to_tflite_resolves_relative_model_path(tmp_path, monkeypatch):
    model = Path("model.onnx")
    output = tmp_path / "out"
    model_path = tmp_path / model
    model_path.write_bytes(b"onnx")
    tool = onnx_tools.Onnx2TfCli(path=tmp_path / "bin" / "onnx2tf", managed=True)
    calls = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(onnx_tools, "resolve_onnx2tf", lambda ensure=False, cache_dir=None, verbose=False: tool)

    def _run(argv, **kwargs):
        calls.append((argv, kwargs))
        converted_model = Path(argv[argv.index("-i") + 1])
        conversion_output = Path(argv[argv.index("-o") + 1])
        assert converted_model == model_path
        assert converted_model.is_absolute()
        assert Path(kwargs["cwd"]) != tmp_path
        tflite = conversion_output / "model.tflite"
        tflite.parent.mkdir(parents=True, exist_ok=True)
        tflite.write_bytes(b"tflite")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(onnx_tools.subprocess, "run", _run)

    result = onnx_tools.convert_onnx_to_tflite(model=model, output=output)

    assert result.model == model_path
    assert result.tflite_models == [output / "model.tflite"]
    assert calls[0][0][calls[0][0].index("-i") + 1] == str(model_path)


def test_convert_onnx_to_tflite_supports_file_output(tmp_path, monkeypatch):
    model = tmp_path / "model.onnx"
    output = tmp_path / "renamed.tflite"
    model.write_bytes(b"onnx")
    tool = onnx_tools.Onnx2TfCli(path=tmp_path / "bin" / "onnx2tf", managed=True)
    calls = []

    monkeypatch.setattr(onnx_tools, "resolve_onnx2tf", lambda ensure=False, cache_dir=None, verbose=False: tool)

    def _run(argv, **kwargs):
        calls.append((argv, kwargs))
        conversion_output = Path(argv[argv.index("-o") + 1])
        assert conversion_output != output
        tflite = conversion_output / "converted.tflite"
        tflite.parent.mkdir(parents=True, exist_ok=True)
        tflite.write_bytes(b"tflite")
        return subprocess.CompletedProcess(argv, 0, stdout="stdout", stderr="")

    monkeypatch.setattr(onnx_tools.subprocess, "run", _run)

    result = onnx_tools.convert_onnx_to_tflite(model=model, output=output)

    assert result.output == output
    assert result.tflite_models == [output]
    assert output.read_bytes() == b"tflite"
    assert calls[0][0][:4] == [str(tool.path), "-i", str(model), "-o"]


def test_convert_onnx_to_tflite_rejects_multiple_models_for_file_output(tmp_path, monkeypatch):
    model = tmp_path / "model.onnx"
    output = tmp_path / "renamed.tflite"
    model.write_bytes(b"onnx")
    tool = onnx_tools.Onnx2TfCli(path=tmp_path / "bin" / "onnx2tf", managed=True)

    monkeypatch.setattr(onnx_tools, "resolve_onnx2tf", lambda ensure=False, cache_dir=None, verbose=False: tool)

    def _run(argv, **kwargs):
        conversion_output = Path(argv[argv.index("-o") + 1])
        conversion_output.mkdir(parents=True, exist_ok=True)
        (conversion_output / "a.tflite").write_bytes(b"a")
        (conversion_output / "b.tflite").write_bytes(b"b")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(onnx_tools.subprocess, "run", _run)

    with pytest.raises(onnx_tools.OnnxToolError) as exc_info:
        onnx_tools.convert_onnx_to_tflite(model=model, output=output)

    message = str(exc_info.value)
    assert "produced multiple .tflite files" in message
    assert "Use an output directory instead" in message


def test_convert_onnx_to_tflite_reports_converter_output_when_missing_tflite(tmp_path, monkeypatch):
    model = tmp_path / "model.onnx"
    output = tmp_path / "missing.tflite"
    model.write_bytes(b"onnx")
    tool = onnx_tools.Onnx2TfCli(path=tmp_path / "bin" / "onnx2tf", managed=True)

    monkeypatch.setattr(onnx_tools, "resolve_onnx2tf", lambda ensure=False, cache_dir=None, verbose=False: tool)

    def _run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout="converter stdout", stderr="converter stderr")

    monkeypatch.setattr(onnx_tools.subprocess, "run", _run)

    with pytest.raises(onnx_tools.OnnxToolError) as exc_info:
        onnx_tools.convert_onnx_to_tflite(model=model, output=output)

    message = str(exc_info.value)
    assert "no .tflite file was found for output file" in message
    assert "converter stdout" in message
    assert "converter stderr" in message
