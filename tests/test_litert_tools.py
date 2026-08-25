import stat

import pytest

from pyspatialml import litert_tools
from pyspatialml.litert_tools import (
    LiteRTCli,
    LiteRTToolError,
    managed_litert_bin,
    managed_litert_spec_file,
    resolve_litert_cli,
)


def _write_executable(path, text="#!/bin/sh\nprintf 'litert 2.0.3\\n'\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _hide_current_python_litert(tmp_path, monkeypatch):
    local_python = tmp_path / "no-litert-venv" / "bin" / "python"
    local_python.parent.mkdir(parents=True, exist_ok=True)
    local_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(litert_tools.sys, "executable", str(local_python))


def test_resolve_litert_uses_env_override(tmp_path, monkeypatch):
    litert = _write_executable(tmp_path / "custom" / "litert")
    monkeypatch.setenv("PYSPATIALML_LITERT", str(litert))
    monkeypatch.setenv("PATH", "")

    resolved = resolve_litert_cli()

    assert resolved.path == litert
    assert not resolved.managed


def test_resolve_litert_rejects_bad_env_override(tmp_path, monkeypatch):
    litert = tmp_path / "missing-litert"
    monkeypatch.setenv("PYSPATIALML_LITERT", str(litert))
    monkeypatch.setenv("PATH", "")

    with pytest.raises(LiteRTToolError, match="PYSPATIALML_LITERT"):
        resolve_litert_cli()


def test_resolve_litert_uses_path(tmp_path, monkeypatch):
    litert = _write_executable(tmp_path / "bin" / "litert")
    monkeypatch.delenv("PYSPATIALML_LITERT", raising=False)
    monkeypatch.setenv("PATH", str(litert.parent))
    _hide_current_python_litert(tmp_path, monkeypatch)

    resolved = resolve_litert_cli()

    assert resolved.path == litert
    assert not resolved.managed


def test_resolve_litert_prefers_current_python_environment(tmp_path, monkeypatch):
    local_python = tmp_path / "venv" / "bin" / "python"
    local_python.parent.mkdir(parents=True)
    local_python.write_text("", encoding="utf-8")
    local_litert = _write_executable(local_python.parent / "litert")
    stale_managed = _write_executable(managed_litert_bin(tmp_path))
    managed_litert_spec_file(tmp_path).write_text("litert-cli==0.0.1\n", encoding="utf-8")
    monkeypatch.delenv("PYSPATIALML_LITERT", raising=False)
    monkeypatch.setenv("PATH", str(stale_managed.parent))
    monkeypatch.setattr(litert_tools.sys, "executable", str(local_python))

    resolved = resolve_litert_cli(cache_dir=tmp_path, ensure=True)

    assert resolved.path == local_litert
    assert not resolved.managed


def test_resolve_litert_uses_managed_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("PYSPATIALML_LITERT", raising=False)
    monkeypatch.setenv("PATH", "")
    _hide_current_python_litert(tmp_path, monkeypatch)
    managed = _write_executable(managed_litert_bin(tmp_path))
    managed_litert_spec_file(tmp_path).write_text("litert-cli==0.1.0\n", encoding="utf-8")

    resolved = resolve_litert_cli(cache_dir=tmp_path)

    assert resolved.path == managed
    assert resolved.managed


def test_resolve_litert_upgrades_managed_cache_when_spec_changes(tmp_path, monkeypatch):
    monkeypatch.delenv("PYSPATIALML_LITERT", raising=False)
    monkeypatch.setenv("PATH", "")
    _hide_current_python_litert(tmp_path, monkeypatch)
    managed = _write_executable(managed_litert_bin(tmp_path))
    managed_litert_spec_file(tmp_path).write_text("litert-cli==0.0.1\n", encoding="utf-8")
    calls = []

    def _install_litert_cli(*, cache_dir=None, package="", version=""):
        calls.append((cache_dir, package, version))
        managed_litert_spec_file(cache_dir).write_text(f"{package}=={version}\n", encoding="utf-8")
        return LiteRTCli(path=managed, managed=True)

    monkeypatch.setattr(litert_tools, "install_litert_cli", _install_litert_cli)

    resolved = resolve_litert_cli(cache_dir=tmp_path, ensure=True, version="0.1.0")

    assert resolved.path == managed
    assert resolved.managed
    assert calls == [(tmp_path, "litert-cli", "0.1.0")]


def test_resolve_litert_missing_tool_reports_install_hint(tmp_path, monkeypatch):
    monkeypatch.delenv("PYSPATIALML_LITERT", raising=False)
    monkeypatch.setenv("PATH", "")
    _hide_current_python_litert(tmp_path, monkeypatch)

    with pytest.raises(LiteRTToolError, match="pyspatialml tools litert install"):
        resolve_litert_cli(cache_dir=tmp_path)


def test_install_litert_wraps_pip_failures(tmp_path, monkeypatch):
    venv_python = tmp_path / "litert-cli" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")

    def _run(*_args, **_kwargs):
        raise litert_tools.subprocess.CalledProcessError(
            1,
            "pip install",
            output="stdout text",
            stderr="stderr text",
        )

    monkeypatch.setattr(litert_tools.subprocess, "run", _run)

    with pytest.raises(LiteRTToolError, match="Failed to install litert-cli==0.1.0: stderr text"):
        litert_tools.install_litert_cli(cache_dir=tmp_path)


def test_install_litert_force_recreates_managed_environment(tmp_path, monkeypatch):
    stale_file = tmp_path / "litert-cli" / "stale.txt"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("stale", encoding="utf-8")
    venv_python = tmp_path / "litert-cli" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")
    litert = _write_executable(managed_litert_bin(tmp_path))
    calls = []

    class _EnvBuilder:
        def __init__(self, *, with_pip=False, clear=False):
            calls.append(("builder", with_pip, clear))

        def create(self, venv_dir):
            calls.append(("create", venv_dir))
            venv_python.parent.mkdir(parents=True, exist_ok=True)
            venv_python.write_text("", encoding="utf-8")

    def _run(argv, **_kwargs):
        calls.append(("run", argv))
        _write_executable(litert)

    monkeypatch.setattr(litert_tools.venv, "EnvBuilder", _EnvBuilder)
    monkeypatch.setattr(litert_tools.subprocess, "run", _run)

    resolved = litert_tools.install_litert_cli(cache_dir=tmp_path, force=True)

    assert resolved.path == litert
    assert resolved.managed
    assert not stale_file.exists()
    assert managed_litert_spec_file(tmp_path).read_text(encoding="utf-8") == "litert-cli==0.1.0\n"
    assert ("create", tmp_path / "litert-cli") in calls
    assert any(item[0] == "run" for item in calls)


def test_repair_litert_recreates_managed_environment(tmp_path, monkeypatch):
    calls = []

    def _install_litert_cli(*, cache_dir=None, package="", version="", force=False):
        calls.append((cache_dir, package, version, force))
        return LiteRTCli(path=managed_litert_bin(cache_dir), managed=True)

    monkeypatch.setattr(litert_tools, "install_litert_cli", _install_litert_cli)

    resolved = litert_tools.repair_litert_cli(cache_dir=tmp_path, package="custom-litert", version="1.2.3")

    assert resolved.path == managed_litert_bin(tmp_path)
    assert resolved.managed
    assert calls == [(tmp_path, "custom-litert", "1.2.3", True)]
