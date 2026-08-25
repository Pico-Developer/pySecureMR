# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Managed ONNX conversion helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
import venv
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Sequence

import numpy as np

from pyspatialml.litert_tools import default_tool_cache


DEFAULT_ONNX2TF_PACKAGE = "onnx2tf"
DEFAULT_ONNX2TF_VERSION = ""
DEFAULT_ONNX2TF_DEPENDENCIES = (
    "numpy",
    "tensorflow",
    "tf-keras",
    "onnx",
    "onnx-graphsurgeon",
    "onnxruntime",
    "sng4onnx",
    "sne4onnx",
    "psutil",
    "requests",
    "flatbuffers",
    "h5py",
    "ai-edge-litert",
)
_MANAGED_SPEC_FILE = ".pyspatialml-onnx2tf-spec"


class OnnxToolError(RuntimeError):
    """Raised when ONNX conversion tooling cannot be used."""


@dataclass(frozen=True)
class Onnx2TfCli:
    """Resolved onnx2tf executable."""

    path: Path
    managed: bool = False


@dataclass(frozen=True)
class OnnxConversionResult:
    """Result of an ONNX to TFLite conversion."""

    model: Path
    output: Path
    tflite_models: list[Path]
    argv: list[str]
    stdout: str
    stderr: str
    tool: Onnx2TfCli


def managed_onnx2tf_bin(cache_dir: Optional[Path] = None) -> Path:
    """Return the expected managed ``onnx2tf`` executable path."""
    cache = cache_dir or default_tool_cache()
    bin_dir = "Scripts" if os.name == "nt" else "bin"
    exe_name = "onnx2tf.exe" if os.name == "nt" else "onnx2tf"
    return cache / "onnx2tf" / bin_dir / exe_name


def managed_onnx2tf_spec_file(cache_dir: Optional[Path] = None) -> Path:
    """Return the managed onnx2tf install package-spec marker path."""
    cache = cache_dir or default_tool_cache()
    return cache / "onnx2tf" / _MANAGED_SPEC_FILE


def resolve_onnx2tf(
    *,
    ensure: bool = False,
    cache_dir: Optional[Path] = None,
    package: str = DEFAULT_ONNX2TF_PACKAGE,
    version: str = DEFAULT_ONNX2TF_VERSION,
    verbose: bool = False,
) -> Onnx2TfCli:
    """Resolve a usable onnx2tf executable, installing or repairing when needed."""
    cache = cache_dir or default_tool_cache()
    managed_bin = managed_onnx2tf_bin(cache)
    if _is_executable(managed_bin):
        expected_spec = _package_spec(package, version)
        if (
            _managed_spec(cache) == _managed_install_spec(package, version)
            and _is_healthy(managed_bin)
        ):
            _log_verbose(verbose, f"Using managed onnx2tf: {managed_bin}")
            return Onnx2TfCli(path=managed_bin, managed=True)
        if ensure:
            _log_verbose(verbose, f"Repairing managed onnx2tf environment: {cache / 'onnx2tf'}")
            return install_onnx2tf(cache_dir=cache, package=package, version=version, force=True, verbose=verbose)

    if ensure:
        _log_verbose(verbose, f"Installing managed onnx2tf environment: {cache / 'onnx2tf'}")
        return install_onnx2tf(cache_dir=cache, package=package, version=version, verbose=verbose)

    system = shutil.which("onnx2tf")
    if system:
        system_cli = Onnx2TfCli(path=Path(system), managed=False)
        if _is_healthy(system_cli.path):
            _log_verbose(verbose, f"Using onnx2tf from PATH: {system_cli.path}")
            return system_cli

    raise OnnxToolError("onnx2tf is not installed. Run this command again to install the managed converter.")


def install_onnx2tf(
    *,
    cache_dir: Optional[Path] = None,
    package: str = DEFAULT_ONNX2TF_PACKAGE,
    version: str = DEFAULT_ONNX2TF_VERSION,
    force: bool = False,
    verbose: bool = False,
) -> Onnx2TfCli:
    """Install onnx2tf into the pySpatialML managed tool cache."""
    cache = cache_dir or default_tool_cache()
    venv_dir = cache / "onnx2tf"
    if force and venv_dir.exists():
        _log_verbose(verbose, f"Removing existing managed onnx2tf environment: {venv_dir}")
        shutil.rmtree(venv_dir)
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    if not venv_dir.exists():
        try:
            _log_verbose(verbose, f"Creating managed onnx2tf Python environment: {venv_dir}")
            venv.EnvBuilder(with_pip=True, clear=False).create(venv_dir)
        except OSError as exc:
            raise OnnxToolError(f"Failed to create managed onnx2tf environment at {venv_dir}: {exc}") from exc

    python = _venv_python(venv_dir)
    package_spec = _package_spec(package, version)
    install_specs = [package_spec, *DEFAULT_ONNX2TF_DEPENDENCIES]
    _log_verbose(verbose, "Installing/updating onnx2tf and runtime dependencies.")
    _log_verbose(verbose, f"Running: {python} -m pip install --upgrade " + " ".join(install_specs))
    try:
        subprocess.run(
            [str(python), "-m", "pip", "install", "--upgrade", *install_specs],
            check=True,
            text=True,
            stdout=None if verbose else subprocess.PIPE,
            stderr=None if verbose else subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or str(exc)).strip()
        raise OnnxToolError(f"Failed to install {package_spec}: {details}") from exc

    onnx2tf = managed_onnx2tf_bin(cache)
    if not _is_executable(onnx2tf):
        raise OnnxToolError(f"Installed {package_spec}, but no onnx2tf executable was found at {onnx2tf}.")
    if not _is_healthy(onnx2tf):
        raise OnnxToolError(f"Installed {package_spec}, but onnx2tf failed its health check.")
    managed_onnx2tf_spec_file(cache).write_text(_managed_install_spec(package, version) + "\n", encoding="utf-8")
    _log_verbose(verbose, f"Managed onnx2tf ready: {onnx2tf}")
    return Onnx2TfCli(path=onnx2tf, managed=True)


def convert_onnx_to_tflite(
    *,
    model: Path,
    output: Path,
    extra_args: Sequence[str] = (),
    cache_dir: Optional[Path] = None,
    verbose: bool = False,
) -> OnnxConversionResult:
    """Convert an ONNX model to TFLite using onnx2tf."""
    if not model.is_file():
        raise OnnxToolError(f"ONNX model file not found: {model}")
    model = model.expanduser().resolve()
    output = _absolute_output_path(output)
    output_is_file = output.suffix.lower() == ".tflite"
    if output_is_file:
        output.parent.mkdir(parents=True, exist_ok=True)
    else:
        output.mkdir(parents=True, exist_ok=True)
    tool = resolve_onnx2tf(ensure=True, cache_dir=cache_dir, verbose=verbose)
    try:
        with TemporaryDirectory(prefix="pyspatialml-onnx2tf-") as tmp:
            tmp_path = Path(tmp)
            _write_default_onnx2tf_sample_data(tmp_path)
            conversion_output = tmp_path / "output" if output_is_file else output
            conversion_output.mkdir(parents=True, exist_ok=True)
            argv = [str(tool.path), "-i", str(model), "-o", str(conversion_output), *extra_args]
            _log_verbose(verbose, f"Converting ONNX with onnx2tf: {model}")
            _log_verbose(verbose, f"onnx2tf output target: {conversion_output}")
            _log_verbose(verbose, "Running: " + " ".join(argv))
            result = subprocess.run(
                argv,
                check=True,
                text=True,
                stdout=None if verbose else subprocess.PIPE,
                stderr=None if verbose else subprocess.PIPE,
                cwd=tmp,
            )
            stdout = "" if verbose else result.stdout
            stderr = "" if verbose else result.stderr
            tflite_models = sorted(conversion_output.rglob("*.tflite"))
            if output_is_file and tflite_models:
                selected = _select_single_tflite_model(tflite_models, output)
                shutil.copy2(selected, output)
                tflite_models = [output]
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or str(exc)).strip()
        raise OnnxToolError(f"ONNX to TFLite conversion failed: {details}") from exc

    if not tflite_models:
        details = _conversion_output_excerpt(stdout, stderr)
        target_kind = "file" if output_is_file else "directory"
        raise OnnxToolError(
            f"ONNX conversion completed, but no .tflite file was found for output {target_kind} {output}."
            f"{details}"
        )
    return OnnxConversionResult(
        model=model,
        output=output,
        tflite_models=tflite_models,
        argv=argv,
        stdout=stdout,
        stderr=stderr,
        tool=tool,
    )


def _venv_python(venv_dir: Path) -> Path:
    bin_dir = "Scripts" if os.name == "nt" else "bin"
    exe_name = "python.exe" if os.name == "nt" else "python"
    return venv_dir / bin_dir / exe_name


def _write_default_onnx2tf_sample_data(directory: Path) -> None:
    sample = directory / "calibration_image_sample_data_20x128x128x3_float32.npy"
    if sample.is_file():
        return
    data = np.zeros((20, 128, 128, 3), dtype=np.float32)
    np.save(sample, data)


def _absolute_output_path(output: Path) -> Path:
    output = output.expanduser()
    if output.is_absolute():
        return output
    return Path.cwd() / output


def _select_single_tflite_model(models: Sequence[Path], output: Path) -> Path:
    if len(models) == 1:
        return models[0]
    names = ", ".join(str(path.name) for path in models)
    raise OnnxToolError(
        f"ONNX conversion produced multiple .tflite files for file output {output}: {names}. "
        "Use an output directory instead, or pass ONNX conversion flags that produce one TFLite file."
    )


def _conversion_output_excerpt(stdout: str, stderr: str, *, max_chars: int = 4000) -> str:
    parts = []
    if stdout.strip():
        parts.append("stdout:\n" + _tail_text(stdout.strip(), max_chars=max_chars))
    if stderr.strip():
        parts.append("stderr:\n" + _tail_text(stderr.strip(), max_chars=max_chars))
    if not parts:
        return ""
    return "\n\nonnx2tf output:\n" + "\n\n".join(parts)


def _tail_text(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return "... " + value[-max_chars:]


def _log_verbose(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[pyspatialml] {message}", flush=True)


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _is_healthy(path: Path) -> bool:
    if not _is_executable(path):
        return False
    try:
        result = subprocess.run(
            [str(path), "-h"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _package_spec(package: str, version: str) -> str:
    return f"{package}=={version}" if version else package


def _managed_install_spec(package: str, version: str) -> str:
    return "\n".join([_package_spec(package, version), *DEFAULT_ONNX2TF_DEPENDENCIES])


def _managed_spec(cache_dir: Path) -> Optional[str]:
    spec_file = managed_onnx2tf_spec_file(cache_dir)
    if not spec_file.is_file():
        return None
    value = spec_file.read_text(encoding="utf-8").strip()
    return value or None
