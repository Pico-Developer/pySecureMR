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

"""LiteRT CLI detection and managed installation helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import venv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence


DEFAULT_LITERT_PACKAGE = "litert-cli"
DEFAULT_LITERT_VERSION = "0.1.0"
_CACHE_ENV = "PYSPATIALML_TOOL_CACHE"
_LITERT_ENV = "PYSPATIALML_LITERT"
_MANAGED_SPEC_FILE = ".pyspatialml-litert-spec"


class LiteRTToolError(RuntimeError):
    """Raised when the LiteRT CLI cannot be resolved or installed."""


@dataclass(frozen=True)
class LiteRTCli:
    """Resolved LiteRT CLI executable."""

    path: Path
    managed: bool = False

    def run(self, args: Sequence[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
        """Run ``litert`` with the provided arguments."""
        return subprocess.run(
            [str(self.path), *args],
            check=check,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def version(self) -> str:
        """Return a best-effort LiteRT CLI version string."""
        for args in (["--version"], ["version"], ["--help"]):
            result = self.run(args)
            output = (result.stdout or result.stderr).strip()
            if result.returncode == 0 and output:
                return output.splitlines()[0]
        return "unknown"


def default_tool_cache() -> Path:
    """Return the default pySpatialML managed tool cache directory."""
    configured = os.getenv(_CACHE_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "pyspatialml" / "tools"


def managed_litert_bin(cache_dir: Optional[Path] = None) -> Path:
    """Return the expected managed ``litert`` executable path."""
    cache = cache_dir or default_tool_cache()
    bin_dir = "Scripts" if os.name == "nt" else "bin"
    exe_name = "litert.exe" if os.name == "nt" else "litert"
    return cache / "litert-cli" / bin_dir / exe_name


def litert_python_for_cli(cli: LiteRTCli) -> Path:
    """Return the Python executable associated with a resolved LiteRT CLI."""
    python = _venv_python(cli.path.parent.parent)
    return python if python.is_file() else Path(sys.executable)


def managed_litert_spec_file(cache_dir: Optional[Path] = None) -> Path:
    """Return the managed install package-spec marker path."""
    cache = cache_dir or default_tool_cache()
    return cache / "litert-cli" / _MANAGED_SPEC_FILE


def resolve_litert_cli(
    *,
    ensure: bool = False,
    cache_dir: Optional[Path] = None,
    package: str = DEFAULT_LITERT_PACKAGE,
    version: str = DEFAULT_LITERT_VERSION,
    env: Optional[Mapping[str, str]] = None,
) -> LiteRTCli:
    """Resolve a usable LiteRT CLI.

    Resolution order:
    1. ``PYSPATIALML_LITERT`` when set.
    2. ``litert`` on ``PATH``.
    3. pySpatialML managed cache.
    4. Managed install when ``ensure`` is true.
    """
    active_env = env or os.environ

    configured = active_env.get(_LITERT_ENV)
    if configured:
        configured_path = Path(configured).expanduser()
        if _is_executable(configured_path):
            return LiteRTCli(path=configured_path, managed=False)
        raise LiteRTToolError(f"{_LITERT_ENV} points to a non-executable file: {configured_path}")

    local_litert = _local_litert_bin()
    if _is_executable(local_litert):
        return LiteRTCli(path=local_litert, managed=False)

    system_litert = shutil.which("litert")
    if system_litert:
        system_path = Path(system_litert)
        managed_path = managed_litert_bin(cache_dir)
        if system_path.resolve() != managed_path.resolve():
            return LiteRTCli(path=system_path, managed=False)

    cache = cache_dir or default_tool_cache()
    managed_bin = managed_litert_bin(cache)
    if _is_executable(managed_bin):
        if ensure and _managed_spec(cache) != _package_spec(package, version):
            return install_litert_cli(cache_dir=cache, package=package, version=version)
        return LiteRTCli(path=managed_bin, managed=True)

    if ensure:
        return install_litert_cli(cache_dir=cache, package=package, version=version)

    raise LiteRTToolError(
        "LiteRT CLI is not installed. Install it with "
        f"`python -m pip install {package}=={version}` or `pyspatialml tools litert install`."
    )


def install_litert_cli(
    *,
    cache_dir: Optional[Path] = None,
    package: str = DEFAULT_LITERT_PACKAGE,
    version: str = DEFAULT_LITERT_VERSION,
) -> LiteRTCli:
    """Install LiteRT CLI into the pySpatialML managed tool cache."""
    cache = cache_dir or default_tool_cache()
    venv_dir = cache / "litert-cli"
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    if not venv_dir.exists():
        venv.EnvBuilder(with_pip=True, clear=False).create(venv_dir)

    python = _venv_python(venv_dir)
    package_spec = _package_spec(package, version)
    try:
        subprocess.run(
            [str(python), "-m", "pip", "install", "--upgrade", package_spec],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or str(exc)).strip()
        raise LiteRTToolError(f"Failed to install {package_spec}: {details}") from exc

    litert = managed_litert_bin(cache)
    if not _is_executable(litert):
        raise LiteRTToolError(
            f"Installed {package_spec}, but no litert executable was found at {litert}."
        )
    managed_litert_spec_file(cache).write_text(package_spec + "\n", encoding="utf-8")
    return LiteRTCli(path=litert, managed=True)


def _venv_python(venv_dir: Path) -> Path:
    bin_dir = "Scripts" if os.name == "nt" else "bin"
    exe_name = "python.exe" if os.name == "nt" else "python"
    return venv_dir / bin_dir / exe_name


def _local_litert_bin() -> Path:
    exe_name = "litert.exe" if os.name == "nt" else "litert"
    return Path(sys.executable).parent / exe_name


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _package_spec(package: str, version: str) -> str:
    return f"{package}=={version}" if version else package


def _managed_spec(cache_dir: Path) -> Optional[str]:
    spec_file = managed_litert_spec_file(cache_dir)
    if not spec_file.is_file():
        return None
    value = spec_file.read_text(encoding="utf-8").strip()
    return value or None


def print_litert_error(exc: Exception) -> None:
    """Print a concise LiteRT setup error for CLI users."""
    print(f"Error [PSM_LITERT_UNAVAILABLE]: {exc}", file=sys.stderr)
