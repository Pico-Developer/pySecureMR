# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Binding loader for platform-specific SecureMR native libraries."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path


class BindingsUnavailableError(RuntimeError):
    """Raised when native bindings are unavailable on the current platform."""


def _bindings_root() -> Path:
    return Path(__file__).resolve().parent / "linux"


def bindings_available() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    bindings_path = _bindings_root()
    if not bindings_path.exists():
        return False
    return any(bindings_path.glob("_securemr*.so"))


def require_bindings() -> None:
    if not bindings_available():
        raise BindingsUnavailableError(
            "SecureMR native bindings are not available on this platform."
        )


def load_bindings() -> None:
    if not sys.platform.startswith("linux"):
        raise BindingsUnavailableError(
            "SecureMR native bindings are only supported on Linux."
        )

    bindings_path = _bindings_root()
    if not bindings_path.exists():
        raise BindingsUnavailableError(
            f"Missing native bindings under {bindings_path}."
        )

    libs = [
        "libopencv_core.so.3.4",
        "libopencv_imgproc.so.3.4",
        "libopencv_flann.so.3.4",
        "libopencv_calib3d.so.3.4",
        "libopencv_imgcodecs.so.3.4",
        "libSNPE.so",
        "libopenmr-backend.so",
    ]
    for lib in libs:
        candidate = bindings_path / lib
        if candidate.exists():
            ctypes.CDLL(str(candidate))

    for candidate in bindings_path.glob("_securemr*.so"):
        ctypes.CDLL(str(candidate))


__all__ = ["BindingsUnavailableError", "bindings_available", "require_bindings", "load_bindings"]
