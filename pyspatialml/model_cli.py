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

"""Model-level pySpatialML commands."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

from pyspatialml import litert_runtime


class ModelCliError(RuntimeError):
    """Raised when model inspection cannot be completed."""


def model_info(model: Path, *, signature_index: int = 0, as_json: bool = False) -> int:
    """Print LiteRT/TFLite model input and output metadata."""
    if not model.is_file():
        raise ModelCliError(f"Model file not found: {model}")
    info = inspect_model(model, signature_index=signature_index)
    if as_json:
        print(json.dumps(info, indent=2))
        return 0

    print(f"Model: {info['model']}")
    print(f"Signature: {info['signature_key']}")
    print("Inputs:")
    for tensor in info["inputs"]:
        print(_format_tensor_line(tensor))
    print("Outputs:")
    for tensor in info["outputs"]:
        print(_format_tensor_line(tensor))
    return 0


def inspect_model(model: Path, *, signature_index: int = 0) -> dict[str, Any]:
    """Return LiteRT/TFLite model input and output metadata."""
    try:
        return litert_runtime.inspect_model(model, signature_index=signature_index)
    except litert_runtime.LiteRTRuntimeError as exc:
        raise ModelCliError(f"LiteRT runtime is unavailable: {exc}") from exc


def print_model_error(exc: Exception) -> None:
    """Print a concise model command error."""
    print(f"Error [PSM_MODEL]: {exc}", file=sys.stderr)


def _format_tensor_line(tensor: Mapping[str, Any]) -> str:
    return (
        f"  {tensor['name']}: "
        f"shape={tuple(tensor.get('shape', []))} "
        f"dtype={tensor.get('dtype')} "
        f"index={tensor.get('index')}"
    )
