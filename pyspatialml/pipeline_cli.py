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

"""File-based SpatialML pipeline builder commands."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from securemr.core.types import EDataType
from securemr.core.utils import mat_flag
from securemr.py2smr import convert
from securemr.py2smr.verifier import validate_pipeline_spec


class PipelineCliError(RuntimeError):
    """Raised when a pipeline CLI operation cannot be completed."""


_DTYPE_ALIASES: Mapping[str, Any] = {
    "uint8": np.uint8,
    "u8": np.uint8,
    "int8": np.int8,
    "i8": np.int8,
    "uint16": np.uint16,
    "u16": np.uint16,
    "int16": np.int16,
    "i16": np.int16,
    "int32": np.int32,
    "i32": np.int32,
    "float32": np.float32,
    "fp32": np.float32,
    "float64": np.float64,
    "fp64": np.float64,
}

_DTYPE_TO_SCHEMA = {
    np.dtype(np.uint8).type: int(EDataType.UINT8),
    np.dtype(np.int8).type: int(EDataType.INT8),
    np.dtype(np.uint16).type: int(EDataType.UINT16),
    np.dtype(np.int16).type: int(EDataType.INT16),
    np.dtype(np.int32).type: int(EDataType.INT32),
    np.dtype(np.float32).type: int(EDataType.FLOAT32),
    np.dtype(np.float64).type: int(EDataType.FLOAT64),
}

_USAGE_ALIASES = {
    "mat": 6,
    "matrix": 6,
    "scalar": 2,
    "point": 1,
    "slice": 3,
    "color": 4,
    "timestamp": 5,
    "gltf": 7,
}

_OP_ALIASES = {
    "arithmetic": "XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO",
    "assignment": "XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO",
    "run_model_inference": "XR_SECURE_MR_OPERATOR_TYPE_RUN_MODEL_INFERENCE_PICO",
}


def init_pipeline(path: Path, *, force: bool = False) -> int:
    """Create an empty pipeline JSON file."""
    if path.exists() and not force:
        raise PipelineCliError(f"Pipeline already exists: {path}")
    spec = {"tensors": {}, "operators": [], "inputs": [], "outputs": []}
    _write_json(path, spec)
    print(f"Initialized pipeline: {path}")
    return 0


def add_tensor(
    path: Path,
    name: str,
    *,
    shape: str,
    dtype: str,
    usage: str = "matrix",
    is_input: bool = False,
    is_output: bool = False,
    value: Optional[str] = None,
) -> int:
    """Add or replace a tensor descriptor."""
    spec = _load_pipeline(path)
    tensors = _tensors(spec)
    if name in tensors:
        raise PipelineCliError(f"Tensor already exists: {name}")

    dimensions, channels = _shape_to_dimensions_and_channels(_parse_int_list(shape))
    data_type = _schema_dtype(dtype)
    usage_value = _usage_value(usage)
    tensor_spec: dict[str, Any] = {
        "dimensions": dimensions,
        "channels": channels,
        "data_type": data_type,
        "is_placeholder": bool(is_input or is_output),
        "usage": usage_value,
    }
    if usage_value == _USAGE_ALIASES["matrix"]:
        tensor_spec["flag"] = mat_flag(EDataType(data_type), channels)
    if value is not None:
        tensor_spec["value"] = _parse_value_list(value)

    tensors[name] = tensor_spec
    if is_input:
        _append_unique(spec.setdefault("inputs", []), name)
    if is_output:
        _append_unique(spec.setdefault("outputs", []), name)
    _validate_and_write(path, spec)
    print(f"Added tensor: {name}")
    return 0


def add_op(
    path: Path,
    op_type: str,
    *,
    inputs: Sequence[str],
    outputs: Sequence[str],
    attrs: Sequence[str] = (),
    expression: Optional[str] = None,
    dtype: Optional[str] = None,
    flag: Optional[str] = None,
    threshold: Optional[float] = None,
    model: Optional[str] = None,
    model_name: Optional[str] = None,
    model_target: str = "npu",
    cpu_target_num_threads: int = 1,
) -> int:
    """Append an operator to a pipeline."""
    spec = _load_pipeline(path)
    tensors = _tensors(spec)
    for tensor_name in list(inputs) + list(outputs):
        if tensor_name and tensor_name not in tensors:
            raise PipelineCliError(f"Unknown tensor referenced by operator: {tensor_name}")

    op = {
        "type": _operator_type(op_type),
        "inputs": list(inputs),
        "outputs": list(outputs),
    }
    if attrs:
        op["attrs"] = list(attrs)
    if expression is not None:
        op["expression"] = expression
    if dtype is not None:
        op["data_type"] = _schema_dtype(dtype)
    if flag is not None:
        op["flag"] = _parse_int(flag)
    if threshold is not None:
        op["threshold"] = float(threshold)
    if model is not None:
        if not model_name:
            model_name = Path(model).stem or "model"
        op["model_type"] = "tflite"
        op["model_target"] = model_target
        op["cpu_target_num_threads"] = int(cpu_target_num_threads)
        op["model_name"] = model_name
        op["model"] = {
            "bin_path": _normalize_model_path(model),
            "model_name": model_name,
            "model_type": "tflite",
            "model_target": model_target,
            "cpu_target_num_threads": int(cpu_target_num_threads),
        }

    spec.setdefault("operators", []).append(op)
    _validate_and_write(path, spec)
    print(f"Added operator: {op_type}")
    return 0


def set_input(path: Path, names: Sequence[str]) -> int:
    """Set top-level pipeline inputs."""
    return _set_boundary(path, "inputs", names)


def set_output(path: Path, names: Sequence[str]) -> int:
    """Set top-level pipeline outputs."""
    return _set_boundary(path, "outputs", names)


def validate_pipeline(path: Path) -> int:
    """Validate a pipeline JSON file."""
    spec = _load_pipeline(path)
    validate_pipeline_spec(spec)
    _validate_tensor_references(spec)
    print(f"Pipeline is valid: {path}")
    return 0


def inspect_pipeline(path: Path) -> int:
    """Print a compact pipeline summary."""
    spec = _load_pipeline(path)
    tensors = spec.get("tensors", {})
    operators = spec.get("operators", [])
    print(f"Pipeline: {path}")
    print(f"Tensors: {len(tensors)}")
    print(f"Operators: {len(operators)}")
    print(f"Inputs: {', '.join(spec.get('inputs', [])) or '-'}")
    print(f"Outputs: {', '.join(spec.get('outputs', [])) or '-'}")
    for index, op in enumerate(operators):
        print(
            f"  [{index}] {op.get('type', '<unknown>')} "
            f"{op.get('inputs', [])} -> {op.get('outputs', [])}"
        )
    return 0


def trace_pipeline(
    source: Path,
    *,
    function_name: str,
    output: Path,
    inputs: Sequence[str],
) -> int:
    """Import a Python file, run a traceable function, and write pipeline JSON."""
    module = _load_module(source)
    target = getattr(module, function_name, None)
    if target is None:
        raise PipelineCliError(f"Function not found: {function_name}")
    trace_fn = getattr(target, "trace", None)
    if trace_fn is None:
        raise PipelineCliError(f"Function is not traceable with @trace: {function_name}")

    input_values = dict(_parse_trace_input(item) for item in inputs)
    _result, ctx = trace_fn(**input_values)
    convert(ctx, output=str(output))
    print(f"Wrote traced pipeline: {output}")
    return 0


def print_pipeline_error(exc: Exception) -> None:
    """Print a concise pipeline command error."""
    print(f"Error [PSM_PIPELINE]: {exc}", file=sys.stderr)


def _set_boundary(path: Path, key: str, names: Sequence[str]) -> int:
    spec = _load_pipeline(path)
    tensors = _tensors(spec)
    missing = [name for name in names if name not in tensors]
    if missing:
        raise PipelineCliError(f"Unknown tensor(s): {', '.join(missing)}")
    spec[key] = list(dict.fromkeys(names))
    for name in names:
        tensors[name]["is_placeholder"] = True
    _validate_and_write(path, spec)
    print(f"Set {key}: {', '.join(names) or '-'}")
    return 0


def _load_pipeline(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PipelineCliError(f"Pipeline not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise PipelineCliError("Pipeline JSON must be an object")
    data.setdefault("tensors", {})
    data.setdefault("operators", [])
    data.setdefault("inputs", [])
    data.setdefault("outputs", [])
    return data


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
    tmp.replace(path)


def _validate_and_write(path: Path, spec: Mapping[str, Any]) -> None:
    validate_pipeline_spec(dict(spec))
    _validate_tensor_references(spec)
    _write_json(path, spec)


def _validate_tensor_references(spec: Mapping[str, Any]) -> None:
    tensors = spec.get("tensors", {})
    if not isinstance(tensors, Mapping):
        raise PipelineCliError("Pipeline 'tensors' must be an object")
    for key in ("inputs", "outputs"):
        refs = spec.get(key, [])
        if not isinstance(refs, list):
            raise PipelineCliError(f"Pipeline '{key}' must be a list")
        missing = [name for name in refs if name not in tensors]
        if missing:
            raise PipelineCliError(f"Unknown {key} tensor(s): {', '.join(missing)}")
    for index, op in enumerate(spec.get("operators", [])):
        if not isinstance(op, Mapping):
            raise PipelineCliError(f"Operator #{index} must be an object")
        for key in ("inputs", "outputs"):
            for ref in op.get(key, []):
                tensor_name = _resolve_ref_name(ref)
                if tensor_name and tensor_name not in tensors:
                    raise PipelineCliError(
                        f"Operator #{index} references unknown tensor '{tensor_name}'"
                    )


def _tensors(spec: dict[str, Any]) -> dict[str, Any]:
    tensors = spec.setdefault("tensors", {})
    if not isinstance(tensors, dict):
        raise PipelineCliError("Pipeline 'tensors' must be an object")
    return tensors


def _resolve_ref_name(ref: Any) -> Optional[str]:
    if isinstance(ref, str):
        return ref
    if isinstance(ref, Mapping):
        return ref.get("tensor") or ref.get("name")
    return None


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _parse_int_list(value: str) -> list[int]:
    if not value:
        raise PipelineCliError("Shape cannot be empty")
    try:
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise PipelineCliError(f"Invalid integer list: {value}") from exc


def _shape_to_dimensions_and_channels(shape: Sequence[int]) -> tuple[list[int], int]:
    if not shape:
        raise PipelineCliError("Shape cannot be empty")
    if any(int(dim) <= 0 for dim in shape):
        raise PipelineCliError(f"Shape dimensions must be positive: {shape}")
    dims = [int(dim) for dim in shape]
    if len(dims) == 1:
        return [dims[0], 1], 1
    if len(dims) == 2:
        return dims, 1
    if len(dims) == 3:
        return dims[:2], dims[2]
    return [int(np.prod(dims[1:])), dims[0]], 1


def _schema_dtype(dtype: str) -> int:
    key = dtype.strip().lower()
    np_type = _DTYPE_ALIASES.get(key)
    if np_type is None:
        raise PipelineCliError(f"Unsupported dtype: {dtype}")
    return _DTYPE_TO_SCHEMA[np.dtype(np_type).type]


def _usage_value(usage: str) -> int:
    key = usage.strip().lower().replace("-", "_")
    if key in _USAGE_ALIASES:
        return _USAGE_ALIASES[key]
    return _parse_int(usage)


def _parse_int(value: str) -> int:
    try:
        return int(str(value), 0)
    except ValueError as exc:
        raise PipelineCliError(f"Invalid integer value: {value}") from exc


def _parse_value_list(value: str) -> list[Any]:
    if not value:
        return []
    result: list[Any] = []
    for part in value.split(","):
        text = part.strip()
        if not text:
            continue
        try:
            if "." in text or "e" in text.lower():
                result.append(float(text))
            else:
                result.append(int(text, 0))
        except ValueError:
            result.append(text)
    return result


def _operator_type(op_type: str) -> str:
    key = op_type.strip()
    alias = _OP_ALIASES.get(key.lower())
    if alias:
        return alias
    if key.startswith("XR_SECURE_MR_OPERATOR_TYPE_"):
        return key
    return f"XR_SECURE_MR_OPERATOR_TYPE_{key.upper()}_PICO"


def _normalize_model_path(model: str) -> str:
    if not model.endswith(".tflite"):
        raise PipelineCliError("Model path must point to a .tflite file")
    normalized = model.replace("\\", "/").lstrip("/")
    if normalized.startswith("../") or "/../" in normalized:
        raise PipelineCliError(f"Invalid package-relative model path: {model}")
    return normalized


def _load_module(source: Path):
    if not source.is_file():
        raise PipelineCliError(f"Python source file not found: {source}")
    module_name = f"_pyspatialml_trace_{source.stem}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise PipelineCliError(f"Unable to import source file: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(source.parent.resolve()))
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(source.parent.resolve()))
        except ValueError:
            pass
    return module


def _parse_trace_input(value: str) -> tuple[str, np.ndarray]:
    if "=" not in value:
        raise PipelineCliError("Trace input must use name=path format")
    name, raw_path = value.split("=", 1)
    if not name:
        raise PipelineCliError("Trace input name cannot be empty")
    path = Path(raw_path)
    if not path.is_file():
        raise PipelineCliError(f"Trace input file not found: {path}")
    return name, _load_input_array(path)


def _load_input_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        data = np.load(path)
        if not data.files:
            raise PipelineCliError(f"NPZ input has no arrays: {path}")
        return data[data.files[0]]
    if suffix in {".png", ".jpg", ".jpeg"}:
        try:
            import cv2
        except ImportError as exc:
            raise PipelineCliError("Image trace inputs require opencv-python") from exc
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise PipelineCliError(f"Failed to read image input: {path}")
        return image
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        return np.asarray(payload)
    if suffix == ".bin":
        return np.fromfile(path, dtype=np.float32)
    raise PipelineCliError(f"Unsupported trace input file type: {path.suffix}")


def copy_pipeline(source: Path, destination: Path) -> None:
    """Copy a pipeline JSON file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
