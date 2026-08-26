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
from typing import Any, Mapping, Optional, Sequence, Union

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
    "scenegraph_visibility": "XR_SECURE_MR_OPERATOR_TYPE_SCENEGRAPH_VISIBILITY_PICO",
    "update_component": "XR_SECURE_MR_OPERATOR_TYPE_UPDATE_COMPONENT_PICO",
}
_OP_SCENEGRAPH_VISIBILITY = _OP_ALIASES["scenegraph_visibility"]
_OP_UPDATE_COMPONENT = _OP_ALIASES["update_component"]
_XR_ONLY_OPERATORS = {
    "LOAD_TEXTURE",
    "RENDER_TEXT",
    "SWITCH_GLTF_RENDER_STATUS",
    "UPDATE_GLTF",
}
_SPATIAL_ONLY_OPERATORS = {
    "SCENEGRAPH_VISIBILITY",
    "UPDATE_COMPONENT",
}

_OP_ARITY: Mapping[str, tuple[int, Optional[int], int, Optional[int]]] = {
    "UNKNOWN": (1, 1, 1, 1),
    "ARITHMETIC_COMPOSE": (1, None, 1, 1),
    "ELEMENTWISE_MIN": (2, 2, 1, 1),
    "ELEMENTWISE_MAX": (2, 2, 1, 1),
    "ELEMENTWISE_MULTIPLY": (2, 2, 1, 1),
    "CUSTOMIZED_COMPARE": (2, 2, 1, 1),
    "ELEMENTWISE_OR": (2, 2, 1, 1),
    "ELEMENTWISE_AND": (2, 2, 1, 1),
    "ALL": (1, 1, 1, 1),
    "ANY": (1, 1, 1, 1),
    "NMS": (2, 2, 1, 1),
    "SOLVE_P_N_P": (3, 3, 2, 2),
    "GET_AFFINE": (2, 2, 1, 1),
    "APPLY_AFFINE": (2, 2, 1, 1),
    "APPLY_AFFINE_POINT": (2, 2, 1, 1),
    "UV_TO_3D_IN_CAM_SPACE": (5, 5, 1, 1),
    "ASSIGNMENT": (1, 2, 1, 1),
    "RUN_MODEL_INFERENCE": (1, None, 1, None),
    "NORMALIZE": (1, 1, 1, 1),
    "CAMERA_SPACE_TO_WORLD": (1, 1, 2, 2),
    "RECTIFIED_VST_ACCESS": (0, 0, 1, 4),
    "ARGMAX": (1, 1, 1, 1),
    "CONVERT_COLOR": (1, 1, 1, 1),
    "SORT_VEC": (1, 1, 1, 2),
    "INVERSION": (1, 1, 1, 1),
    "GET_TRANSFORM_MAT": (2, 3, 1, 1),
    "SORT_MAT": (1, 1, 1, 2),
    "SWITCH_GLTF_RENDER_STATUS": (1, 4, 0, 0),
    "UPDATE_GLTF": (1, 3, 0, 0),
    "RENDER_TEXT": (1, 5, 0, 0),
    "LOAD_TEXTURE": (2, 2, 1, 1),
    "SVD": (1, 1, 3, 3),
    "NORM": (1, 1, 1, 1),
    "SWAP_HWC_CHW": (1, 1, 1, 1),
    "SCENEGRAPH_VISIBILITY": (1, 2, 0, 0),
    "UPDATE_COMPONENT": (1, 2, 0, 0),
    "JAVASCRIPT": (0, None, 1, None),
    "MICROPHONE": (0, 0, 1, 1),
    "SPEAKER": (1, 1, 1, 1),
    "DEPTH": (0, 0, 1, 1),
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
    if usage_value == _USAGE_ALIASES["gltf"]:
        tensor_spec["tensor_type"] = "gltf"
        tensor_spec["is_gltf"] = True
        tensor_spec["is_placeholder"] = True
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

    normalized_op_type = _operator_type(op_type)
    _validate_operator_arity(normalized_op_type, inputs=inputs, outputs=outputs)
    if normalized_op_type == _OP_ALIASES["arithmetic"] and not expression:
        raise PipelineCliError("Arithmetic operators require --expression")
    _validate_required_operator_metadata(
        normalized_op_type,
        inputs=inputs,
        attrs=attrs,
        flag=flag,
        model=model,
    )

    op = {
        "type": normalized_op_type,
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
    _apply_spatial_operator_fields(op, attrs)

    spec.setdefault("operators", []).append(op)
    _validate_and_write_operator_update(path, spec)
    print(f"Added operator: {op_type}")
    return 0


def remove_op(path: Path, index: int) -> int:
    """Remove an operator from a pipeline by index."""
    spec = _load_pipeline(path)
    operators = spec.get("operators", [])
    if not isinstance(operators, list):
        raise PipelineCliError("Pipeline 'operators' must be a list")
    if index < 0 or index >= len(operators):
        raise PipelineCliError(f"Operator index out of range: {index}")
    removed = operators.pop(index)
    if not isinstance(removed, Mapping):
        removed_type = "<invalid>"
    else:
        removed_type = str(removed.get("type") or removed.get("operator_type") or "<unknown>")
    _validate_and_write_operator_update(path, spec)
    print(f"Removed operator #{index}: {removed_type}")
    return 0


def remove_tensor(path: Path, name: str, *, force: bool = False) -> int:
    """Remove a tensor descriptor from a pipeline."""
    spec = _load_pipeline(path)
    tensors = _tensors(spec)
    if name not in tensors:
        raise PipelineCliError(f"Tensor not found: {name}")

    references = _operator_references_to_tensor(spec, name)
    if references and not force:
        details = ", ".join(f"#{index} {op_type}" for index, op_type in references)
        raise PipelineCliError(
            f"Tensor '{name}' is referenced by operator(s): {details}. "
            "Remove those operators first or rerun with --force."
        )

    del tensors[name]
    for key in ("inputs", "outputs"):
        values = spec.get(key, [])
        if isinstance(values, list):
            spec[key] = [value for value in values if value != name]
    if force and references:
        validate_pipeline_spec(dict(spec))
        _write_json(path, spec)
    else:
        _validate_and_write(path, spec)
    print(f"Removed tensor: {name}")
    if references and force:
        details = ", ".join(f"#{index} {op_type}" for index, op_type in references)
        print(f"Warning: removed tensor was referenced by operator(s): {details}")
    return 0


def _validate_operator_arity(
    op_type: str,
    *,
    inputs: Sequence[str],
    outputs: Sequence[str],
) -> None:
    op_name = _operator_enum_name(op_type)
    arity = _OP_ARITY.get(op_name)
    if arity is None:
        return
    min_inputs, max_inputs, min_outputs, max_outputs = arity
    _validate_count(
        op_name,
        "input",
        len(inputs),
        minimum=min_inputs,
        maximum=max_inputs,
    )
    _validate_count(
        op_name,
        "output",
        len(outputs),
        minimum=min_outputs,
        maximum=max_outputs,
    )


def _validate_count(
    op_name: str,
    label: str,
    count: int,
    *,
    minimum: int,
    maximum: Optional[int],
) -> None:
    if count < minimum or (maximum is not None and count > maximum):
        expected = _format_count_range(minimum, maximum)
        raise PipelineCliError(
            f"{op_name.lower()} operators require {expected} {label} tensor(s); got {count}"
        )


def _format_count_range(minimum: int, maximum: Optional[int]) -> str:
    if maximum is None:
        return f"at least {minimum}"
    if minimum == maximum:
        return f"exactly {minimum}"
    return f"{minimum} to {maximum}"


def _validate_required_operator_metadata(
    op_type: str,
    *,
    inputs: Sequence[str],
    attrs: Sequence[str],
    flag: Optional[str],
    model: Optional[str],
) -> None:
    if op_type.endswith("CONVERT_COLOR_PICO") and flag is None and not attrs:
        raise PipelineCliError("convert_color operators require --flag")
    if op_type.endswith("CUSTOMIZED_COMPARE_PICO") and not attrs:
        raise PipelineCliError("customized_compare operators require --attr with a compare operator")
    if op_type.endswith("JAVASCRIPT_PICO") and not attrs:
        raise PipelineCliError("javascript operators require --attr with JavaScript code")
    if op_type.endswith("RENDER_TEXT_PICO") and len(attrs) < 2:
        raise PipelineCliError(
            "render_text operators require --attr config and --attr text"
        )
    if op_type.endswith("UPDATE_GLTF_PICO") and not attrs:
        raise PipelineCliError("update_gltf operators require --attr with update type")
    if op_type == _OP_UPDATE_COMPONENT and len(inputs) < 2 and not attrs:
        raise PipelineCliError("update_component operators require a second input tensor or --attr enabled/data")
    if op_type == _OP_ALIASES["run_model_inference"] and model is None:
        raise PipelineCliError("run_model_inference operators require --model")


def _apply_spatial_operator_fields(op: dict[str, Any], attrs: Sequence[str]) -> None:
    if op["type"] == _OP_SCENEGRAPH_VISIBILITY:
        op["type"] = "scenegraph_visibility"
        if op["inputs"]:
            op["scenegraph"] = op["inputs"][0]
        if attrs:
            op["visible"] = _parse_bool_or_tensor(attrs[0])
    elif op["type"] == _OP_UPDATE_COMPONENT:
        op["type"] = "update_component"
        if op["inputs"]:
            op["scenegraph"] = op["inputs"][0]
        if attrs:
            value = _parse_bool_or_tensor(attrs[0])
            op["enabled" if isinstance(value, bool) else "data"] = value
    if op["type"] in {"scenegraph_visibility", "update_component"}:
        op.pop("attrs", None)


def _parse_bool_or_tensor(value: str) -> Union[bool, str]:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return value


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
    _write_json_updates([(path, payload)])


def _write_json_updates(updates: Sequence[tuple[Path, Mapping[str, Any]]]) -> None:
    temp_paths: list[Path] = []
    try:
        for path, payload in updates:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, ensure_ascii=False)
                file.write("\n")
            temp_paths.append(tmp)
        for tmp, (path, _payload) in zip(temp_paths, updates):
            tmp.replace(path)
    finally:
        for tmp in temp_paths:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass


def _validate_and_write_operator_update(path: Path, spec: Mapping[str, Any]) -> None:
    _validate_pipeline_update(spec)
    manifest_update = _prepare_manifest_modes_for_pipeline(path, spec)
    updates: list[tuple[Path, Mapping[str, Any]]] = [(path, spec)]
    if manifest_update is not None:
        updates.append(manifest_update)
    _write_json_updates(updates)


def _validate_pipeline_update(spec: Mapping[str, Any]) -> None:
    validate_pipeline_spec(dict(spec))
    _validate_tensor_references(spec)


def _validate_and_write(path: Path, spec: Mapping[str, Any]) -> None:
    _validate_pipeline_update(spec)
    _write_json(path, spec)


def _prepare_manifest_modes_for_pipeline(
    path: Path,
    spec: Mapping[str, Any],
) -> Optional[tuple[Path, Mapping[str, Any]]]:
    manifest_context = _find_manifest_context(path)
    if manifest_context is None:
        classification = _classify_mode_specific_operators(spec.get("operators", []))
        _raise_if_modes_conflict(classification)
        return None

    manifest_path, manifest = manifest_context
    runtime = manifest.setdefault("runtime", {})
    if not isinstance(runtime, dict):
        raise PipelineCliError("Package manifest 'runtime' must be an object")
    supported_modes = runtime.get("supported_modes", [])
    if supported_modes and not isinstance(supported_modes, list):
        raise PipelineCliError("Package manifest runtime.supported_modes must be a list")

    pipeline_specs = _load_manifest_pipeline_specs_with_candidate(
        manifest,
        manifest_path.parent,
        path,
        spec,
    )
    classification = _classify_mode_specific_pipeline_specs(pipeline_specs)
    _raise_if_modes_conflict(classification)

    current_modes = {str(mode).strip().lower() for mode in supported_modes}
    if classification["xr"]:
        if current_modes and "xr" not in current_modes:
            raise PipelineCliError(
                "Cannot add XR-only operator to pipeline referenced by a manifest "
                "whose runtime.supported_modes only includes spatial."
            )
        desired_modes = ["xr"]
    elif classification["spatial"]:
        if current_modes and "spatial" not in current_modes:
            raise PipelineCliError(
                "Cannot add Spatial-only operator to pipeline referenced by a manifest "
                "whose runtime.supported_modes only includes xr."
            )
        desired_modes = ["spatial"]
    elif current_modes:
        desired_modes = ["xr", "spatial"]
    else:
        return None
    if list(supported_modes) == desired_modes:
        return None
    runtime["supported_modes"] = desired_modes
    return manifest_path, manifest


def _load_manifest_pipeline_specs_with_candidate(
    manifest: Mapping[str, Any],
    root: Path,
    candidate_path: Path,
    candidate_spec: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any]]]:
    pipelines = manifest.get("pipelines", [])
    if not isinstance(pipelines, list):
        raise PipelineCliError("Package manifest 'pipelines' must be a list")

    resolved_candidate_path = candidate_path.resolve()
    pipeline_specs: list[tuple[str, Mapping[str, Any]]] = []
    matched_candidate = False
    for index, item in enumerate(pipelines):
        if not isinstance(item, Mapping):
            raise PipelineCliError("Package manifest pipeline entries must be objects")
        rel_path = item.get("path")
        if not isinstance(rel_path, str) or not rel_path:
            raise PipelineCliError("Package manifest pipeline entries require a non-empty path")
        pipeline_id = str(item.get("id") or rel_path or index)
        pipeline_path = _resolve_manifest_pipeline_path(root, rel_path)
        if pipeline_path == resolved_candidate_path:
            pipeline_specs.append((pipeline_id, candidate_spec))
            matched_candidate = True
            continue
        pipeline_specs.append((pipeline_id, _load_pipeline(pipeline_path)))
    if not matched_candidate:
        raise PipelineCliError("Package manifest no longer references the edited pipeline")
    return pipeline_specs


def _resolve_manifest_pipeline_path(root: Path, path: str) -> Path:
    raw_path = Path(path)
    normalized = path.replace("\\", "/").strip("/")
    if raw_path.is_absolute() or path.replace("\\", "/").startswith("/") or not normalized:
        raise PipelineCliError(f"Package manifest pipeline path must be package-relative: {path}")
    parts = normalized.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise PipelineCliError(f"Invalid package manifest pipeline path: {path}")
    resolved_root = root.resolve()
    resolved_path = (root / normalized).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise PipelineCliError(f"Package manifest pipeline path escapes package root: {path}") from exc
    return resolved_path


def _classify_mode_specific_pipeline_specs(
    pipeline_specs: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[str, set[str]]:
    xr_only = []
    spatial_only = []
    for pipeline_id, spec in pipeline_specs:
        classification = _classify_mode_specific_operators(spec.get("operators", []))
        xr_only.extend(f"{pipeline_id}:{op_type}" for op_type in sorted(classification["xr"]))
        spatial_only.extend(f"{pipeline_id}:{op_type}" for op_type in sorted(classification["spatial"]))
    return {"xr": set(xr_only), "spatial": set(spatial_only)}


def _raise_if_modes_conflict(classification: Mapping[str, set[str]]) -> None:
    if classification["xr"] and classification["spatial"]:
        raise PipelineCliError(
            "Cannot update pipeline because it would mix XR-only and Spatial-only operators. "
            f"XR-only: {', '.join(sorted(classification['xr']))}. "
            f"Spatial-only: {', '.join(sorted(classification['spatial']))}."
        )


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


def _operator_references_to_tensor(spec: Mapping[str, Any], tensor_name: str) -> list[tuple[int, str]]:
    references = []
    for index, op in enumerate(spec.get("operators", [])):
        if not isinstance(op, Mapping):
            continue
        for key in ("inputs", "outputs"):
            for ref in op.get(key, []):
                if _resolve_ref_name(ref) == tensor_name:
                    references.append((index, str(op.get("type") or op.get("operator_type") or "<unknown>")))
                    break
            else:
                continue
            break
    return references


def _classify_mode_specific_operators(operators: Sequence[Any]) -> dict[str, set[str]]:
    xr_only = []
    spatial_only = []
    for op in operators:
        if not isinstance(op, Mapping):
            continue
        op_type = _operator_enum_name(str(op.get("type") or op.get("operator_type") or ""))
        if op_type in _XR_ONLY_OPERATORS:
            xr_only.append(op_type)
        if op_type in _SPATIAL_ONLY_OPERATORS:
            spatial_only.append(op_type)
    return {"xr": set(xr_only), "spatial": set(spatial_only)}


def _find_manifest_context(path: Path) -> Optional[tuple[Path, dict[str, Any]]]:
    pipeline_path = path.resolve()
    for manifest_path in _candidate_manifest_paths(path):
        if not manifest_path.is_file():
            continue
        try:
            manifest = _read_json_file(manifest_path)
        except PipelineCliError:
            continue
        if not _manifest_references_pipeline(manifest, manifest_path.parent, pipeline_path):
            continue
        return manifest_path, manifest
    return None


def _candidate_manifest_paths(path: Path) -> list[Path]:
    candidates = []
    for directory in [path.parent, *path.parents]:
        candidate = directory / "manifest.json"
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _manifest_references_pipeline(manifest: Mapping[str, Any], root: Path, pipeline_path: Path) -> bool:
    pipelines = manifest.get("pipelines", [])
    if not isinstance(pipelines, list):
        return False
    for item in pipelines:
        if not isinstance(item, Mapping):
            continue
        rel_path = item.get("path")
        if not isinstance(rel_path, str) or not rel_path:
            continue
        try:
            manifest_pipeline = _resolve_manifest_pipeline_path(root, rel_path)
        except (OSError, PipelineCliError):
            continue
        if manifest_pipeline == pipeline_path:
            return True
    return False


def _read_json_file(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise PipelineCliError(f"JSON file must contain an object: {path}")
    return payload


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


def _operator_enum_name(op_type: str) -> str:
    value = str(op_type).strip()
    if value.startswith("XR_SECURE_MR_OPERATOR_TYPE_"):
        value = value[len("XR_SECURE_MR_OPERATOR_TYPE_"):]
    if value.endswith("_PICO"):
        value = value[: -len("_PICO")]
    return value.upper()


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
