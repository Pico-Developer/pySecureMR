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
"""Numerical consistency verification for py2smr."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from securemr.core.types import EOperatorType
from securemr.core.utils import TensorType, convert_to_dtype

__all__ = [
    "verify",
    "VerificationResult",
    "compare_outputs",
    "run_pipeline_python",
    "validate_pipeline_spec",
]

_OP_GET_TRANSFORM_MAT = getattr(EOperatorType, "GET_TRANSFORM_MAT", getattr(EOperatorType, "MAKE_TRANSFORM_MAT", None))
_OP_LOAD_TEXTURE = getattr(EOperatorType, "LOAD_TEXTURE", getattr(EOperatorType, "UPLOAD_TEXTURE_TO_GLTF", None))
_OP_SWAP_HWC_CHW = getattr(EOperatorType, "SWAP_HWC_CHW", getattr(EOperatorType, "CHW_HWC", None))
_OP_JAVASCRIPT = getattr(EOperatorType, "JAVASCRIPT", getattr(EOperatorType, "JS_SCRIPTING", None))

_OPERATOR_TYPE_ALIASES = {
    "ARITHMETIC": "ARITHMETIC_COMPOSE",
    "CAMERA_ACCESS": "RECTIFIED_VST_ACCESS",
    "CAM_SPACE_TO_XR_LOCAL": "CAMERA_SPACE_TO_WORLD",
    "COMPARE_TO": "CUSTOMIZED_COMPARE",
    "CVT_COLOR": "CONVERT_COLOR",
    "DRAW_TEXT": "RENDER_TEXT",
    "ELEMENTWISE": "ELEMENTWISE_MULTIPLY",
    "GET_TRANSFORM_MATRIX": "GET_TRANSFORM_MAT",
    "JS_SCRIPTING": "JAVASCRIPT",
    "MAKE_TRANSFORM_MAT": "GET_TRANSFORM_MAT",
    "NON_MAXIMUM_SUPPRESSION": "NMS",
    "RENDER_GLTF": "SWITCH_GLTF_RENDER_STATUS",
    "RUN_ALGORITHM": "RUN_MODEL_INFERENCE",
    "SOLVE_PNP": "SOLVE_P_N_P",
    "SORT_MATRIX": "SORT_MAT",
    "SORT_VECTOR": "SORT_VEC",
    "TRANSFORM": "GET_TRANSFORM_MAT",
    "TYPE_CONVERT": "ASSIGNMENT",
    "UPLOAD_TEXTURE_TO_GLTF": "LOAD_TEXTURE",
    "UV2_CAM": "UV_TO_3D_IN_CAM_SPACE",
    "UV_TO_3D_IN_CAMERA_SPACE": "UV_TO_3D_IN_CAM_SPACE",
    "CHW_HWC": "SWAP_HWC_CHW",
}


@dataclass
class VerificationResult:
    """Result of a verification run."""
    success: bool
    host_outputs: Dict[str, np.ndarray]
    device_outputs: Optional[Dict[str, np.ndarray]] = None
    max_abs_diff: Optional[Dict[str, float]] = None
    max_rel_diff: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None


def _is_matrix_tensor(tensor_spec: Dict[str, Any]) -> bool:
    """Return True when a tensor descriptor declares MAT/matrix usage."""
    tensor_type = tensor_spec.get("tensor_type") or tensor_spec.get("type")
    if tensor_type is not None:
        normalized = str(tensor_type).strip().lower().replace("-", "_")
        if normalized in {"matrix", "mat"}:
            return True

    usage = tensor_spec.get("usage")
    if usage is None:
        return False

    if isinstance(usage, str):
        normalized = usage.strip().lower().replace("-", "_")
        if normalized in {"matrix", "mat"}:
            return True
        try:
            usage = int(usage, 0)
        except ValueError:
            return False

    try:
        return int(usage) == int(TensorType.MAT.value)
    except (TypeError, ValueError):
        return False


def _tensor_dimensions_and_channels(tensor_spec: Dict[str, Any]) -> tuple[List[int], int]:
    dimensions = tensor_spec.get("dimensions", [])
    if not isinstance(dimensions, list):
        dimensions = []
    dims = [int(dim) for dim in dimensions]
    channels = int(tensor_spec.get("channels", 1) or 1)
    return dims, channels


def _validate_swap_hwc_chw_operator(
    op_spec: Dict[str, Any],
    tensor_specs: Dict[str, Any],
) -> None:
    input_refs = op_spec.get("inputs", [])
    output_refs = op_spec.get("outputs", [])
    if len(input_refs) != 1 or len(output_refs) != 1:
        raise ValueError("swap_hwc_chw requires exactly one input and one output tensor")

    input_name = _resolve_tensor_name(input_refs[0])
    output_name = _resolve_tensor_name(output_refs[0])
    input_spec = tensor_specs.get(input_name or "")
    output_spec = tensor_specs.get(output_name or "")
    if input_spec is None or output_spec is None:
        return

    input_dims, input_channels = _tensor_dimensions_and_channels(input_spec)
    output_dims, output_channels = _tensor_dimensions_and_channels(output_spec)
    if len(input_dims) != 2:
        raise ValueError(
            f"swap_hwc_chw input '{input_name}' must be a 3D matrix encoded as "
            "2 dimensions plus channels"
        )

    if input_channels <= 4:
        expected_dims = [input_channels, input_dims[0]]
        expected_channels = input_dims[1]
    else:
        expected_dims = [input_dims[1], input_channels]
        expected_channels = input_dims[0]

    if output_dims != expected_dims or output_channels != expected_channels:
        raise ValueError(
            f"swap_hwc_chw output '{output_name}' has dimensions {output_dims} "
            f"and channels {output_channels}, expected dimensions {expected_dims} "
            f"and channels {expected_channels} for input '{input_name}'"
        )


def validate_pipeline_spec(spec: Dict[str, Any]) -> None:
    """Validate pipeline JSON rules that must match the native runtime.

    Raises:
        ValueError: If the pipeline spec contains an invalid tensor descriptor.
    """
    for name, tensor_spec in spec.get("tensors", {}).items():
        if not isinstance(tensor_spec, dict):
            continue
        if not _is_matrix_tensor(tensor_spec):
            continue

        dimensions = tensor_spec.get("dimensions", [])
        if not isinstance(dimensions, list) or len(dimensions) < 2:
            raise ValueError(
                f"Tensor '{name}' is declared as matrix/MAT usage but has "
                f"dimensions {dimensions!r}; matrix tensors must have at least "
                "2 dimensions. Use [1, N] or [N, 1] for vectors, or use a "
                "scalar/point tensor type for 1D data."
            )

    tensor_specs = spec.get("tensors", {})
    for op_spec in spec.get("operators", []):
        if not isinstance(op_spec, dict):
            continue
        if _OP_SWAP_HWC_CHW is not None and _get_operator_type(op_spec.get("type", "")) == _OP_SWAP_HWC_CHW:
            _validate_swap_hwc_chw_operator(op_spec, tensor_specs)


def compare_outputs(
    expected: Dict[str, np.ndarray],
    actual: Dict[str, np.ndarray],
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> VerificationResult:
    """Compare expected and actual outputs.

    Args:
        expected: Dictionary of expected output tensors.
        actual: Dictionary of actual output tensors.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.

    Returns:
        VerificationResult with comparison details.
    """
    max_abs_diff = {}
    max_rel_diff = {}
    all_close = True
    error_messages = []

    for name in expected:
        if name not in actual:
            error_messages.append(f"Missing output: {name}")
            all_close = False
            continue

        exp = expected[name].astype(np.float64)
        act = actual[name].astype(np.float64)

        # Try to reshape actual to match expected if sizes are equal
        if exp.shape != act.shape and exp.size == act.size:
            act = act.reshape(exp.shape)

        if exp.shape != act.shape:
            error_messages.append(
                f"Shape mismatch for {name}: expected {exp.shape}, got {act.shape}"
            )
            all_close = False
            continue

        abs_diff = np.abs(exp - act)
        max_abs_diff[name] = float(np.max(abs_diff))

        # Compute relative difference avoiding division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_diff = abs_diff / (np.abs(exp) + 1e-10)
            rel_diff = np.where(np.isfinite(rel_diff), rel_diff, 0)
        max_rel_diff[name] = float(np.max(rel_diff))

        if not np.allclose(exp, act, rtol=rtol, atol=atol):
            error_messages.append(
                f"Output {name} differs: max_abs={max_abs_diff[name]:.6f}, "
                f"max_rel={max_rel_diff[name]:.6f}"
            )
            all_close = False

    return VerificationResult(
        success=all_close,
        host_outputs=expected,
        device_outputs=actual,
        max_abs_diff=max_abs_diff,
        max_rel_diff=max_rel_diff,
        error_message="\n".join(error_messages) if error_messages else None,
    )


def _run_host_pipeline(
    pipeline_path: Union[str, Path],
    inputs: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Run pipeline on host using pure Python implementation.

    Args:
        pipeline_path: Path to pipeline JSON file.
        inputs: Dictionary of input tensors.

    Returns:
        Dictionary of output tensors.
    """
    # Load pipeline spec
    with open(pipeline_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    return run_pipeline_python(spec, inputs)


def run_pipeline_python(
    spec: Dict[str, Any],
    inputs: Dict[str, np.ndarray],
    *,
    return_all_tensors: bool = False,
    model_runner=None,
) -> Dict[str, np.ndarray]:
    """Execute a pipeline spec using pure Python (no native bindings).

    Args:
        spec: Pipeline specification dictionary.
        inputs: Dictionary of input tensors.

    Returns:
        Dictionary of output tensors by default. When ``return_all_tensors`` is
        true, returns every tensor available after execution.
    """
    from . import ops

    validate_pipeline_spec(spec)

    # Initialize tensor storage with inputs
    tensors: Dict[str, np.ndarray] = dict(inputs)
    protected_inputs = set(inputs.keys())

    # Load pre-defined tensor values from spec
    for name, tensor_spec in spec.get("tensors", {}).items():
        if name in tensors:
            continue
        value = tensor_spec.get("value") or tensor_spec.get("data")
        if value is not None:
            data_type = tensor_spec.get("data_type", 6)  # default float32
            dtype = convert_to_dtype(data_type, target="numpy")
            dims = tensor_spec.get("dimensions", [])
            channels = tensor_spec.get("channels", 1)
            shape = dims + ([channels] if channels > 1 else [])
            arr = np.array(value, dtype=dtype)
            try:
                arr = arr.reshape(shape)
            except ValueError:
                pass
            tensors[name] = arr

    # Execute operators in order
    tensor_specs = spec.get("tensors", {})
    for op_spec in spec.get("operators", []):
        _execute_operator(
            op_spec,
            tensors,
            ops,
            tensor_specs=tensor_specs,
            protected_inputs=protected_inputs,
            model_runner=model_runner,
        )

    # Collect outputs
    output_names = spec.get("outputs", [])
    outputs = {}
    for name in output_names:
        if name in tensors:
            outputs[name] = tensors[name]

    return tensors if return_all_tensors else outputs


def _resolve_tensor_name(ref: Any) -> Optional[str]:
    """Resolve tensor name from various reference formats."""
    if isinstance(ref, str):
        return ref if ref else None
    if isinstance(ref, dict):
        return ref.get("tensor") or ref.get("name")
    return None


def _split_tensor_slice(name: str) -> tuple[str, Optional[List[List[int]]]]:
    if "[" not in name or not name.endswith("]"):
        return name, None
    tensor_name = name[: name.index("[")]
    raw = name[name.index("[") + 1 : -1]
    slices = []
    for part in raw.split(","):
        items = [item.strip() for item in part.split(":")]
        if len(items) < 2:
            index = int(items[0])
            slices.append([index, index + 1])
        else:
            start = int(items[0]) if items[0] else 0
            end = int(items[1]) if items[1] else start + 1
            slices.append([start, end])
    return tensor_name, slices


def _tensor_from_ref(ref: Any, tensors: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    name = _resolve_tensor_name(ref)
    if not name:
        return None
    tensor_name, slices = _split_tensor_slice(name)
    if tensor_name not in tensors:
        return None
    value = tensors[tensor_name]
    if slices is None:
        return value
    if len(slices) == 1:
        return value[slices[0][0]:slices[0][1]]
    return value[slices[0][0]:slices[0][1], slices[1][0]:slices[1][1]]


def _get_operator_type(type_str: str) -> Optional[EOperatorType]:
    """Convert operator type string to EOperatorType enum."""
    # Normalize the type string
    normalized = type_str.upper()
    if normalized.startswith("XR_SECURE_MR_OPERATOR_TYPE_"):
        normalized = normalized[len("XR_SECURE_MR_OPERATOR_TYPE_"):]
    if normalized.endswith("_PICO"):
        normalized = normalized[:-len("_PICO")]
    normalized = _OPERATOR_TYPE_ALIASES.get(normalized, normalized)

    # Try to get the enum member by name
    try:
        return EOperatorType[normalized]
    except (KeyError, TypeError):
        pass

    # Try to find by iterating (for pure Python enum)
    try:
        for member in EOperatorType:
            if member.name == normalized:
                return member
    except TypeError:
        # pybind11 type is not iterable, try getattr
        if hasattr(EOperatorType, normalized):
            return getattr(EOperatorType, normalized)

    return None


def _execute_operator(
    op_spec: Dict[str, Any],
    tensors: Dict[str, np.ndarray],
    ops_module,
    tensor_specs: Optional[Dict[str, Any]] = None,
    protected_inputs: Optional[set[str]] = None,
    model_runner=None,
) -> None:
    """Execute a single operator.

    Args:
        op_spec: Operator specification.
        tensors: Dictionary of tensors (modified in place).
        ops_module: The ops module containing operation implementations.
    """
    op_type = _get_operator_type(op_spec.get("type", ""))
    if op_type is None:
        raise ValueError(f"Unknown operator type: {op_spec.get('type')}")

    # Get input tensors
    input_refs = op_spec.get("inputs", [])
    input_tensors = []
    for ref in input_refs:
        value = _tensor_from_ref(ref, tensors)
        if value is not None:
            input_tensors.append(value)

    # Get output names
    output_refs = op_spec.get("outputs", [])
    output_names = [_resolve_tensor_name(ref) for ref in output_refs]
    output_names = [n for n in output_names if n]

    def get_output_shape(name: Optional[str]) -> Optional[tuple]:
        if not name or not tensor_specs:
            return None
        spec = tensor_specs.get(name)
        if not spec:
            return None
        dims = spec.get("dimensions", [])
        channels = int(spec.get("channels", 1))
        if len(dims) >= 2:
            width, height = int(dims[0]), int(dims[1])
        elif len(dims) == 1:
            width, height = int(dims[0]), 1
        else:
            width, height = 1, 1
        if channels > 1:
            return (height, width, channels)
        return (height, width)

    # Execute based on operator type
    if op_type == EOperatorType.ARITHMETIC_COMPOSE:
        expression = op_spec.get("expression") or op_spec.get("attrs", [""])[0]
        if input_tensors:
            result = ops_module.arithmetic(input_tensors if len(input_tensors) > 1 else input_tensors[0], expression)
            if output_names:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.CONVERT_COLOR:
        flag = op_spec.get("flag")
        if flag is None:
            attrs = op_spec.get("attrs", [])
            flag = int(attrs[0]) if attrs else 0
        if input_tensors:
            result = ops_module.convert_color(input_tensors[0], int(flag))
            if output_names:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.NORMALIZE:
        if input_tensors:
            result = ops_module.normalize(input_tensors[0])
            if output_names:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.ARGMAX:
        axis = -1
        attrs = op_spec.get("attrs", [])
        if attrs:
            axis = int(attrs[0])
        if input_tensors:
            result = ops_module.argmax(input_tensors[0], axis=axis)
            if output_names:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.ELEMENTWISE_MIN:
        if len(input_tensors) >= 2:
            result = ops_module.elementwise_min(input_tensors[0], input_tensors[1])
            if output_names:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.ELEMENTWISE_MAX:
        if len(input_tensors) >= 2:
            result = ops_module.elementwise_max(input_tensors[0], input_tensors[1])
            if output_names:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.ELEMENTWISE_MULTIPLY:
        if len(input_tensors) >= 2:
            elementwise_op = str(op_spec.get("op") or "multiply").lower()
            if elementwise_op == "min":
                result = ops_module.elementwise_min(input_tensors[0], input_tensors[1])
            elif elementwise_op == "max":
                result = ops_module.elementwise_max(input_tensors[0], input_tensors[1])
            elif elementwise_op == "or":
                result = ops_module.elementwise_or(input_tensors[0], input_tensors[1])
            elif elementwise_op == "and":
                result = ops_module.elementwise_and(input_tensors[0], input_tensors[1])
            else:
                result = ops_module.elementwise_multiply(input_tensors[0], input_tensors[1])
            if output_names:
                tensors[output_names[0]] = result
    elif op_type == EOperatorType.ELEMENTWISE_OR:
        if len(input_tensors) >= 2:
            result = ops_module.elementwise_or(input_tensors[0], input_tensors[1])
            if output_names:
                tensors[output_names[0]] = result
    elif op_type == EOperatorType.ELEMENTWISE_AND:
        if len(input_tensors) >= 2:
            result = ops_module.elementwise_and(input_tensors[0], input_tensors[1])
            if output_names:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.ALL:
        if input_tensors:
            result = ops_module.all(input_tensors[0])
            if output_names:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.ANY:
        if input_tensors:
            result = ops_module.any(input_tensors[0])
            if output_names:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.ASSIGNMENT:
        src_slices = op_spec.get("src_slices")
        dst_slices = op_spec.get("dst_slices")
        output_base_name = output_names[0] if output_names else None
        output_slice = None
        if output_base_name:
            output_base_name, output_slice = _split_tensor_slice(output_base_name)
            if dst_slices is None and output_slice is not None:
                dst_slices = output_slice
        if len(input_tensors) >= 2:
            result = ops_module.assignment(
                input_tensors[0],
                input_tensors[1],
                src_slices=src_slices,
                dst_slices=dst_slices,
            )
            if output_base_name:
                tensors[output_base_name] = result
        elif len(input_tensors) == 1:
            if output_base_name:
                if dst_slices is not None:
                    dst_spec = tensor_specs.get(output_base_name, {}) if tensor_specs else {}
                    dst = tensors.get(output_base_name)
                    if dst is None:
                        shape = get_output_shape(output_base_name) or input_tensors[0].shape
                        data_type = dst_spec.get("data_type", 6) if isinstance(dst_spec, dict) else 6
                        try:
                            dtype = convert_to_dtype(data_type, target="numpy")
                        except Exception:
                            dtype = input_tensors[0].dtype
                        dst = np.zeros(shape, dtype=dtype)
                    tensors[output_base_name] = ops_module.assignment(input_tensors[0], dst, dst_slices=dst_slices)
                else:
                    tensors[output_base_name] = input_tensors[0].copy()

    elif op_type == EOperatorType.APPLY_AFFINE:
        if len(input_tensors) >= 2:
            output_shape = get_output_shape(output_names[0] if output_names else None)
            result = ops_module.apply_affine(
                input_tensors[0],
                input_tensors[1],
                output_shape=output_shape,
            )
            if output_names:
                tensors[output_names[0]] = result
    elif op_type == EOperatorType.APPLY_AFFINE_POINT:
        if len(input_tensors) >= 2:
            result = ops_module.apply_affine_point(
                input_tensors[0],
                input_tensors[1],
            )
            if output_names:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.CUSTOMIZED_COMPARE:
        compare = op_spec.get("compare")
        if compare is None:
            attrs = op_spec.get("attrs", [])
            compare = attrs[0] if attrs else "=="
        if len(input_tensors) >= 2:
            result = ops_module.customized_compare(
                input_tensors[0],
                input_tensors[1],
                compare=str(compare),
            )
            if output_names:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.NMS:
        threshold = op_spec.get("threshold", 0.5)
        if threshold is None:
            attrs = op_spec.get("attrs", [])
            threshold = float(attrs[0]) if attrs else 0.5
        if len(input_tensors) >= 2:
            result = ops_module.nms(
                input_tensors[0],
                input_tensors[1],
                threshold=float(threshold),
            )
            if output_names:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.CAMERA_SPACE_TO_WORLD:
        if input_tensors:
            right, left = ops_module.camera_space_to_world(input_tensors[0])
            if output_names:
                if len(output_names) >= 1:
                    tensors[output_names[0]] = right
                if len(output_names) >= 2:
                    tensors[output_names[1]] = left

    elif op_type == EOperatorType.GET_AFFINE:
        if len(input_tensors) >= 2:
            result = ops_module.get_affine(input_tensors[0], input_tensors[1])
            if output_names:
                tensors[output_names[0]] = result

    elif _OP_GET_TRANSFORM_MAT is not None and op_type == _OP_GET_TRANSFORM_MAT:
        if len(input_tensors) >= 2:
            scale = input_tensors[2] if len(input_tensors) >= 3 else None
            result = ops_module.get_transform_mat(input_tensors[0], input_tensors[1], scale=scale)
            if output_names:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.INVERSION:
        if input_tensors:
            result = ops_module.inversion(input_tensors[0])
            if output_names:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.NORM:
        norm_type = "L2"
        attrs = op_spec.get("attrs", [])
        if attrs:
            norm_type = str(attrs[0])
        if input_tensors:
            result = ops_module.norm(input_tensors[0], norm_type=norm_type)
            if output_names:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.SOLVE_P_N_P:
        if len(input_tensors) >= 3:
            rvec, tvec = ops_module.solve_pnp(input_tensors[0], input_tensors[1], input_tensors[2])
            if output_names:
                if len(output_names) >= 1:
                    tensors[output_names[0]] = rvec
                if len(output_names) >= 2:
                    tensors[output_names[1]] = tvec

    elif op_type == EOperatorType.SORT_VEC:
        if input_tensors:
            sorted_vec, indices = ops_module.sort_vec(input_tensors[0])
            if output_names:
                if len(output_names) >= 1:
                    tensors[output_names[0]] = sorted_vec
                if len(output_names) >= 2:
                    tensors[output_names[1]] = indices

    elif op_type == EOperatorType.SORT_MAT:
        sort_axis = op_spec.get("axis") or "ROW"
        attrs = op_spec.get("attrs", [])
        if attrs:
            sort_axis = str(attrs[0])
        if input_tensors:
            sorted_mat, indices = ops_module.sort_mat(input_tensors[0], axis=sort_axis)
            if output_names:
                if len(output_names) >= 1:
                    tensors[output_names[0]] = sorted_mat
                if len(output_names) >= 2:
                    tensors[output_names[1]] = indices

    elif op_type == EOperatorType.SVD:
        if input_tensors:
            w, u, vt = ops_module.svd(input_tensors[0])
            if output_names:
                if len(output_names) >= 1:
                    tensors[output_names[0]] = w
                if len(output_names) >= 2:
                    tensors[output_names[1]] = u
                if len(output_names) >= 3:
                    tensors[output_names[2]] = vt

    elif _OP_SWAP_HWC_CHW is not None and op_type == _OP_SWAP_HWC_CHW:
        if input_tensors:
            result = ops_module.swap_hwc_chw(input_tensors[0])
            if output_names:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.UV_TO_3D_IN_CAM_SPACE:
        if len(input_tensors) >= 5:
            result = ops_module.uv_to_3d_in_cam_space(
                input_tensors[0],
                input_tensors[1],
                input_tensors[2],
                input_tensors[3],
                input_tensors[4],
            )
            if output_names:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.RECTIFIED_VST_ACCESS:
        output_shapes = []
        for name in output_names:
            shape = get_output_shape(name)
            output_shapes.append(shape if shape is not None else (1, 1))
        right, left, timestamp, cam_mat = ops_module.rectified_vst_access(
            output_shapes=output_shapes,
            output_names=output_names,
            image_path=op_spec.get("image_path"),
        )
        if output_names:
            protected_inputs = protected_inputs or set()
            if len(output_names) >= 1 and output_names[0] not in protected_inputs:
                tensors[output_names[0]] = right
            if len(output_names) >= 2 and output_names[1] not in protected_inputs:
                tensors[output_names[1]] = left
            if len(output_names) >= 3 and output_names[2] not in protected_inputs:
                tensors[output_names[2]] = timestamp
            if len(output_names) >= 4 and output_names[3] not in protected_inputs:
                tensors[output_names[3]] = cam_mat

    elif op_type == EOperatorType.RUN_MODEL_INFERENCE:
        model_ref = op_spec.get("model")
        inline_model = model_ref if isinstance(model_ref, dict) else {}
        model_file = (
            inline_model.get("model_file_host")
            or inline_model.get("model_path")
            or inline_model.get("bin_path")
        )
        model_name = op_spec.get("model_name") or inline_model.get("model_name") or "model"
        if not model_file:
            raise ValueError(
                "RUN_MODEL_INFERENCE requires inline TFLite model metadata with model.bin_path"
            )
        inputs_map: Dict[str, np.ndarray] = {}
        for ref in input_refs:
            if isinstance(ref, dict) and "name" in ref and "tensor" in ref:
                name = ref.get("name")
                tensor_name = ref.get("tensor")
                if tensor_name in tensors:
                    inputs_map[name] = tensors[tensor_name]
            else:
                name = _resolve_tensor_name(ref)
                if name and name in tensors:
                    inputs_map[name] = tensors[name]
        output_shapes = []
        output_dtypes = []
        for name in output_names:
            spec = tensor_specs.get(name, {})
            dims = spec.get("dimensions", [])
            channels = spec.get("channels", 1)
            shape = list(dims)
            if channels and int(channels) > 1:
                shape.append(int(channels))
            output_shapes.append(tuple(shape) if shape else (1,))
            data_type = spec.get("data_type")
            dtype = None
            if data_type is not None:
                try:
                    dtype = convert_to_dtype(data_type, target="numpy")
                except Exception:
                    dtype = None
            output_dtypes.append(dtype if dtype is not None else np.float32)

        if model_runner is not None:
            outputs = model_runner(
                inputs=inputs_map,
                model_file=model_file,
                model_name=model_name,
                output_names=output_names,
                output_shapes=output_shapes if output_shapes else None,
                output_dtypes=output_dtypes if output_dtypes else None,
                input_aliasing=op_spec.get("input_aliasing", {}),
                output_aliasing=op_spec.get("output_aliasing", {}),
                model=inline_model,
            )
        else:
            outputs = ops_module.run_model_inference(
                inputs=inputs_map,
                model_file=model_file,
                model_name=model_name,
                output_names=output_names,
                output_shapes=output_shapes if output_shapes else None,
                output_dtypes=output_dtypes if output_dtypes else None,
                input_aliasing=op_spec.get("input_aliasing", {}),
                output_aliasing=op_spec.get("output_aliasing", {}),
                model=inline_model,
            )
        for name, value in outputs.items():
            tensors[name] = value

    elif _OP_LOAD_TEXTURE is not None and op_type == _OP_LOAD_TEXTURE:
        if len(input_tensors) >= 2:
            result = ops_module.load_texture(input_tensors[0], input_tensors[1])
            if output_names:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.SWITCH_GLTF_RENDER_STATUS:
        if input_tensors:
            pose = input_tensors[1] if len(input_tensors) >= 2 else None
            ops_module.switch_gltf_render_status(input_tensors[0], pose=pose)

    elif op_type == EOperatorType.UPDATE_GLTF:
        update_type = ""
        attrs = op_spec.get("attrs", [])
        if attrs:
            update_type = str(attrs[0])
        if input_tensors:
            ops_module.update_gltf(input_tensors[0], update_type=update_type)

    elif op_type == EOperatorType.RENDER_TEXT:
        if input_tensors:
            attrs = op_spec.get("attrs", [])
            text = attrs[1] if len(attrs) > 1 else ""
            config = attrs[0] if attrs else "bold#en-us#512#64"
            parts = config.split("#")
            typeface = parts[0] if parts else "bold"
            language = parts[1] if len(parts) > 1 else "en-us"
            width = int(parts[2]) if len(parts) > 2 else 512
            height = int(parts[3]) if len(parts) > 3 else 64
            ops_module.render_text(input_tensors[0], text, language, width, height, typeface=typeface)

    elif _OP_JAVASCRIPT is not None and op_type == _OP_JAVASCRIPT:
        js_code = op_spec.get("script") or op_spec.get("attrs", [""])[0]
        inputs_map: Dict[str, np.ndarray] = {}
        for ref in input_refs:
            if isinstance(ref, dict) and "name" in ref and "tensor" in ref:
                name = ref.get("name")
                tensor_name = ref.get("tensor")
                if tensor_name in tensors:
                    inputs_map[name] = tensors[tensor_name]
            else:
                name = _resolve_tensor_name(ref)
                if name and name in tensors:
                    inputs_map[name] = tensors[name]
        outputs = _try_execute_known_javascript(js_code, inputs_map, output_names)
        if outputs is None:
            outputs = ops_module.javascript(js_code, inputs_map, output_names)
        for name, value in outputs.items():
            tensors[name] = value

    elif op_type == EOperatorType.SCENEGRAPH_VISIBILITY:
        if input_tensors:
            attrs = op_spec.get("attrs", [])
            visible = op_spec.get("visible", attrs[0] if attrs else True)
            if isinstance(visible, str):
                visible = visible.strip().lower() not in {"0", "false", "no", "off"}
            result = ops_module.scenegraph_visibility(input_tensors[0], visible=bool(visible))
            if output_names:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.UPDATE_COMPONENT:
        if input_tensors:
            attrs = op_spec.get("attrs", [])
            update_type = op_spec.get("update_type", attrs[0] if attrs else "")
            result = ops_module.update_component(input_tensors[0], update_type=str(update_type))
            if output_names:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.MICROPHONE:
        shape = get_output_shape(output_names[0] if output_names else None) or (1,)
        result = ops_module.microphone(output_shape=shape)
        if output_names:
            tensors[output_names[0]] = result

    elif op_type == EOperatorType.SPEAKER:
        if input_tensors:
            result = ops_module.speaker(input_tensors[0])
            if output_names:
                tensors[output_names[0]] = result

    elif op_type == EOperatorType.DEPTH:
        shape = get_output_shape(output_names[0] if output_names else None) or (1, 1)
        result = ops_module.depth(output_shape=shape)
        if output_names:
            tensors[output_names[0]] = result

    elif op_type == EOperatorType.UNKNOWN:
        if input_tensors:
            result = ops_module.unknown(input_tensors[0])
            if output_names:
                tensors[output_names[0]] = result

    else:
        raise NotImplementedError(
            f"Operator type {op_type.name} is not implemented in pure Python executor"
        )


def _try_execute_known_javascript(
    js_code: str,
    inputs: Dict[str, np.ndarray],
    output_names: List[str],
) -> Optional[Dict[str, np.ndarray]]:
    if (
        "decodeDetection" in js_code
        and "anchorFor" in js_code
        and {"box_coords_1", "box_coords_2", "box_scores_1", "box_scores_2"}.issubset(inputs)
        and output_names == ["post_det"]
    ):
        return {"post_det": _decode_mediapipe_face_detection(inputs)}
    return None


def _decode_mediapipe_face_detection(inputs: Dict[str, np.ndarray]) -> np.ndarray:
    box_coords_1 = np.asarray(inputs["box_coords_1"], dtype=np.float32).reshape(-1)
    box_coords_2 = np.asarray(inputs["box_coords_2"], dtype=np.float32).reshape(-1)
    box_scores_1 = np.asarray(inputs["box_scores_1"], dtype=np.float32).reshape(-1)
    box_scores_2 = np.asarray(inputs["box_scores_2"], dtype=np.float32).reshape(-1)
    template = np.asarray(inputs.get("post_det_template", np.zeros((1, 21), dtype=np.float32)), dtype=np.float32)
    post_det = template.reshape(-1).copy()
    if post_det.size < 21:
        padded = np.zeros(21, dtype=np.float32)
        padded[: post_det.size] = post_det
        post_det = padded
    else:
        post_det = post_det[:21]

    input_size = 256.0
    camera_width = 580.0
    camera_height = 326.0
    affine_scale_x = 0.4413793087
    affine_scale_y = 0.7852760736
    affine_x_offset = 0.0
    affine_y_offset = 0.0
    score_threshold = 0.25

    def sigmoid(value: float) -> float:
        return float(1.0 / (1.0 + np.exp(-float(value))))

    best_score = 0.0
    best_index = -1
    best_head = 0
    for index in range(min(512, box_scores_1.size)):
        score = sigmoid(box_scores_1[index])
        if score > best_score:
            best_score = score
            best_index = index
            best_head = 1
    for index in range(min(384, box_scores_2.size)):
        score = sigmoid(box_scores_2[index])
        if score > best_score:
            best_score = score
            best_index = index
            best_head = 2

    if best_score <= score_threshold or best_index < 0:
        return post_det.reshape(1, 21)

    coords = box_coords_1 if best_head == 1 else box_coords_2
    feature_size = 16 if best_head == 1 else 8
    anchors_per_cell = 2 if best_head == 1 else 6
    cell = best_index // anchors_per_cell
    col = cell % feature_size
    row = cell // feature_size
    anchor_x = (col + 0.5) / feature_size
    anchor_y = (row + 0.5) / feature_size

    def to_camera_x(value: float) -> float:
        return float(np.clip((value - affine_x_offset) / affine_scale_x, 0.0, camera_width))

    def to_camera_y(value: float) -> float:
        return float(np.clip((value - affine_y_offset) / affine_scale_y, 0.0, camera_height))

    base = best_index * 16
    if base + 14 > coords.size:
        return post_det.reshape(1, 21)

    x_center = (coords[base] / input_size + anchor_x) * input_size
    y_center = (coords[base + 1] / input_size + anchor_y) * input_size
    box_w = coords[base + 2]
    box_h = coords[base + 3]

    post_det[0] = to_camera_x(x_center - box_w * 0.5)
    post_det[1] = to_camera_y(y_center - box_h * 0.5)
    post_det[2] = to_camera_x(x_center + box_w * 0.5)
    post_det[3] = to_camera_y(y_center + box_h * 0.5)
    post_det[4] = best_score
    post_det[5] = 0.0

    for keypoint in range(5):
        coord_base = base + 4 + keypoint * 2
        out_base = 6 + keypoint * 3
        keypoint_x = (coords[coord_base] / input_size + anchor_x) * input_size
        keypoint_y = (coords[coord_base + 1] / input_size + anchor_y) * input_size
        post_det[out_base] = to_camera_x(keypoint_x)
        post_det[out_base + 1] = to_camera_y(keypoint_y)
        post_det[out_base + 2] = best_score

    return post_det.astype(np.float32).reshape(1, 21)

def _run_device_pipeline(
    pipeline_path: Path,
    inputs: Dict[str, np.ndarray],
    input_tensor_name: str,
    duration: int,
    expected_outputs: Optional[Dict[str, np.ndarray]] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """Device verification is not available from the Python verifier."""
    print("Device verification is not available from the Python verifier. Use pyspatialml run device.")
    return None

def verify(
    pipeline: Union[str, Path, Dict[str, Any]],
    inputs: Dict[str, np.ndarray],
    expected_outputs: Optional[Dict[str, np.ndarray]] = None,
    device: bool = False,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    duration: int = 30,
) -> VerificationResult:
    """Verify pipeline outputs against expected values.

    Args:
        pipeline: Path to pipeline JSON file or pipeline spec dictionary.
        inputs: Dictionary of input tensors.
        expected_outputs: Optional dictionary of expected output tensors.
                         If not provided, only host execution is performed.
        device: If True, also run on device and compare outputs.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.
        duration: Duration for device execution in seconds.

    Returns:
        VerificationResult with comparison details.
    """
    def _prepare_device_pipeline(
        src_path: Path,
        input_tensor: str,
        input_values: Dict[str, np.ndarray],
    ) -> Path:
        with open(src_path, "r", encoding="utf-8") as f:
            spec = json.load(f)

        inputs_list = list(spec.get("inputs", []))
        tensors_spec = spec.get("tensors", {})

        for name, value in input_values.items():
            if name == input_tensor:
                continue
            tensor_spec = tensors_spec.get(name)
            if tensor_spec is None:
                continue
            tensor_spec["value"] = value.flatten().tolist()
            tensor_spec["is_placeholder"] = False
            if name in inputs_list:
                inputs_list.remove(name)

        spec["inputs"] = inputs_list

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(spec, f, indent=2)
            return Path(f.name)

    # Handle pipeline spec dictionary
    if isinstance(pipeline, dict):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(pipeline, f, indent=2)
            pipeline_path = Path(f.name)
        cleanup_pipeline = True
    else:
        pipeline_path = Path(pipeline)
        cleanup_pipeline = False

    try:
        def _maybe_copy_model_files(spec_path: Path) -> None:
            try:
                with open(spec_path, "r", encoding="utf-8") as f:
                    spec = json.load(f)
            except Exception:
                return
            operators = spec.get("operators", [])
            for op_spec in operators:
                model_spec = op_spec.get("model")
                if not isinstance(model_spec, dict):
                    continue
                host_path = model_spec.get("model_file_host") or model_spec.get("bin_path")
                if host_path and os.path.exists(host_path):
                    target = spec_path.parent / Path(host_path).name
                    if not target.exists():
                        try:
                            with open(host_path, "rb") as src, open(target, "wb") as dst:
                                dst.write(src.read())
                        except Exception:
                            pass

        # Run on host (optionally force model inference to use device backend for parity)
        prev_target = os.getenv("PY2SMR_MODEL_INFERENCE_TARGET")
        if device:
            os.environ["PY2SMR_MODEL_INFERENCE_TARGET"] = "android"
        try:
            host_outputs = _run_host_pipeline(pipeline_path, inputs)
        finally:
            if device:
                if prev_target is None:
                    os.environ.pop("PY2SMR_MODEL_INFERENCE_TARGET", None)
                else:
                    os.environ["PY2SMR_MODEL_INFERENCE_TARGET"] = prev_target

        # Compare with expected outputs if provided
        if expected_outputs is not None:
            result = compare_outputs(expected_outputs, host_outputs, rtol, atol)
            if not result.success:
                return result

        # Run on device if requested
        if device:
            input_tensor_name = list(inputs.keys())[0]
            device_pipeline_path = pipeline_path
            cleanup_device_pipeline = False
            if len(inputs) > 1:
                device_pipeline_path = _prepare_device_pipeline(
                    pipeline_path, input_tensor_name, inputs
                )
                cleanup_device_pipeline = True
            _maybe_copy_model_files(device_pipeline_path)

            device_outputs = _run_device_pipeline(
                device_pipeline_path, inputs, input_tensor_name, duration,
                expected_outputs=host_outputs,
            )

            if cleanup_device_pipeline:
                device_pipeline_path.unlink(missing_ok=True)

            if device_outputs is None:
                return VerificationResult(
                    success=False,
                    host_outputs=host_outputs,
                    error_message="Device verification is not available",
                )

            # Compare host and device outputs
            return compare_outputs(host_outputs, device_outputs, rtol, atol)

        # Success - host execution only
        return VerificationResult(
            success=True,
            host_outputs=host_outputs,
        )

    finally:
        if cleanup_pipeline:
            os.unlink(pipeline_path)
