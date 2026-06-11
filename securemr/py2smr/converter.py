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
"""Convert trace context to SecureMR pipeline JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from securemr.core.types import BaseType, EDataType, EOperatorType
from securemr.core.utils import convert_from_dtype, convert_to_dtype, mat_flag, type_to_name

from .tracer import TraceContext, TensorInfo, TracedOp

__all__ = ["convert", "trace_to_pipeline_spec"]

_OP_JAVASCRIPT = getattr(EOperatorType, "JAVASCRIPT", getattr(EOperatorType, "JS_SCRIPTING", None))
_OP_SORT_MAT = getattr(EOperatorType, "SORT_MAT", None)
_OP_NORM = getattr(EOperatorType, "NORM", None)
_OP_RENDER_TEXT = getattr(EOperatorType, "RENDER_TEXT", None)
_OP_UPDATE_GLTF = getattr(EOperatorType, "UPDATE_GLTF", None)
_OP_RUN_MODEL_INFERENCE = getattr(EOperatorType, "RUN_MODEL_INFERENCE", None)
_OP_GET_AFFINE = getattr(EOperatorType, "GET_AFFINE", None)
_OP_SOLVE_PNP = getattr(EOperatorType, "SOLVE_P_N_P", None)
_OP_SORT_VEC = getattr(EOperatorType, "SORT_VEC", None)


# Mapping from numpy dtype to EDataType
NUMPY_TO_EDATATYPE = {
    np.uint8: EDataType.UINT8,
    np.int8: EDataType.INT8,
    np.uint16: EDataType.UINT16,
    np.int16: EDataType.INT16,
    np.int32: EDataType.INT32,
    np.float32: EDataType.FLOAT32,
    np.float64: EDataType.FLOAT64,
}


def _get_edatatype(dtype: np.dtype) -> EDataType:
    """Convert numpy dtype to EDataType."""
    dtype_type = np.dtype(dtype).type
    if dtype_type in NUMPY_TO_EDATATYPE:
        return NUMPY_TO_EDATATYPE[dtype_type]
    # Default to float32 for unknown types
    return EDataType.FLOAT32


def _tensor_info_to_spec(info: TensorInfo) -> Dict[str, Any]:
    """Convert TensorInfo to pipeline tensor spec."""
    shape = info.shape
    dtype = info.dtype
    name_lower = info.name.lower()

    # Determine dimensions and channels
    if len(shape) == 0:
        dimensions = [1, 1]
        channels = 1
    elif len(shape) == 1:
        dimensions = [int(shape[0]), 1]
        channels = 1
    elif len(shape) == 2:
        dimensions = [int(shape[0]), int(shape[1])]  # H, W
        channels = 1
    elif len(shape) == 3:
        dimensions = [int(shape[0]), int(shape[1])]  # H, W
        channels = int(shape[2])
    else:
        # For higher dimensions, flatten to 2D
        dimensions = [int(np.prod(shape[1:])), int(shape[0])]
        channels = 1

    usage = 6  # MAT by default
    if "timestamp" in name_lower and np.dtype(dtype).type == np.int32:
        total = int(np.prod(shape)) if len(shape) > 0 else 1
        if total == 4:
            dimensions = [1]
            channels = 4
            usage = 5  # TIMESTAMP

    edatatype = _get_edatatype(dtype)
    data_type_val = convert_from_dtype(edatatype, source="smr")
    flag = mat_flag(edatatype, channels)

    spec = {
        "dimensions": dimensions,
        "channels": channels,
        "data_type": data_type_val,
        "is_placeholder": info.is_input or info.is_output,
        "usage": usage,
        "flag": flag,
    }

    # Include value for non-placeholder tensors with stored values
    if info.value is not None and not info.is_input:
        spec["value"] = [float(x) for x in info.value.flatten()]

    return spec


def _op_to_spec(op: TracedOp) -> Dict[str, Any]:
    """Convert TracedOp to pipeline operator spec."""
    spec = {
        "type": type_to_name(op.op_type),
        "inputs": op.input_names,
        "outputs": op.output_names,
    }

    # Add operator-specific fields
    if op.op_type == EOperatorType.ARITHMETIC_COMPOSE and op.attrs:
        spec["expression"] = op.attrs[0]
    elif op.op_type == EOperatorType.CONVERT_COLOR and op.attrs:
        try:
            spec["flag"] = int(op.attrs[0])
        except ValueError:
            spec["flag"] = op.attrs[0]
    elif op.op_type == EOperatorType.NMS and op.attrs:
        try:
            spec["threshold"] = float(op.attrs[0])
        except ValueError:
            spec["threshold"] = op.attrs[0]
    elif op.op_type == EOperatorType.CUSTOMIZED_COMPARE and op.attrs:
        spec["compare"] = op.attrs[0]
    elif _OP_NORM is not None and op.op_type == _OP_NORM and op.attrs:
        spec["norm_type"] = op.attrs[0]
    elif _OP_SORT_MAT is not None and op.op_type == _OP_SORT_MAT and op.attrs:
        spec["axis"] = op.attrs[0]
    elif _OP_JAVASCRIPT is not None and op.op_type == _OP_JAVASCRIPT and op.attrs:
        spec["script"] = op.attrs[0]
    elif _OP_RENDER_TEXT is not None and op.op_type == _OP_RENDER_TEXT and op.attrs:
        spec["config"] = op.attrs[0]
        if len(op.attrs) > 1:
            spec["text"] = op.attrs[1]
    elif _OP_UPDATE_GLTF is not None and op.op_type == _OP_UPDATE_GLTF and op.attrs:
        spec["update_type"] = op.attrs[0]
    elif _OP_RUN_MODEL_INFERENCE is not None and op.op_type == _OP_RUN_MODEL_INFERENCE:
        if "model_type" in op.extra_info:
            spec["model_type"] = op.extra_info["model_type"]
        if "model_target" in op.extra_info:
            spec["model_target"] = op.extra_info["model_target"]
        if "cpu_target_num_threads" in op.extra_info:
            spec["cpu_target_num_threads"] = op.extra_info["cpu_target_num_threads"]
        if "model" in op.extra_info:
            spec["model"] = op.extra_info["model"]
        if "model_id" in op.extra_info:
            spec["model_id"] = op.extra_info["model_id"]
        if "model_asset" in op.extra_info:
            spec["model_asset"] = op.extra_info["model_asset"]
        elif "device_model_file" in op.extra_info:
            spec["model_file"] = op.extra_info["device_model_file"]
        elif "model_file" in op.extra_info:
            spec["model_file"] = op.extra_info["model_file"]
        if "model_name" in op.extra_info:
            spec["model_name"] = op.extra_info["model_name"]
        if "input_aliasing" in op.extra_info:
            spec["input_aliasing"] = op.extra_info["input_aliasing"]
        if "output_aliasing" in op.extra_info:
            spec["output_aliasing"] = op.extra_info["output_aliasing"]
        if "model_file_host" in op.extra_info:
            spec["model_file_host"] = op.extra_info["model_file_host"]
    elif op.op_type == EOperatorType.ASSIGNMENT:
        if "src_slices" in op.extra_info:
            spec["src_slices"] = op.extra_info["src_slices"]
        if "dst_slices" in op.extra_info:
            spec["dst_slices"] = op.extra_info["dst_slices"]

    return spec


def trace_to_pipeline_spec(ctx: TraceContext) -> Dict[str, Any]:
    """Convert a TraceContext to a pipeline specification dictionary.

    Args:
        ctx: The trace context containing recorded operations.

    Returns:
        Pipeline specification dictionary ready for JSON serialization.
    """
    # Build tensor specs
    tensors = {}
    for name, info in ctx.tensors.items():
        tensors[name] = _tensor_info_to_spec(info)

    # Build operator specs
    operators = []
    for op in ctx.operations:
        operators.append(_op_to_spec(op))

    # Determine inputs and outputs
    inputs = [name for name, info in ctx.tensors.items() if info.is_input]
    outputs = [name for name, info in ctx.tensors.items() if info.is_output]

    # Fix up tensor specs for scalar-result operators (ALL/ANY).
    scalar_result_ops = {EOperatorType.ALL, EOperatorType.ANY}
    scalar_outputs = set()
    for op in ctx.operations:
        if op.op_type in scalar_result_ops and op.output_names:
            scalar_outputs.add(op.output_names[0])
    for name in scalar_outputs:
        spec = tensors.get(name)
        if spec is None:
            continue
        spec["dimensions"] = [1]
        spec["channels"] = 1
        spec["usage"] = 2  # TensorType.SCALAR
        spec.pop("flag", None)

    # Fix up tensor specs for argmax outputs (channel-wise indices).
    for op in ctx.operations:
        if op.op_type != EOperatorType.ARGMAX or not op.input_names or not op.output_names:
            continue
        input_spec = tensors.get(op.input_names[0])
        output_spec = tensors.get(op.output_names[0])
        if input_spec is None or output_spec is None:
            continue
        input_channels = int(input_spec.get("channels", 1))
        input_dims = len(input_spec.get("dimensions", [])) or 1
        output_spec["dimensions"] = [1, input_channels] if input_channels > 0 else [1, 1]
        output_spec["channels"] = input_dims
        output_spec["usage"] = 6  # TensorType.MAT
        output_spec.pop("flag", None)

    # Fix up tensor specs for get_affine inputs (expects 2-channel point arrays).
    if _OP_GET_AFFINE is not None:
        for op in ctx.operations:
            if op.op_type != _OP_GET_AFFINE:
                continue
            for name in op.input_names[:2]:
                spec = tensors.get(name)
                if spec is None:
                    continue
                if int(spec.get("channels", 1)) == 2:
                    continue
                dims = spec.get("dimensions", [])
                total = 1
                for dim in dims:
                    try:
                        total *= int(dim)
                    except (TypeError, ValueError):
                        total = 0
                        break
                total *= int(spec.get("channels", 1) or 1)
                if total != 6:
                    continue
                spec["dimensions"] = [3, 1]
                spec["channels"] = 2
                data_type_val = spec.get("data_type")
                if data_type_val is not None:
                    try:
                        dtype_enum = convert_to_dtype(data_type_val, target="smr")
                        spec["flag"] = mat_flag(dtype_enum, 2)
                    except Exception:
                        pass

    # Fix up tensor specs for solve_pnp inputs (expects channelized point arrays).
    if _OP_SOLVE_PNP is not None:
        for op in ctx.operations:
            if op.op_type != _OP_SOLVE_PNP or len(op.input_names) < 2:
                continue
            for name, channels in zip(op.input_names[:2], [3, 2]):
                spec = tensors.get(name)
                if spec is None:
                    continue
                dims = spec.get("dimensions", [])
                total = 1
                for dim in dims:
                    try:
                        total *= int(dim)
                    except (TypeError, ValueError):
                        total = 0
                        break
                total *= int(spec.get("channels", 1) or 1)
                if channels <= 0 or total % channels != 0:
                    continue
                count = total // channels
                spec["dimensions"] = [int(count), 1]
                spec["channels"] = channels
                data_type_val = spec.get("data_type")
                if data_type_val is not None:
                    try:
                        dtype_enum = convert_to_dtype(data_type_val, target="smr")
                        spec["flag"] = mat_flag(dtype_enum, channels)
                    except Exception:
                        pass

    # Fix up tensor specs for sort_vec inputs/outputs (vector stored as scalar usage).
    if _OP_SORT_VEC is not None:
        for op in ctx.operations:
            if op.op_type != _OP_SORT_VEC or not op.input_names:
                continue
            for name in list(op.input_names) + list(op.output_names or []):
                spec = tensors.get(name)
                if spec is None:
                    continue
                dims = spec.get("dimensions", [])
                total = 1
                for dim in dims:
                    try:
                        total *= int(dim)
                    except (TypeError, ValueError):
                        total = 0
                        break
                total *= int(spec.get("channels", 1) or 1)
                if total <= 0:
                    continue
                spec["dimensions"] = [int(total)]
                spec["channels"] = 1
                spec["usage"] = 2  # TensorType.SCALAR

    # Build final spec
    spec = {
        "metadata": {"version": 1},
        "tensors": tensors,
        "operators": operators,
        "inputs": inputs,
        "outputs": outputs,
    }

    return spec


def convert(
    ctx: TraceContext,
    output: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Convert a trace context to pipeline JSON.

    Args:
        ctx: The trace context containing recorded operations.
        output: Optional path to save the pipeline JSON file.

    Returns:
        Pipeline specification dictionary.
    """
    spec = trace_to_pipeline_spec(ctx)

    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2, ensure_ascii=False)

    return spec
