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

"""
Pure Python utilities to record and restore SecureMR pipelines.

This module extends the bound `smr.Pipeline` with a recording spec that captures
allocated tensors, operators, and their connections, then saves/loads from JSON.
DeserializedPipeline builds a runnable pipeline from a JSON spec saved by Pipeline.
"""

import json
import os
import typing
from typing import Dict, List, Any, Optional, Union, Iterable

import numpy as np
import securemr as smr
import re


__all__ = ["Pipeline", "DeserializedPipeline", "add_vst_operator", "add_model_inference_operator"]

SPEC_VERSION = "1.0.0"

NUMPY_DTYPE = [None, np.uint8, np.int8, np.uint16, np.int16, np.int32, np.float32, np.float64]

SMR_DTYPE = [None, smr.EDataType.UINT8, smr.EDataType.INT8, smr.EDataType.UINT16,
             smr.EDataType.INT16, smr.EDataType.INT32,smr.EDataType.FLOAT32,
             smr.EDataType.FLOAT64]

def convert_to_dtype(data_type, target="numpy") -> "type":
    """
    Convert XrSecureMrTensorDataTypePICO to numpy / smr data type.
    """
    if target == "numpy":
        return NUMPY_DTYPE[data_type]
    elif target == "smr":
        return SMR_DTYPE[data_type]
    else:
        raise NotImplementedError


def convert_from_dtype(data_type, source="numpy") -> "type":
    """
    Convert numpy / smr data type to XrSecureMrTensorDataTypePICO.
    """
    if source =="numpy":
        return NUMPY_DTYPE.index(data_type)
    elif source == "smr":
        return SMR_DTYPE.index(data_type)
    else:
        raise NotImplementedError


def mat_flag(dtype: smr.EDataType, channels: int) -> int:
    return int(dtype) | smr.BaseType.MAT | (smr.BaseType.CHANNEL_MASK & channels)


def unmat_flag(flag: int) -> tuple[smr.EDataType, int]:
    # Optional: ensure this is a MAT-typed flag
    if not (int(flag) & int(smr.BaseType.MAT)):
        raise ValueError("flag does n encode a MAT type")

    # Extract channels
    channels = int(flag) & int(smr.BaseType.CHANNEL_MASK)

    # Recover dtype: clear MAT and channel bits, then cast to EDataType
    clear_mask = int(smr.BaseType.MAT) | int(smr.BaseType.CHANNEL_MASK)
    dtype_bits = int(flag) & ~clear_mask
    dtype = smr.EDataType(dtype_bits)
    
    return dtype, channels

def as_list(data):
    if isinstance(data, (list, tuple)):
        return list(data)
    elif data is None:
        return []
    else:
        return data


def name_to_type(name: str) -> smr.EOperatorType:
    # Build a map of enum members dynamically so we cover all types
    members: Dict[str, Any] = {}
    for attr in dir(smr.EOperatorType):
        if attr.startswith("__"):
            continue
        try:
            val = getattr(smr.EOperatorType, attr)
            # Ensure this attr is an enum value (int-like)
            _ = int(val)
            members[attr.upper()] = val
        except Exception:
            continue

    # Accept ints or numeric strings directly
    if isinstance(name, int):
        val = int(name)
        for k, v in members.items():
            try:
                if int(v) == val:
                    return v
            except Exception:
                pass
        # Fallback to enum constructor by value if available
        try:
            return smr.EOperatorType(val)
        except Exception:
            return val  # last resort; many bindings accept int-castable

    s = str(name).strip()
    # Handle patterns: unknown_#, numeric, or enum-like strings
    m = re.fullmatch(r"unknown_(\d+)", s, flags=re.IGNORECASE)
    if m:
        return name_to_type(int(m.group(1)))
    if s.isdigit():
        return name_to_type(int(s))

    # Provide aliases for common short names used in Pipeline serialization
    alias: Dict[str, str] = {
        "get_affine": "GET_AFFINE",
        "apply_affine": "APPLY_AFFINE",
        "assignment": "ASSIGNMENT",
        "cvt_color": "CONVERT_COLOR",
        "convert_color": "CONVERT_COLOR",
        "arithmetic": "ARITHMETIC_COMPOSE",
        "arithmetic_compose": "ARITHMETIC_COMPOSE",
        "type_convert": "TYPE_CONVERT",
        "camera_access": "RECTIFIED_VST_ACCESS",
        "rectified_vst_access": "RECTIFIED_VST_ACCESS",
    }

    # 'ALL',
    # 'ANY',
    # 'APPLY_AFFINE',
    # 'APPLY_AFFINE_POINT',
    # 'ARGMAX',
    # 'ARITHMETIC_COMPOSE',
    # 'ASSIGNMENT',
    # 'CAMERA_SPACE_TO_WORLD',
    # 'CONVERT_COLOR',
    # 'CUSTOMIZED_COMPARE',
    # 'ELEMENTWISE_AND',
    # 'ELEMENTWISE_MAX',
    # 'ELEMENTWISE_MIN',
    # 'ELEMENTWISE_MULTIPLY',
    # 'ELEMENTWISE_OR',
    # 'GET_AFFINE',
    # 'INVERSION',
    # 'MAKE_TRANSFORM_MAT',
    # 'NMS',
    # 'NORMALIZE',
    # 'RECTIFIED_VST_ACCESS',
    # 'RENDER_TEXT',
    # 'RUN_MODEL_INFERENCE',
    # 'SOLVE_P_N_P',
    # 'SORT_MAT',
    # 'SORT_VEC',
    # 'SWITCH_GLTF_RENDER_STATUS',
    # 'UNKNOWN',
    # 'UPDATE_GLTF',
    # 'UPLOAD_TEXTURE_TO_GLTF',
    # 'UV_TO_3D_IN_CAM_SPACE'

    key = alias.get(s.lower())
    if key and key in members:
        return members[key]

    # Normalize any other token to UPPER_CASE_WITH_UNDERSCORES and try direct match
    norm = re.sub(r"[^0-9A-Za-z]+", "_", s).upper()
    if norm in members:
        return members[norm]

    # As a last attempt, do a case-insensitive match over known members
    for k, v in members.items():
        if k.upper() == norm.upper():
            return v

    raise KeyError(f"Unsupported operator type name: {name}")


def type_to_name(op_type: smr.EOperatorType) -> str:
    """Return a stable, human-friendly name for an operator type.

    - Covers all smr.EOperatorType values dynamically.
    - Produces lowercase_with_underscores.
    - Keeps historical aliases for certain ops for compatibility.
    - Falls back to "unknown_<id>" if not recognized.
    """
    try:
        key_val = int(op_type)
    except Exception:
        key_val = int(op_type)

    # Build reverse map: value -> enum member name
    value_to_name: Dict[int, str] = {}
    for attr in dir(smr.EOperatorType):
        if attr.startswith("__"):
            continue
        try:
            val = getattr(smr.EOperatorType, attr)
            value_to_name[int(val)] = attr
        except Exception:
            continue

    enum_name = value_to_name.get(key_val)
    if not enum_name:
        return f"unknown_{key_val}"

    # Normalize to lower_case_with_underscores
    pretty = re.sub(r"[^0-9A-Za-z]+", "_", enum_name).lower()

    # Historic aliases
    if enum_name == "CONVERT_COLOR":
        return "cvt_color"
    if enum_name == "ARITHMETIC_COMPOSE":
        return "arithmetic"

    return pretty


def _ensure_list_size(lst: List[typing.Optional[int]], size: int) -> None:
    if len(lst) < size:
        lst.extend([None] * (size - len(lst)))

# (Legacy helpers removed; functionality moved into Pipeline/DeserializedPipeline)


class Pipeline(smr.Pipeline):
    """
    An extened pipeline for serialization.

    """
    def __init__(self):
        super(Pipeline, self).__init__()
        self.spec: Dict[str, Any] = {
            # match mnist_inference_pipeline.json key names
            "metadata": {"version": 1},
            "tensors": {},      # name -> {dimensions, channels, data_type, is_placeholder, usage, flag}
            "operators": [],    # list of {type, inputs, outputs, ...}
            # extra IO lists for convenience
            "inputs": [],
            "outputs": [],
        }
        self._tensor_id_to_name: Dict[int, str] = {}
        self._op_id_to_spec_idx: Dict[int, int] = {}
        self._tensor_obj_to_id: Dict[int, int] = {}

    def _record_tensor(self,
                       tid: int,
                       shape: Iterable[int],
                       flag: int,
                       is_placeholder: bool,
                       name: str,
                       value: np.ndarray,
                       ) -> None:
        # generate a deterministic name for this tensor id if not present
        if not name:
            name = f"tensor_{int(tid)}"
        self._tensor_id_to_name[int(tid)] = name
        # Fix-up for point types: they are always float32 with 2 channels
        if int(flag) & int(smr.BaseType.MAT):
            dtype, channels = unmat_flag(flag)
            data_type_val = convert_from_dtype(dtype, source="smr")
        elif int(flag) & int(smr.BaseType.POINT_2):
            channels = 2
            data_type_val = convert_from_dtype(np.float32)
        else:
            raise NotImplementedError

        self.spec["tensors"][name] = {
            "dimensions": list(shape),
            "channels": int(channels) if channels > 0 else 1,
            "data_type": int(data_type_val),
            "is_placeholder": bool(is_placeholder),
            "usage": 6,
            # keep flag for lossless reconstruction
            "flag": int(flag),
            "value": [float(x) for x in value.flatten()] if value is not None else None,
        }

    def allocate_placeholder(self, shape: Iterable[int], flag: int, name: str = None):
        tid = super(Pipeline, self).allocate_placeholder(shape, flag)
        self._record_tensor(int(tid), shape, flag, True, name, None)
        return tid

    def allocate_local_tensor(self, shape: Iterable[int], flag: int, name: str = None, value: np.ndarray = None):
        tid = super(Pipeline, self).allocate_local_tensor(shape, flag)
        self._record_tensor(int(tid), shape, flag, False, name, value)
        return tid

    def query_local_tensor(self, tensor_id: int):
        t = super(Pipeline, self).query_local_tensor(tensor_id)
        self._tensor_obj_to_id[id(t)] = int(tensor_id)
        return t

    def allocate_operator(self, op_type: smr.EOperatorType, attrs: Optional[List[str]] = None):
        oid = super(Pipeline, self).allocate_operator(op_type, [] if attrs is None else attrs)
        spec_idx = len(self.spec["operators"])
        self._op_id_to_spec_idx[int(oid)] = spec_idx
        op_entry: Dict[str, Any] = {
            "type": type_to_name(op_type),
            "inputs": [],
            "outputs": [],
        }
        # map attrs into named fields for known ops
        if attrs:
            # TODO: support more operators
            if op_type == smr.EOperatorType.CONVERT_COLOR and len(attrs) >= 1:
                try:
                    op_entry["flag"] = int(attrs[0])
                except Exception:
                    op_entry["flag"] = str(attrs[0])
            elif op_type == smr.EOperatorType.ARITHMETIC_COMPOSE and len(attrs) >= 1:
                op_entry["expression"] = str(attrs[0])
            elif op_type == smr.EOperatorType.NMS and len(attrs) >= 1:
                # record IoU threshold for reconstruction
                try:
                    op_entry["threshold"] = float(attrs[0])
                except Exception:
                    op_entry["threshold"] = str(attrs[0])
        self.spec["operators"].append(op_entry)
        return oid

    def query_operator(self, op_id: int):
        real_op = super(Pipeline, self).query_operator(op_id)
        pipeline = self

        class _OpProxy:
            def __init__(self, _real):
                self._real = _real

            def __getattr__(self, item):
                return getattr(self._real, item)

            def data_as_operand(self, tensor: smr.Tensor, index: int):
                spec_idx = pipeline._op_id_to_spec_idx.get(int(op_id))
                if spec_idx is not None:
                    opspec = pipeline.spec["operators"][spec_idx]
                    tid = pipeline._tensor_obj_to_id.get(id(tensor))
                    name = pipeline._tensor_id_to_name.get(int(tid)) if tid is not None else None
                    _ensure_list_size(opspec["inputs"], index + 1)
                    opspec["inputs"][index] = name
                return self._real.data_as_operand(tensor, index)

            def connect_result_to_data_array(self, index: int, tensor: smr.Tensor):
                spec_idx = pipeline._op_id_to_spec_idx.get(int(op_id))
                if spec_idx is not None:
                    opspec = pipeline.spec["operators"][spec_idx]
                    tid = pipeline._tensor_obj_to_id.get(id(tensor))
                    name = pipeline._tensor_id_to_name.get(int(tid)) if tid is not None else None
                    _ensure_list_size(opspec["outputs"], index + 1)
                    opspec["outputs"][index] = name
                return self._real.connect_result_to_data_array(index, tensor)

        return _OpProxy(real_op)
    
    def register_tensor(self, name, tensor, with_value=True):
        """Record a non-placeholder tensor with concrete values into spec.

        The tensor won't be allocated into the graph automatically; this API only
        records metadata and data for serialization. DeserializedPipeline will
        allocate it as a placeholder with preloaded data so it can be used as an
        operand if referenced by operators by name.
        """
        np_array = tensor.numpy()
        if np_array.ndim == 1:
            dimensions = [int(np_array.shape[0])]
            channels = 1
        elif np_array.ndim == 2:
            dimensions = [int(x) for x in np_array.shape[:2]]
            channels = 1
        elif np_array.ndim == 3:
            dimensions = [int(x) for x in np_array.shape[:2]]
            channels = int(np_array.shape[2])
        else:
            raise NotImplementedError

        entry = {
            "dimensions": dimensions,
            "channels": channels,
            "data_type": convert_from_dtype(np_array.dtype),
            "is_placeholder": False,
            "usage": 6,
            # put a generic MAT flag derived from dtype+channels
            "flag": int(smr.BaseType.MAT) | (int(smr.BaseType.CHANNEL_MASK) & channels) | int(entry["data_type"]) if False else 0,
        }
        if with_value:
            entry["data"] = [float(x) for x in np_array.flatten()]
        self.spec["tensors"][name] = entry


    def set_inputs(self, inputs: Union[int, str, Iterable[Union[int, str]]]):
        items = as_list(inputs)
        names: List[str] = []
        for x in items:
            if isinstance(x, str):
                names.append(x)
            else:
                names.append(self._tensor_id_to_name.get(int(x), f"tensor_{int(x)}"))
        self.spec["inputs"] = names
        self._normalize_placeholders()

    def set_outputs(self, outputs: Union[int, str, Iterable[Union[int, str]]]):
        items = as_list(outputs)
        names: List[str] = []
        for x in items:
            if isinstance(x, str):
                names.append(x)
            else:
                names.append(self._tensor_id_to_name.get(int(x), f"tensor_{int(x)}"))
        self.spec["outputs"] = names
        self._normalize_placeholders()

    def set_tensor_name(self, tensor: Union[int, str], new_name: str) -> None:
        """Rename a recorded tensor for readability.

        Accepts a tensor id (placeholder/local id) or current name.
        Updates spec["tensors"] keys, tensor descriptor "name", operator wiring and IO lists.
        """
        if isinstance(tensor, str):
            old_name = tensor
            # try find id for mapping
            tid = None
            for k, v in self.spec["tensors"].items():
                if k == old_name:
                    # attempt: find id via reverse lookup in _tensor_id_to_name
                    break
            # best effort; if name not found, nothing to do
            if old_name not in self.spec["tensors"]:
                return
        else:
            tid = int(tensor)
            old_name = self._tensor_id_to_name.get(tid, f"tensor_{tid}")

        if old_name == new_name:
            return
        if new_name in self.spec["tensors"] and new_name != old_name:
            raise ValueError(f"Tensor name already exists: {new_name}")

        # move tensor descriptor
        desc = self.spec["tensors"].pop(old_name, None)
        if desc is None:
            # create an empty entry if missing (shouldn't happen in normal flow)
            desc = {}
        self.spec["tensors"][new_name] = desc

        # update id->name map
        if isinstance(tensor, int):
            self._tensor_id_to_name[int(tensor)] = new_name

        # update operator wirings
        for opspec in self.spec["operators"]:
            opspec["inputs"] = [new_name if x == old_name else x for x in opspec.get("inputs", [])]
            opspec["outputs"] = [new_name if x == old_name else x for x in opspec.get("outputs", [])]

        # update IO lists
        self.spec["inputs"] = [new_name if x == old_name else x for x in self.spec.get("inputs", [])]
        self.spec["outputs"] = [new_name if x == old_name else x for x in self.spec.get("outputs", [])]

    def set_tensor_data(self, tensor: Union[int, str], np_array: np.ndarray, mark_non_placeholder: bool = True) -> None:
        """Attach concrete data to an existing recorded tensor.

        If mark_non_placeholder is True, sets is_placeholder=False in the spec.
        """
        if isinstance(tensor, str):
            name = tensor
        else:
            name = self._tensor_id_to_name.get(int(tensor), f"tensor_{int(tensor)}")
        if name not in self.spec["tensors"]:
            raise KeyError(f"Tensor not found in spec: {name}")

        if np_array.ndim == 1:
            dimensions = [int(np_array.shape[0])]
            channels = 1
        elif np_array.ndim == 2:
            dimensions = [int(x) for x in np_array.shape[:2]]
            channels = 1
        elif np_array.ndim == 3:
            dimensions = [int(x) for x in np_array.shape[:2]]
            channels = int(np_array.shape[2])
        else:
            raise NotImplementedError

        entry = self.spec["tensors"][name]
        entry["dimensions"] = dimensions
        entry["channels"] = channels
        entry["data_type"] = convert_from_dtype(np_array.dtype)
        if mark_non_placeholder:
            entry["is_placeholder"] = False
        entry["usage"] = entry.get("usage", 6)
        # compute MAT flag from dtype/channels if not set
        if not entry.get("flag"):
            entry["flag"] = int(smr.BaseType.MAT) | (int(smr.BaseType.CHANNEL_MASK) & channels) | int(entry["data_type"])
        entry["data"] = [float(x) for x in np_array.flatten()]

    def _normalize_placeholders(self) -> None:
        """Ensure only inputs/outputs are placeholders; others are local tensors."""
        io = set(self.spec.get("inputs", [])) | set(self.spec.get("outputs", []))
        for name, desc in self.spec["tensors"].items():
            desc["is_placeholder"] = name in io

    def save(self, file_path):
        # Normalize placeholder flags before saving
        self._normalize_placeholders()
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.spec, f, indent=2, ensure_ascii=False)


def add_vst_operator(pipeline, replace_pair):
    """Append a VST access operator to the saved spec and rewire input.

    Args:
        pipeline: an instance of Pipeline (the extended smr.Pipeline).
        replace_pair: tuple(old_name, new_name). All occurrences of the
            old tensor name in existing operators are replaced with the new
            name. The VST operator will produce the new_name as its left RGB
            output alongside additional standard outputs.

    Notes:
        - This augments only the serialized JSON spec and does not add a
          runnable operator to the underlying smr.Pipeline graph.
        - Output names roughly follow the reference in mnist_inference_pipeline.json.
    """
    if not hasattr(pipeline, "spec"):
        raise TypeError("pipeline must be a Pipeline instance with a .spec")

    old_name, new_name = replace_pair
    spec = pipeline.spec

    # Choose output names. Keep the user-provided left image name and add companions.
    right_name = "right_rgb"
    left_name = str(new_name)
    ts_name = "timestamp_tensor"
    cam_mat_name = "camera_matrix_tensor"

    # Insert the VST access operator at the front to mirror a source op.
    vst_op = {
        "type": "camera_access",  # alias for RECTIFIED_VST_ACCESS
        "inputs": [],
        "outputs": [right_name, left_name, ts_name, cam_mat_name],
    }
    spec["operators"].insert(0, vst_op)

    # Ensure tensor descriptors exist for new outputs, inferring from the old input if possible.
    tensors = spec.setdefault("tensors", {})
    base = tensors.get(old_name, {})
    # Fall back sizes if original input not recorded
    dims = list(base.get("dimensions", [])) or [2464, 3248]

    def _ensure_tensor(name: str, channels: int, data_type: int, dimensions=None, is_placeholder=False, usage=6):
        if name in tensors:
            return
        tensors[name] = {
            "dimensions": list(dimensions) if dimensions is not None else list(dims),
            "channels": int(channels),
            "data_type": int(data_type),
            "is_placeholder": bool(is_placeholder),
            "usage": int(usage),
        }

    # right/left RGB as uint8 3-channel mats
    _ensure_tensor(right_name, channels=3, data_type=convert_from_dtype(np.uint8), dimensions=dims, is_placeholder=False, usage=6)
    _ensure_tensor(left_name, channels=3, data_type=convert_from_dtype(np.uint8), dimensions=dims, is_placeholder=False, usage=6)
    # timestamp and camera matrix tensors (float32)
    _ensure_tensor(ts_name, channels=4, data_type=convert_from_dtype(np.int32), dimensions=[1], is_placeholder=False, usage=5)
    _ensure_tensor(cam_mat_name, channels=1, data_type=convert_from_dtype(np.float32), dimensions=[3, 3], is_placeholder=False, usage=6)

    # Rewire existing operator references from old_name -> left_name
    for op in spec.get("operators", []):
        ins = op.get("inputs", [])
        outs = op.get("outputs", [])
        op["inputs"] = [left_name if x == old_name else x for x in ins]
        op["outputs"] = [left_name if x == old_name else x for x in outs]

    # Update IO declaration: remove old input placeholder since now fed by camera op
    if "inputs" in spec:
        spec["inputs"] = [x for x in as_list(spec["inputs"]) if x != old_name]
    tensors.pop(old_name)


def add_model_inference_operator(
    pipeline,
    *,
    context_binary_file: str,
    model_name: str,
    model_input: List[Dict],
    model_output: List[Dict],
    model_output_tensor_info: List[Dict],
):
    """Append a model inference operator to the saved spec.

    The operator type is recorded as "run_model_inference" with model metadata
    fields. Inputs/outputs are simple tensor name lists for loader compatibility.

    Args:
        pipeline: Pipeline to augment (serialization spec only).
        context_binary_file: Model asset filename (e.g., "mnist.serialized.bin").
        model_name: Logical model name.
        model_input: Iterable of input tensor names.
        model_output: Iterable of output tensor names.
    """
    if not hasattr(pipeline, "spec"):
        raise TypeError("pipeline must be a Pipeline instance with a .spec")

    spec = pipeline.spec

    # Append operator entry
    op = {
        "type": "run_algorithm",
        "inputs": model_input,
        "outputs": model_output,
        "model_asset": str(context_binary_file),
        "model_name": str(model_name),
    }
    spec.setdefault("operators", []).append(op)

    # Ensure output tensors exist and are marked as placeholders (downstream IO)
    tensors = spec.setdefault("tensors", {})
    for output, info in zip(model_output, model_output_tensor_info):
        name = output["tensor"]
        assert name not in tensors
        tensors[name] = {
            "dimensions": info["dimensions"],
            "channels": info["channels"],
            "data_type": convert_from_dtype(info["data_type"]),
            "is_placeholder": True,
            "usage": 2,
        }
        if "outputs" in spec:
            spec["outputs"].append(name)

    for input_ in model_input:
        name = input_["tensor"]
        if name in tensors:
            tensors[name]["is_placeholder"] = False
        if "inputs" in spec and name in spec["outputs"]:
            spec["outputs"].remove(name)


class DeserializedPipeline:
    """Load a Pipeline.save JSON and run it (mnist-style keys)."""
    def __init__(self, json_or_path: Union[str, Dict[str, Any]]):
        if isinstance(json_or_path, str) and os.path.exists(json_or_path):
            with open(json_or_path, 'r', encoding='utf-8') as f:
                self.pipeline_spec = json.load(f)
        else:
            self.pipeline_spec = json_or_path

        self.pipeline = smr.Pipeline()
        self._name_to_id: Dict[str, int] = {}
        self._inputs_names: List[str] = as_list(self.pipeline_spec.get("inputs", []))
        self._outputs_names: List[str] = as_list(self.pipeline_spec.get("outputs", []))

        self._build_graph()
        self.placeholder_map: Dict[int, smr.Tensor] = {}
        self._create_backing_tensors()


    def _build_graph(self) -> None:
        tensors = self.pipeline_spec.get("tensors", {})
        for name, t in tensors.items():
            dims = list(t.get("dimensions", []))
            channels = int(t.get("channels", 1))
            data_type_val = int(t.get("data_type", 0))
            flag = int(t.get("flag", (data_type_val | int(smr.BaseType.MAT) | (int(smr.BaseType.CHANNEL_MASK) & channels))))
            # Only tensors listed in inputs/outputs are placeholders; others are local
            if name in self._inputs_names or name in self._outputs_names:
                new_id = int(self.pipeline.allocate_placeholder(dims, flag))
            else:
                new_id = int(self.pipeline.allocate_local_tensor(dims, flag))
            self._name_to_id[name] = new_id

        for op in self.pipeline_spec.get("operators", []):
            type_name = op.get("type")
            attrs: List[str] = []
            if "flag" in op:
                attrs = [str(op.get("flag"))]
            if "expression" in op:
                attrs = [str(op.get("expression"))]
            if str(type_name).lower() == "nms" and "threshold" in op:
                attrs = [str(op.get("threshold"))]
            if str(type_name).lower() == "sort_mat":
                # mode can be COLUMN/ROW etc.; default to COLUMN if provided
                mode = op.get("mode") or op.get("axis")
                if mode is not None:
                    attrs = [str(mode)]
            oid = self.pipeline.allocate_operator(name_to_type(type_name), attrs)
            proxy = self.pipeline.query_operator(oid)
            for idx, name in enumerate(op.get("inputs", [])):
                if not name:
                    continue
                tid = self._name_to_id[name]
                proxy.data_as_operand(self.pipeline.query_local_tensor(tid), idx)
            for idx, name in enumerate(op.get("outputs", [])):
                if not name:
                    continue
                tid = self._name_to_id[name]
                proxy.connect_result_to_data_array(idx, self.pipeline.query_local_tensor(tid))

            # Special handling: ASSIGNMENT slices wiring
            if str(type_name).lower() == "assignment":
                src_slices = op.get("src_slices")
                dst_slices = op.get("dst_slices")
                if src_slices is not None:
                    # Build vec2 list with [[row_start,row_end],[col_start,col_end]]
                    arr = np.array(src_slices, dtype=np.int32)
                    vec = smr.TensorFactory.create([2], int(smr.EDataType.INT32) | smr.BaseType.VEC_2)
                    if hasattr(vec, 'load_from_raw_byte_arrays'):
                        vec.load_from_raw_byte_arrays(np.ascontiguousarray(arr).tobytes())
                    proxy.data_as_operand(vec, 1)
                if dst_slices is not None:
                    arr = np.array(dst_slices, dtype=np.int32)
                    vec = smr.TensorFactory.create([2], int(smr.EDataType.INT32) | smr.BaseType.VEC_2)
                    if hasattr(vec, 'load_from_raw_byte_arrays'):
                        vec.load_from_raw_byte_arrays(np.ascontiguousarray(arr).tobytes())
                    proxy.data_as_operand(vec, 3)

                # Support dynamic slice tensors referenced by name
                src_slices_tensor = op.get("src_slices_tensor")
                if src_slices_tensor is not None:
                    tid = self._name_to_id[src_slices_tensor]
                    proxy.data_as_operand(self.pipeline.query_local_tensor(tid), 1)
                dst_slices_tensor = op.get("dst_slices_tensor")
                if dst_slices_tensor is not None:
                    tid = self._name_to_id[dst_slices_tensor]
                    proxy.data_as_operand(self.pipeline.query_local_tensor(tid), 3)

    def _create_backing_tensors(self) -> None:
        tensors = self.pipeline_spec.get("tensors", {})
        for name, t in tensors.items():
            tid = self._name_to_id[name]
            dims = list(t.get("dimensions", []))
            flag = int(t.get("flag", 0))
            if flag == 0:
                channels = int(t.get("channels", 1))
                data_type_val = int(t.get("data_type", 0))
                flag = int(data_type_val) | int(smr.BaseType.MAT) | (int(smr.BaseType.CHANNEL_MASK) & channels)

            if name in self._inputs_names or name in self._outputs_names:
                # placeholders get real Tensor instances in placeholder_map
                if flag & int(smr.BaseType.MAT):
                    tensor = smr.TensorFactory.create(dims, flag)
                elif flag & int(smr.BaseType.POINT_2):
                    n = dims[0] if dims else 0
                    tensor = smr.TensorPoint2Float.from_numpy(np.zeros((n, 2), dtype=np.float32))
                else:
                    tensor = smr.TensorFactory.create(dims, int(smr.EDataType.FLOAT32) | smr.BaseType.MAT | (smr.BaseType.CHANNEL_MASK & 1))
                # accept both legacy "value" and canonical "data" fields
                data = t.get("data")
                if data is None:
                    data = t.get("value")
                if data is not None and hasattr(tensor, 'load_from_raw_byte_arrays'):
                    dt = int(t.get("data_type", 0))
                    # POINT_2 tensors are always float32 regardless of recorded data_type
                    if flag & int(smr.BaseType.POINT_2):
                        np_dtype = np.float32
                    else:
                        np_dtype = convert_to_dtype(dt, target="numpy")
                    channels = int(t.get("channels", 1))
                    shape = dims + [channels] if channels > 1 else dims
                    np_arr = np.array(data, dtype=np_dtype)
                    try:
                        np_arr = np_arr.reshape(shape)
                    except Exception:
                        pass
                    tensor.load_from_raw_byte_arrays(np.ascontiguousarray(np_arr).tobytes())
                self.placeholder_map[int(tid)] = tensor
            else:
                data = t.get("data")
                if data is None:
                    data = t.get("value")
                if data is not None:
                    lt = self.pipeline.query_local_tensor(tid)
                    dt = int(t.get("data_type", 0))
                    np_dtype = convert_to_dtype(dt, target="numpy")
                    channels = int(t.get("channels", 1))
                    shape = dims + [channels] if channels > 1 else dims
                    np_arr = np.array(data, dtype=np_dtype)
                    try:
                        np_arr = np_arr.reshape(shape)
                    except Exception:
                        pass
                    if hasattr(lt, 'load_from_raw_byte_arrays'):
                        lt.load_from_raw_byte_arrays(np.ascontiguousarray(np_arr).tobytes())

    def __call__(self, inputs: Union[smr.Tensor, np.ndarray, Dict[Union[str, int], Union[smr.Tensor, np.ndarray]]]):
        ph_map: Dict[int, smr.Tensor] = dict(self.placeholder_map)

        def _assign(target_tid: int, value: Union[smr.Tensor, np.ndarray]):
            if isinstance(value, smr.Tensor):
                ph_map[target_tid] = value
            elif isinstance(value, np.ndarray):
                t = ph_map[target_tid]
                t.load_from_raw_byte_arrays(np.ascontiguousarray(value).tobytes())
            else:
                raise TypeError("Unsupported input type; must be smr.Tensor or numpy.ndarray")

        if isinstance(inputs, dict):
            for k, v in inputs.items():
                if isinstance(k, str):
                    tid = self._name_to_id[k]
                else:
                    tid = int(k)
                _assign(tid, v)
        else:
            if len(self._inputs_names) != 1:
                raise ValueError("Multiple inputs expected; provide a dict name->tensor.")
            tid = self._name_to_id[self._inputs_names[0]]
            _assign(tid, inputs)

        task = smr.Task(self.pipeline, ph_map, 0, None)
        task.verify_all_place_holder_contained()
        task.setup_place_holder_mapping()
        pool = smr.ThreadPool2()
        pool.enqueue(task)

        import time
        for _ in range(200):
            if not self.pipeline.cannot_modified():
                break
            time.sleep(0.01)

        if not self._outputs_names:
            return ph_map
        outs: List[smr.Tensor] = []
        for name in self._outputs_names:
            tid = self._name_to_id[name]
            outs.append(ph_map[tid])
        return outs[0] if len(outs) == 1 else outs
