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

import contextlib
import json
import os
import typing
import time
import uuid
from typing import Dict, List, Any, Optional, Union, Iterable

import numpy as np
import securemr as smr
import re
from securemr.operators import custom_operator as custom_ops

from .utils import (
    convert_from_dtype,
    convert_to_dtype,
    mat_flag,
    normalize_qnn_dtype,
    numpy_dtype_to_smr,
    unmat_flag,
)

__all__ = [
    "Pipeline",
    "DeserializedPipeline",
    "add_vst_operator",
    "add_model_inference_operator",
    "convert_python_custom_to_run_algorithm",
    "convert_run_algorithm_to_python_custom",
]

SPEC_VERSION = "1.0.0"
_OP_ENUM_PREFIX = "XR_SECURE_MR_OPERATOR_TYPE_"
_OP_ENUM_SUFFIX = "_PICO"

def _extract_qnn_io_from_metadata(data: Any) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    if not isinstance(data, dict):
        return None

    def _collect(entries: Optional[List[Any]]) -> List[Dict[str, Any]]:
        collected: List[Dict[str, Any]] = []
        for entry in entries or []:
            info = entry.get("info") if isinstance(entry, dict) and "info" in entry else entry
            if not isinstance(info, dict):
                continue
            dtype_value = info.get("encoding_type") or info.get("dataType") or info.get("dtype")
            collected.append(
                {
                    "name": info.get("name"),
                    "dimensions": info.get("dimensions") or info.get("shape"),
                    "numpy_dtype": normalize_qnn_dtype(dtype_value),
                    "raw_dtype": dtype_value,
                }
            )
        return collected

    info = data.get("info")
    if isinstance(info, dict):
        graphs = info.get("graphs")
        if isinstance(graphs, list) and graphs:
            graph_info = graphs[0]
            graph = graph_info.get("info") if isinstance(graph_info, dict) and "info" in graph_info else graph_info
            if isinstance(graph, dict):
                inputs = _collect(graph.get("graphInputs"))
                outputs = _collect(graph.get("graphOutputs"))
                if inputs or outputs:
                    return {"inputs": inputs, "outputs": outputs}

    if "input" in data or "output" in data:
        inputs = _collect(data.get("input"))
        outputs = _collect(data.get("output"))
        if inputs or outputs:
            return {"inputs": inputs, "outputs": outputs}

    return None


def _load_qnn_metadata(model_path: str) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    if not model_path:
        return None
    abs_path = os.path.abspath(model_path)
    parent = os.path.dirname(abs_path)
    base = os.path.basename(abs_path)
    stem, ext = os.path.splitext(base)

    candidates = {
        abs_path + ".json",
        os.path.join(parent, base + ".json"),
        os.path.join(parent, base + ".bin.json"),
        os.path.join(parent, stem + ".json"),
        os.path.join(parent, stem + ".bin.json"),
        os.path.join(parent, f"{base}.json"),
        os.path.join(parent, "model.serialized.bin.json"),
        os.path.join(parent, "model_info.json"),
    }

    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        io_info = _extract_qnn_io_from_metadata(data)
        if io_info:
            return io_info
    return None


def qnn_dimensions_to_smr(qnn_dimensions):
    """Convert qnn_dimensions (BHWC) to (W,H) and C."""
    if len(qnn_dimensions) == 4:
        B, H, W, C = qnn_dimensions
        assert B == 1
        return [W, H], C
    elif len(qnn_dimensions) == 3:
        B, HW, C = qnn_dimensions
        return [HW, C], 1
    elif len(qnn_dimensions) == 2:
        B, HW = qnn_dimensions
        return [HW, 1], 1
    elif len(qnn_dimensions) == 2:
        return [1, 1], 1


def _ensure_tensor_from_q_info(spec: Dict[str, Any], tensor_name: str, q_info: Optional[Any]) -> None:
    numpy_dtype = q_info.get("numpy_dtype")
    if spec is None or numpy_dtype is None:
        return
    tensors = spec.get("tensors")
    if not isinstance(tensors, dict):
        return

    if "tensor_name" not in tensors:
        print(f"{tensor_name} not found in spec, create one")
        dimensions, channels = qnn_dimensions_to_smr(q_info.get("dimensions"))
        tensors[tensor_name] = {
                "dimensions": dimensions,
                "channels": channels,
                "is_placeholder": False}
    tensor_desc = tensors.get(tensor_name)
    if "usage" not in tensor_desc:
        tensor_desc["usage"] = 6
    if "value" not in tensor_desc:
        tensor_desc["value"] = None

    try:
        canonical = np.dtype(numpy_dtype).type
    except TypeError:
        return

    if canonical is np.float16:
        canonical = np.float32 
    dtype_idx = convert_from_dtype(canonical)

    tensor_desc["data_type"] = dtype_idx
    channels = tensor_desc.get("channels")
    dtype_enum = numpy_dtype_to_smr(canonical)
    if channels is not None and dtype_enum is not None:
        tensor_desc["flag"] = mat_flag(dtype_enum, int(channels))


def _extract_custom_token_from_attrs(attrs: Optional[List[Any]]) -> Optional[str]:
    for attr in attrs or []:
        if isinstance(attr, str) and attr.startswith("token:"):
            return attr.split(":", 1)[1]
    return None


def _select_qnn_outputs(
    all_outputs: Optional[List[Dict[str, Any]]],
    active_names: Optional[List[str]],
) -> Optional[List[Dict[str, Any]]]:
    if all_outputs is None:
        return None
    if not active_names:
        return list(all_outputs)
    mapping = {
        str(entry.get("name")): entry
        for entry in all_outputs
        if isinstance(entry, dict) and entry.get("name") is not None
    }
    ordered: List[Dict[str, Any]] = []
    for name in active_names:
        info = mapping.get(str(name))
        if info is not None:
            ordered.append(info)
    if ordered:
        return ordered
    return list(all_outputs)


def _build_io_entries(
    tensor_names: List[str],
    qnn_infos: Optional[List[Dict[str, Any]]],
    spec: Dict[str, Any],
) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    for idx, tensor_name in enumerate(tensor_names):
        q_info = qnn_infos[idx] if qnn_infos and idx < len(qnn_infos) else None
        q_name = q_info.get("name") if q_info and q_info.get("name") else tensor_name
        entries.append({"name": str(q_name), "tensor": str(tensor_name)})
        if q_info:
            _ensure_tensor_from_q_info(spec, tensor_name, q_info)
    return entries

def as_list(data):
    if isinstance(data, (list, tuple)):
        return list(data)
    elif data is None:
        return []
    else:
        return data


def _operator_members() -> Dict[str, Any]:
    members: Dict[str, Any] = {}
    for attr in dir(smr.EOperatorType):
        if attr.startswith("__"):
            continue
        try:
            val = getattr(smr.EOperatorType, attr)
            _ = int(val)
        except Exception:
            continue
        members[attr.upper()] = val
        members["XR_SECURE_MR_OPERATOR_TYPE_" + attr.upper() + "_PICO"] = val
    return members


def _normalize_operator_token(token: str) -> str:
    norm = re.sub(r"[^0-9A-Za-z]+", "_", token).upper()
    if norm.startswith(_OP_ENUM_PREFIX):
        norm = norm[len(_OP_ENUM_PREFIX) :]
    if norm.endswith(_OP_ENUM_SUFFIX):
        norm = norm[: -len(_OP_ENUM_SUFFIX)]
    return norm


def name_to_type(name: Union[str, int, smr.EOperatorType]) -> smr.EOperatorType:
    members = _operator_members()

    if isinstance(name, smr.EOperatorType):
        return name

    if isinstance(name, int):
        val = int(name)
        for candidate in members.values():
            try:
                if int(candidate) == val:
                    return candidate
            except Exception:
                continue
        try:
            return smr.EOperatorType(val)
        except Exception:
            return val

    s = str(name).strip()
    if not s:
        raise KeyError("Operator type name cannot be empty.")

    # Handle explicit unknown_X tokens
    unknown_match = re.fullmatch(r"unknown_(\d+)", s, flags=re.IGNORECASE)
    if unknown_match:
        return name_to_type(int(unknown_match.group(1)))
    if s.isdigit():
        return name_to_type(int(s))

    alias: Dict[str, str] = {
        "arithmetic": "ARITHMETIC_COMPOSE",
        "camera_access": "RECTIFIED_VST_ACCESS",
        "cvt_color": "CONVERT_COLOR",
        "js_scripting": "JS_SCRIPTING",
        "run_algorithm": "RUN_MODEL_INFERENCE",
        "type_convert": "ASSIGNMENT",
    }

    lookup_key = alias.get(s.lower())
    if lookup_key is None:
        lookup_key = _normalize_operator_token(s)

    if lookup_key in members:
        return members[lookup_key]

    # Case-insensitive search as a final attempt
    for key, value in members.items():
        if key.upper() == lookup_key.upper():
            return value
    raise KeyError(f"Unsupported operator type name: {name}")


def type_to_name(op_type: smr.EOperatorType) -> str:
    """Return the canonical JSON `type` token for a SecureMR operator."""
    try:
        key_val = int(op_type)
    except Exception:
        key_val = int(op_type)

    value_to_name: Dict[int, str] = {}
    for attr in dir(smr.EOperatorType):
        if attr.startswith("__"):
            continue
        try:
            value_to_name[int(getattr(smr.EOperatorType, attr))] = attr
        except Exception:
            continue

    enum_name = value_to_name.get(key_val)
    if not enum_name:
        return f"unknown_{key_val}"

    return f"{_OP_ENUM_PREFIX}{enum_name}{_OP_ENUM_SUFFIX}"


def _ensure_list_size(lst: List[typing.Optional[int]], size: int) -> None:
    if len(lst) < size:
        lst.extend([None] * (size - len(lst)))


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
            op_entry["attrs"] = [str(a) for a in attrs]
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
        "type": type_to_name(smr.EOperatorType.RECTIFIED_VST_ACCESS),
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
        "type": type_to_name(smr.EOperatorType.RUN_MODEL_INFERENCE),
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


def convert_python_custom_to_run_algorithm(
    pipeline: Union["Pipeline", Dict[str, Any]],
    *,
    model_path: Optional[str] = None,
    extras: Optional[Dict] = None,
) -> bool:
    """Rewrite python_custom operators to run_algorithm entries in the pipeline spec.

    Args:
        pipeline: Serializable pipeline instance or raw spec dictionary to mutate.
        model_path: Optional model path used to fill asset/file metadata.

    Returns:
        True if at least one operator was rewritten, otherwise False.
    """
    if pipeline is None:
        return False

    if hasattr(pipeline, "spec"):
        spec = pipeline.spec
    elif isinstance(pipeline, dict):
        spec = pipeline
    else:
        raise TypeError("pipeline must be a SerializablePipeline or spec dictionary.")

    if spec is None:
        return False

    ops = spec.get("operators") or []
    if not isinstance(ops, list) or not ops:
        return False

    if not model_path:
        raise ValueError("model_path is required.")
    abs_model_path = model_path if os.path.isabs(model_path) else os.path.abspath(model_path)
    context_file = os.path.basename(abs_model_path)
    model_name = context_file.split(".")[0] or "model"
    is_qnn_model = context_file.lower().endswith(".bin")
    qnn_metadata = _load_qnn_metadata(abs_model_path) if is_qnn_model else None
    if is_qnn_model and qnn_metadata is None:
        raise FileNotFoundError(f"Unable to locate QNN metadata for model: {abs_model_path}")
    qnn_inputs_all = qnn_metadata.get("inputs") if qnn_metadata else None
    qnn_outputs_all = qnn_metadata.get("outputs") if qnn_metadata else None

    def _resolve_names(entries: Optional[List[Any]]) -> List[str]:
        names: List[str] = []
        for entry in entries or []:
            if isinstance(entry, dict):
                tensor_name = entry.get("tensor") or entry.get("name")
            else:
                tensor_name = entry
            if tensor_name:
                names.append(str(tensor_name))
        return names

    replaced = False
    for op in ops:
        op_enum: Optional[smr.EOperatorType]
        try:
            op_enum = name_to_type(op.get("type"))
        except Exception:
            op_enum = None
        if op_enum is None:
            if str(op.get("type", "")).strip().lower() != "custom":
                continue
        elif int(op_enum) != int(smr.EOperatorType.PYTHON_CUSTOM):
            continue

        input_tensors = _resolve_names(op.get("inputs"))
        output_tensors = _resolve_names(op.get("outputs"))

        token = _extract_custom_token_from_attrs(op.get("attrs"))
        impl = custom_ops.get_registered_custom_operator(token) if token else None

        active_outputs: Optional[List[str]] = None
        if impl is not None and hasattr(impl, "_model"):
            output_ids = getattr(impl._model, "output_node_ids", None)
            if isinstance(output_ids, str):
                active_outputs = [output_ids]
            elif isinstance(output_ids, (list, tuple)):
                active_outputs = [str(x) for x in output_ids]

        qnn_inputs = qnn_inputs_all
        qnn_outputs = _select_qnn_outputs(qnn_outputs_all, active_outputs)

        if qnn_inputs is not None and len(qnn_inputs) != len(input_tensors):
            raise ValueError(
                f"QNN model input count ({len(qnn_inputs)}) does not match pipeline tensors ({len(input_tensors)})."
            )
        if qnn_outputs is not None and len(qnn_outputs) != len(output_tensors):
            if "output_tensors" in extras:
                output_tensors = extras["output_tensors"]
            else:
                raise ValueError(
                    f"QNN model output count ({len(qnn_outputs)}) does not match pipeline tensors ({len(output_tensors)})."
                )

        op["type"] = type_to_name(smr.EOperatorType.RUN_MODEL_INFERENCE)
        op["inputs"] = _build_io_entries(input_tensors, qnn_inputs, spec)
        op["outputs"] = _build_io_entries(output_tensors, qnn_outputs, spec)
        op.pop("attrs", None)
        op["model_name"] = model_name

        if is_qnn_model:
            op["model_asset"] = context_file
            op.pop("model_file", None)
        else:
            op["model_file"] = abs_model_path
            op.pop("model_asset", None)

        replaced = True

    return replaced


def convert_run_algorithm_to_python_custom(
    pipeline: Union["Pipeline", Dict[str, Any]],
) -> Dict[str, Any]:
    """Convert run_algorithm operators to python_custom equivalents.

    This is primarily used when loading a serialized pipeline on the host, where
    the native run_algorithm operator is unavailable. Each run_algorithm entry is
    replaced by a python_custom operator backed by ModelInferenceOperator.
    """
    if pipeline is None:
        return pipeline

    if hasattr(pipeline, "spec"):
        spec = pipeline.spec
    elif isinstance(pipeline, dict):
        spec = pipeline
    else:
        raise TypeError("pipeline must be a Pipeline instance or a spec dictionary.")

    if spec is None:
        return spec

    operators = spec.get("operators")
    if not isinstance(operators, list) or not operators:
        return spec

    metadata = spec.get("metadata", {}) or {}

    # Late import to avoid circular dependency at module import time.
    from securemr.operators.inference_operator import ModelInferenceOperator

    def _extract_tensor_info(entries: Optional[List[Any]]) -> tuple[List[str], List[str]]:
        tensor_names: List[str] = []
        logical_names: List[str] = []
        for entry in entries or []:
            tensor_name: Optional[str]
            logical_name: Optional[str]
            if isinstance(entry, dict):
                tensor_name = entry.get("tensor") or entry.get("name")
                logical_name = entry.get("name") or tensor_name
            else:
                tensor_name = entry
                logical_name = entry
            if tensor_name:
                tensor_names.append(str(tensor_name))
                logical_names.append(str(logical_name) if logical_name else str(tensor_name))
        return tensor_names, logical_names

    def _candidate_directories(op_dict: Dict[str, Any]) -> List[str]:
        dirs: List[str] = []
        op_dir = op_dict.get("model_dir")
        if op_dir:
            dirs.append(str(op_dir))
        for key in ("model_dir", "base_dir", "assets_dir"):
            val = metadata.get(key)
            if val:
                dirs.append(str(val))
        env_dir = os.getenv("PYSECUREMR_ASSET_DIR")
        if env_dir:
            dirs.append(env_dir)
        return dirs

    def _resolve_model_path(op_dict: Dict[str, Any]) -> str:
        candidates: List[str] = []
        for key in ("model_file", "model_path"):
            path = op_dict.get(key)
            if path:
                candidates.append(str(path))
        for key in ("model", "model_asset"):
            path = op_dict.get(key)
            if path:
                candidates.append(str(path))

        search_dirs = _candidate_directories(op_dict)

        for path in candidates:
            if os.path.isabs(path):
                return path
            for root in search_dirs:
                full = os.path.abspath(os.path.join(root, path))
                if os.path.exists(full):
                    return full

        if candidates:
            # Fall back to the first candidate (absolute or relative).
            path = candidates[0]
            if not os.path.isabs(path) and search_dirs:
                return os.path.abspath(os.path.join(search_dirs[0], path))
            return os.path.abspath(path)

        raise ValueError("Unable to resolve model path for run_algorithm operator.")

    for op in operators:
        op_enum: Optional[smr.EOperatorType]
        try:
            op_enum = name_to_type(op.get("type"))
        except Exception:
            op_enum = None
        if op_enum is None:
            if str(op.get("type", "")).strip().lower() != "run_algorithm":
                continue
        elif int(op_enum) != int(smr.EOperatorType.RUN_MODEL_INFERENCE):
            continue

        input_tensors, operand_names = _extract_tensor_info(op.get("inputs"))
        output_tensors, result_names = _extract_tensor_info(op.get("outputs"))

        if not input_tensors:
            raise ValueError("run_algorithm operator must provide at least one input tensor.")
        if not operand_names:
            operand_names = input_tensors
        if not result_names:
            result_names = output_tensors if output_tensors else ["output"]

        model_path = _resolve_model_path(op)
        convert_output_dir = op.get("convert_output_dir")
        output_node_ids = op.get("output_node_ids")

        custom_impl = ModelInferenceOperator(
            model_path=model_path,
            device="auto",
            convert_output_dir=convert_output_dir,
            onnx_to_qnn=False,
            operand_names=operand_names,
            result_names=result_names,
            output_node_ids=output_node_ids,
        )
        registry = getattr(custom_ops, "_TOKEN_TO_IMPLEMENTATION", None)
        if registry is None:
            raise RuntimeError("Custom operator registry is unavailable.")
        token = f"run-algorithm-{uuid.uuid4().hex}"
        while token in registry:
            token = f"run-algorithm-{uuid.uuid4().hex}"
        registry[token] = custom_impl

        op.clear()
        op["type"] = type_to_name(smr.EOperatorType.PYTHON_CUSTOM)
        op["attrs"] = [f"token:{token}"]
        op["inputs"] = input_tensors
        op["outputs"] = output_tensors

    return spec


class DeserializedPipeline:
    """Load a Pipeline.save JSON and run it (mnist-style keys).

    Args:
        json_or_path: JSON dictionary or path to the serialized pipeline.
    """

    def __init__(
        self,
        json_or_path: Union[str, Dict[str, Any]],
    ):
        if isinstance(json_or_path, str) and os.path.exists(json_or_path):
            with open(json_or_path, 'r', encoding='utf-8') as f:
                self.pipeline_spec = json.load(f)
        else:
            self.pipeline_spec = json_or_path

        self.pipeline = smr.Pipeline()
        self._name_to_id: Dict[str, int] = {}
        self._inputs_names: List[str] = as_list(self.pipeline_spec.get("inputs", []))
        self._outputs_names: List[str] = as_list(self.pipeline_spec.get("outputs", []))
        self._custom_operator_handles: List[custom_ops.CustomOperatorHandle] = []

        self._build_graph()
        self.placeholder_map: Dict[int, smr.Tensor] = {}
        self._create_backing_tensors()

    @staticmethod
    def _resolve_tensor_ref(ref: Any) -> Optional[str]:
        if isinstance(ref, str):
            return ref if ref else None
        if isinstance(ref, dict):
            tensor = ref.get("tensor")
            if tensor:
                return str(tensor)
            name = ref.get("name")
            if name:
                return str(name)
            return None
        if ref is None:
            return None
        return str(ref)

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
        
        # run_algorithm operator not supported on host/python, replace it with python_custom
        # reverse of convert_python_custom_to_run_algorithm
        self.pipeline_spec = convert_run_algorithm_to_python_custom(self.pipeline_spec)

        for op_idx, op in enumerate(self.pipeline_spec.get("operators", [])):
            type_name = op.get("type")
            op_type = name_to_type(type_name)
            attrs = self._prepare_operator_attrs(op, op_type)
            oid = self.pipeline.allocate_operator(op_type, attrs)
            proxy = self.pipeline.query_operator(oid)
            for idx, ref in enumerate(op.get("inputs", [])):
                tensor_name = self._resolve_tensor_ref(ref)
                if not tensor_name:
                    continue
                if tensor_name not in self._name_to_id:
                    raise KeyError(f"Tensor '{tensor_name}' referenced by operator inputs is not defined.")
                tid = self._name_to_id[tensor_name]
                proxy.data_as_operand(self.pipeline.query_local_tensor(tid), idx)
            for idx, ref in enumerate(op.get("outputs", [])):
                tensor_name = self._resolve_tensor_ref(ref)
                if not tensor_name:
                    continue
                if tensor_name not in self._name_to_id:
                    raise KeyError(f"Tensor '{tensor_name}' referenced by operator outputs is not defined.")
                tid = self._name_to_id[tensor_name]
                proxy.connect_result_to_data_array(idx, self.pipeline.query_local_tensor(tid))

            # Special handling: ASSIGNMENT slices wiring
            # TODO: remove special handling
            if int(op_type) == int(smr.EOperatorType.ASSIGNMENT):
                src_slices = op.get("src_slices")
                dst_slices = op.get("dst_slices")
                def _make_slice_tensor(arr: np.ndarray) -> smr.Tensor:
                    arr = np.ascontiguousarray(arr.astype(np.int32))
                    dims = [int(arr.shape[0])]
                    channels = int(arr.shape[1]) if arr.ndim == 2 else 1
                    base_vec = getattr(smr.BaseType, "VEC_2", None)
                    if base_vec is None:
                        flag = int(smr.EDataType.INT32) | int(smr.BaseType.BIT_VEC) | (int(smr.BaseType.CHANNEL_MASK) & channels)
                    else:
                        flag = int(smr.EDataType.INT32) | int(base_vec)
                        if channels != 2:
                            flag |= int(smr.BaseType.CHANNEL_MASK) & channels
                    tensor = smr.TensorFactory.create(dims, flag)
                    if hasattr(tensor, "load_from_raw_byte_arrays"):
                        tensor.load_from_raw_byte_arrays(arr.tobytes())
                    return tensor

                if src_slices is not None:
                    # Build vec2 list with [[row_start,row_end],[col_start,col_end]]
                    arr = np.array(src_slices, dtype=np.int32)
                    vec = _make_slice_tensor(arr)
                    proxy.data_as_operand(vec, 1)
                if dst_slices is not None:
                    arr = np.array(dst_slices, dtype=np.int32)
                    vec = _make_slice_tensor(arr)
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

                src_channel_slice = op.get("src_channel_slice")
                if src_channel_slice is not None:
                    arr = np.array([src_channel_slice], dtype=np.int32)
                    vec = _make_slice_tensor(arr)
                    proxy.data_as_operand(vec, 2)

                dst_channel_slice = op.get("dst_channel_slice")
                if dst_channel_slice is not None:
                    arr = np.array([dst_channel_slice], dtype=np.int32)
                    vec = _make_slice_tensor(arr)
                    proxy.data_as_operand(vec, 4)

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

    def __call__(
        self,
        inputs: Optional[Union[smr.Tensor, np.ndarray, Dict[Union[str, int], Union[smr.Tensor, np.ndarray]]]] = None,
        timeout=2,
    ):
        ph_map: Dict[int, smr.Tensor] = dict(self.placeholder_map)

        def _assign(target_tid: int, value: Union[smr.Tensor, np.ndarray]):
            if isinstance(value, smr.Tensor):
                ph_map[target_tid] = value
            elif isinstance(value, np.ndarray):
                t = ph_map[target_tid]
                t.load_from_raw_byte_arrays(np.ascontiguousarray(value).tobytes())
            else:
                raise TypeError("Unsupported input type; must be smr.Tensor or numpy.ndarray")

        if inputs is None:
            if self._inputs_names:
                raise ValueError("Pipeline expects input tensors; provide data via 'inputs'.")
        elif isinstance(inputs, dict):
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

        for _ in range(timeout * 100):
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

    def _prepare_operator_attrs(self, op: Dict[str, Any], op_type: smr.EOperatorType) -> List[str]:
        attrs: List[str] = [str(a) for a in op.get("attrs", [])]
        op_enum: Optional[smr.EOperatorType]
        try:
            op_enum = name_to_type(op.get("type"))
        except Exception:
            op_enum = None
        if not attrs:
            if "flag" in op:
                attrs = [str(op.get("flag"))]
            if "expression" in op:
                attrs = [str(op.get("expression"))]
            if (
                op_enum is not None
                and int(op_enum) == int(smr.EOperatorType.NMS)
                and "threshold" in op
            ):
                attrs = [str(op.get("threshold"))]
            if op_enum is not None and int(op_enum) == int(smr.EOperatorType.SORT_MAT):
                mode = op.get("mode") or op.get("axis")
                if mode is not None:
                    attrs = [str(mode)]

        if int(op_type) == int(smr.EOperatorType.PYTHON_CUSTOM):
            attrs = self._resolve_custom_operator_attrs(op, attrs)

        op["attrs"] = attrs
        return attrs

    def _resolve_custom_operator_attrs(self, op: Dict[str, Any], attrs: List[str]) -> List[str]:
        for attr in attrs:
            token = self._extract_custom_token(attr)
            if token is None:
                continue
            implementation = custom_ops.get_registered_custom_operator(token)
            if implementation is None:
                raise RuntimeError(
                    f"Custom operator token '{token}' is not registered. "
                    "Ensure the custom operator instance remains alive and registered before loading the serialized pipeline."
                )
            registry = getattr(custom_ops, "_TOKEN_TO_IMPLEMENTATION", None)
            if isinstance(registry, dict) and token in registry:
                registry.pop(token, None)
            handle = custom_ops.CustomOperatorHandle(implementation)
            self._custom_operator_handles.append(handle)
            new_attrs = [str(x) for x in handle.configs()]
            return new_attrs
        return attrs

    @staticmethod
    def _extract_custom_token(attr: Any) -> Optional[str]:
        if not isinstance(attr, str):
            return None
        prefix = "token:"
        if attr.lower().startswith(prefix):
            return attr[len(prefix):]
        return None

    def close(self) -> None:
        for handle in self._custom_operator_handles:
            with contextlib.suppress(Exception):
                handle.release()
        self._custom_operator_handles.clear()

    def __del__(self):  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass
