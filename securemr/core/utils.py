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
"""Utility functions for the SecureMR package."""

import subprocess as commands
from typing import Any, Dict, Optional, Sequence, Tuple
from enum import Enum

import numpy as np

from .types import BaseType, EDataType

__all__ = [
    "run",
    "TORCH_INSTALLED",
    "ONNX_INSTALLED",
    "NUMPY_DTYPE",
    "SMR_DTYPE",
    "NUMPY_TO_SMR_DATATYPE",
    "convert_to_dtype",
    "convert_from_dtype",
    "mat_flag",
    "unmat_flag",
    "numpy_dtype_to_smr",
    "ensure_tensor_dimensions",
    "TensorType",
    "type_to_name",
]

_OP_ENUM_PREFIX = "XR_SECURE_MR_OPERATOR_TYPE_"
_OP_ENUM_SUFFIX = "_PICO"


def run(cmd) -> None:
    """Run bash command with print to stdout.

    Args:
        cmd: command string
    """
    print(f"\033[0;33m>> {cmd}\033[0m")
    (status, output) = commands.getstatusoutput(cmd)
    print(output)
    if status != 0:
        raise RuntimeError(f"Faild to execute {cmd}")


try:
    import torch  # noqa
    import torchvision  # noqa

    TORCH_INSTALLED = True
except ImportError:
    TORCH_INSTALLED = False

try:
    import onnx  # noqa

    ONNX_INSTALLED = True
except ImportError:
    ONNX_INSTALLED = False

NUMPY_DTYPE = [
    None,
    np.uint8,
    np.int8,
    np.uint16,
    np.int16,
    np.int32,
    np.float32,
    np.float64,
]

SMR_DTYPE = [
    None,
    EDataType.UINT8,
    EDataType.INT8,
    EDataType.UINT16,
    EDataType.INT16,
    EDataType.INT32,
    EDataType.FLOAT32,
    EDataType.FLOAT64,
]

NUMPY_TO_SMR_DATATYPE: Dict[Any, EDataType] = {
    np.uint8: EDataType.UINT8,
    np.int8: EDataType.INT8,
    np.uint16: EDataType.UINT16,
    np.int16: EDataType.INT16,
    np.int32: EDataType.INT32,
    np.float32: EDataType.FLOAT32,
    np.float64: EDataType.FLOAT64,
}


class TensorType(Enum):
    POINT = 1
    SCALAR = 2
    SLICE = 3
    COLOR = 4
    TIMESTAMP = 5
    MAT = 6
    GLTF = 7


def convert_to_dtype(data_type: int, target: str = "numpy"):
    if isinstance(data_type, str):
        normalized = data_type.strip().lower()
        aliases = {
            "uint8": 1,
            "byte": 1,
            "int8": 2,
            "sbyte": 2,
            "uint16": 3,
            "ushort": 3,
            "int16": 4,
            "short": 4,
            "int32": 5,
            "int": 5,
            "float32": 6,
            "float": 6,
            "fp32": 6,
            "float64": 7,
            "double": 7,
        }
        data_type = aliases.get(normalized, data_type)
    if target == "numpy":
        return NUMPY_DTYPE[data_type]
    if target == "smr":
        return SMR_DTYPE[data_type]
    raise NotImplementedError


def convert_from_dtype(data_type, source: str = "numpy"):
    if source == "numpy":
        return NUMPY_DTYPE.index(data_type)
    if source == "smr":
        return SMR_DTYPE.index(data_type)
    raise NotImplementedError


def mat_flag(dtype: EDataType, channels: int) -> int:
    return (
        int(dtype)
        | int(BaseType.MAT)
        | (int(BaseType.CHANNEL_MASK) & int(channels))
    )


def unmat_flag(flag: int) -> Tuple[EDataType, int]:
    if not (int(flag) & int(BaseType.MAT)):
        raise ValueError("flag does not encode a MAT type")
    channels = int(flag) & int(BaseType.CHANNEL_MASK)
    clear_mask = int(BaseType.MAT) | int(BaseType.CHANNEL_MASK)
    dtype_bits = int(flag) & ~clear_mask
    dtype = EDataType(dtype_bits)
    return dtype, channels


def numpy_dtype_to_smr(dtype_value: Any) -> Optional[EDataType]:
    if dtype_value is None:
        return None
    try:
        canonical = np.dtype(dtype_value).type
    except TypeError:
        return None
    if canonical is np.float16:
        canonical = np.float32
    return NUMPY_TO_SMR_DATATYPE.get(canonical)

def ensure_tensor_dimensions(spatial_dims: Sequence[int]) -> list[int]:
    dims = [max(int(d), 1) for d in spatial_dims]
    if not dims:
        return [1, 1]
    if len(dims) == 1:
        return [dims[0], 1]
    return dims


def type_to_name(op_type) -> str:
    """Return the canonical JSON `type` token for a SecureMR operator."""
    try:
        from .. import EOperatorType
    except ImportError:
        EOperatorType = None

    try:
        key_val = int(op_type)
    except Exception:
        key_val = int(op_type)

    value_to_name: Dict[int, str] = {}
    if EOperatorType is not None:
        for attr in dir(EOperatorType):
            if attr.startswith("__"):
                continue
            try:
                value_to_name[int(getattr(EOperatorType, attr))] = attr
            except Exception:
                continue

    enum_name = value_to_name.get(key_val)
    if not enum_name:
        return f"unknown_{key_val}"

    return f"{_OP_ENUM_PREFIX}{enum_name}{_OP_ENUM_SUFFIX}"
