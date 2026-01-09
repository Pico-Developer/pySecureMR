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

import os
import subprocess as commands
from typing import Any, Dict, Optional, Sequence, Tuple
from enum import Enum

import numpy as np

from .types import BaseType, EDataType

__all__ = [
    "run",
    "DEBUG_QNN",
    "TORCH_INSTALLED",
    "ONNX_INSTALLED",
    "NUMPY_DTYPE",
    "SMR_DTYPE",
    "QNN_DTYPE_TO_NUMPY",
    "NUMPY_TO_SMR_DATATYPE",
    "convert_to_dtype",
    "convert_from_dtype",
    "mat_flag",
    "unmat_flag",
    "normalize_qnn_dtype",
    "numpy_dtype_to_smr",
    "qnn_dtype_to_smr",
    "ensure_tensor_dimensions",
    "TensorType",
]


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


DEBUG_QNN = bool(os.getenv("DEBUG_QNN", "0") == "1")

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

QNN_DTYPE_TO_NUMPY: Dict[str, Any] = {
    "QNN_DATATYPE_FLOAT_32": np.float32,
    "QNN_DATATYPE_FLOAT32": np.float32,
    "QNN_DATATYPE_FLOAT": np.float32,
    "FLOAT_32": np.float32,
    "FLOAT32": np.float32,
    "FP32": np.float32,
    "QNN_DATATYPE_FLOAT_16": np.float16,
    "QNN_DATATYPE_FLOAT16": np.float16,
    "FLOAT_16": np.float16,
    "FLOAT16": np.float16,
    "FP16": np.float16,
    "QNN_DATATYPE_UINT8": np.uint8,
    "UINT8": np.uint8,
    "U8": np.uint8,
    "QNN_DATATYPE_INT8": np.int8,
    "INT8": np.int8,
    "QNN_DATATYPE_UINT16": np.uint16,
    "UINT16": np.uint16,
    "QNN_DATATYPE_INT16": np.int16,
    "INT16": np.int16,
    "QNN_DATATYPE_INT32": np.int32,
    "INT32": np.int32,
    "QNN_DATATYPE_FLOAT_64": np.float64,
    "QNN_DATATYPE_FLOAT64": np.float64,
    "FLOAT64": np.float64,
    "FP64": np.float64,
}

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


def normalize_qnn_dtype(dtype_value: Any) -> Optional[Any]:
    if dtype_value is None:
        return None
    if isinstance(dtype_value, np.dtype):
        return dtype_value.type
    if isinstance(dtype_value, type) and issubclass(dtype_value, np.generic):
        return dtype_value
    key = str(dtype_value).strip()
    if not key:
        return None
    normalized = key.upper().replace(" ", "").replace("-", "_")
    candidates = [
        normalized,
        normalized.replace("QNN_DATATYPE_", ""),
        f"QNN_DATATYPE_{normalized}",
    ]
    for cand in candidates:
        if cand in QNN_DTYPE_TO_NUMPY:
            return QNN_DTYPE_TO_NUMPY[cand]
    return None


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


def qnn_dtype_to_smr(dtype_value: Any, default: EDataType = EDataType.FLOAT32) -> EDataType:
    dtype_enum = numpy_dtype_to_smr(normalize_qnn_dtype(dtype_value))
    return dtype_enum if dtype_enum is not None else default


def ensure_tensor_dimensions(spatial_dims: Sequence[int]) -> list[int]:
    dims = [max(int(d), 1) for d in spatial_dims]
    if not dims:
        return [1, 1]
    if len(dims) == 1:
        return [dims[0], 1]
    return dims
