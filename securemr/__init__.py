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

# flake8: noqa: E402
"""SecureMR package for securemr model representation and deployment."""

import ctypes
import os

bindings_path = os.path.join(os.path.dirname(__file__), "_bindings")
ctypes.CDLL(os.path.join(bindings_path, "libopencv_core.so.3.4"))
ctypes.CDLL(os.path.join(bindings_path, "libopencv_imgproc.so.3.4"))
ctypes.CDLL(os.path.join(bindings_path, "libopencv_flann.so.3.4"))
ctypes.CDLL(os.path.join(bindings_path, "libopencv_calib3d.so.3.4"))
ctypes.CDLL(os.path.join(bindings_path, "libopencv_imgcodecs.so.3.4"))
ctypes.CDLL(os.path.join(bindings_path, "libSNPE.so"))
ctypes.CDLL(os.path.join(bindings_path, "libopenmr-backend.so"))
ctypes.CDLL(os.path.join(bindings_path, "_securemr.cpython-310-x86_64-linux-gnu.so"))
# isort: off
from ._bindings._securemr import (
    BaseType,
    EDataType,
    EOperatorType,
    OperatorFactory,
    Tensor,
    TensorFactory,
    TensorMat,
    TensorPoint2Double,
    TensorPoint2Float,
    TensorPoint2Int,
    TensorPoint3Double,
    TensorPoint3Float,
    TensorPoint3Int,
)

# isort: on

from .pytorch_to_qnn import pytorch_to_qnn
from .qnn_model import QnnModel
from .utils import TORCH_INSTALLED

__version__ = "0.0.1"
