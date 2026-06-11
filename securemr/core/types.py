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
"""Core enum types with optional binding-backed definitions."""

from enum import IntEnum


try:  # pragma: no cover - exercised only when bindings are present
    from ..bindings.loader import bindings_available, load_bindings

    if bindings_available():
        load_bindings()
        from ..bindings.linux._securemr import BaseType, EDataType, EOperatorType  # noqa: F401
        BINDINGS_AVAILABLE = True
    else:
        raise RuntimeError("Bindings unavailable")
except Exception:  # noqa: BLE001
    BINDINGS_AVAILABLE = False

    class EDataType(IntEnum):
        NONE = 0
        UINT8 = 1
        INT8 = 2
        UINT16 = 3
        INT16 = 4
        INT32 = 5
        FLOAT32 = 6
        FLOAT64 = 7

    class BaseType(IntEnum):
        MAT = 1 << 8
        POINT_2 = 1 << 9
        POINT_3 = 1 << 10
        SCALAR = 1 << 11
        SLICE = 1 << 12
        COLOR = 1 << 13
        TIMESTAMP = 1 << 14
        GLTF = 1 << 15
        BIT_VEC = 1 << 16
        VEC_2 = 1 << 17
        CHANNEL_MASK = 0xFF

    class EOperatorType(IntEnum):
        UNKNOWN = 0
        ARITHMETIC_COMPOSE = 1
        ELEMENTWISE_MIN = 4
        ELEMENTWISE_MAX = 5
        ELEMENTWISE_MULTIPLY = 6
        CUSTOMIZED_COMPARE = 7
        ELEMENTWISE_OR = 8
        ELEMENTWISE_AND = 9
        ALL = 10
        ANY = 11
        NMS = 12
        SOLVE_P_N_P = 13
        GET_AFFINE = 14
        APPLY_AFFINE = 15
        APPLY_AFFINE_POINT = 16
        UV_TO_3D_IN_CAM_SPACE = 17
        ASSIGNMENT = 18
        RUN_MODEL_INFERENCE = 19
        NORMALIZE = 21
        CAMERA_SPACE_TO_WORLD = 22
        RECTIFIED_VST_ACCESS = 23
        ARGMAX = 24
        CONVERT_COLOR = 25
        SORT_VEC = 26
        INVERSION = 27
        GET_TRANSFORM_MAT = 28
        SORT_MAT = 29
        SWITCH_GLTF_RENDER_STATUS = 30
        UPDATE_GLTF = 31
        RENDER_TEXT = 32
        LOAD_TEXTURE = 33
        SVD = 34
        NORM = 35
        SWAP_HWC_CHW = 36
        SCENEGRAPH_VISIBILITY = 37
        UPDATE_COMPONENT = 38
        JAVASCRIPT = 39
        MICROPHONE = 40
        SPEAKER = 41
        DEPTH = 42


__all__ = ["BaseType", "EDataType", "EOperatorType", "BINDINGS_AVAILABLE"]
