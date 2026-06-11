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

"""SecureMR package for securemr model representation and deployment."""

from .bindings import BindingsUnavailableError, bindings_available, load_bindings
from .core.types import BaseType, EDataType, EOperatorType


def _missing_bindings_type(name: str):
    class _MissingBindingsType:  # noqa: D401
        """Placeholder type for unavailable native bindings."""

        def __init__(self, *args, **kwargs):  # noqa: ANN001
            raise BindingsUnavailableError(
                f"{name} is unavailable because native bindings could not be loaded."
            )

        def __getattr__(self, attr: str):
            if attr.startswith("__"):
                raise AttributeError(attr)
            raise BindingsUnavailableError(
                f"{name} is unavailable because native bindings could not be loaded."
            )

        @classmethod
        def __getattr__(cls, attr: str):
            if attr.startswith("__"):
                raise AttributeError(attr)
            raise BindingsUnavailableError(
                f"{name} is unavailable because native bindings could not be loaded."
            )

    _MissingBindingsType.__name__ = name
    return _MissingBindingsType


_HAS_BINDINGS = False
if bindings_available():
    try:  # pragma: no cover - platform dependent
        load_bindings()
        from .bindings.linux._securemr import (  # noqa: E402
            OperatorFactory,
            Pipeline,
            Task,
            ThreadPool2,
            Tensor,
            TensorFactory,
            TensorMat,
            TensorPoint2Double,
            TensorPoint2Float,
            TensorPoint2Int,
            TensorPoint3Double,
            TensorPoint3Float,
            TensorPoint3Int,
            TensorScalarDouble,
            TensorScalarFloat,
            TensorScalarInt16,
            TensorScalarInt32,
            TensorScalarInt8,
            TensorScalarUInt16,
            TensorScalarUInt8,
            TensorSliceInt16,
            TensorSliceInt32,
            TensorSliceInt8,
            TensorSliceUInt16,
            TensorSliceUInt8,
            TensorColorDouble,
            TensorColorFloat,
            TensorColorInt16,
            TensorColorInt32,
            TensorColorInt8,
            TensorColorUInt16,
            TensorColorUInt8,
            TensorTimestampInt32,
        )
        _HAS_BINDINGS = True
    except Exception:  # noqa: BLE001
        _HAS_BINDINGS = False

if not _HAS_BINDINGS:
    OperatorFactory = _missing_bindings_type("OperatorFactory")
    Pipeline = _missing_bindings_type("Pipeline")
    Task = _missing_bindings_type("Task")
    ThreadPool2 = _missing_bindings_type("ThreadPool2")
    Tensor = _missing_bindings_type("Tensor")
    TensorFactory = _missing_bindings_type("TensorFactory")
    TensorMat = _missing_bindings_type("TensorMat")
    TensorPoint2Double = _missing_bindings_type("TensorPoint2Double")
    TensorPoint2Float = _missing_bindings_type("TensorPoint2Float")
    TensorPoint2Int = _missing_bindings_type("TensorPoint2Int")
    TensorPoint3Double = _missing_bindings_type("TensorPoint3Double")
    TensorPoint3Float = _missing_bindings_type("TensorPoint3Float")
    TensorPoint3Int = _missing_bindings_type("TensorPoint3Int")
    TensorScalarDouble = _missing_bindings_type("TensorScalarDouble")
    TensorScalarFloat = _missing_bindings_type("TensorScalarFloat")
    TensorScalarInt16 = _missing_bindings_type("TensorScalarInt16")
    TensorScalarInt32 = _missing_bindings_type("TensorScalarInt32")
    TensorScalarInt8 = _missing_bindings_type("TensorScalarInt8")
    TensorScalarUInt16 = _missing_bindings_type("TensorScalarUInt16")
    TensorScalarUInt8 = _missing_bindings_type("TensorScalarUInt8")
    TensorSliceInt16 = _missing_bindings_type("TensorSliceInt16")
    TensorSliceInt32 = _missing_bindings_type("TensorSliceInt32")
    TensorSliceInt8 = _missing_bindings_type("TensorSliceInt8")
    TensorSliceUInt16 = _missing_bindings_type("TensorSliceUInt16")
    TensorSliceUInt8 = _missing_bindings_type("TensorSliceUInt8")
    TensorColorDouble = _missing_bindings_type("TensorColorDouble")
    TensorColorFloat = _missing_bindings_type("TensorColorFloat")
    TensorColorInt16 = _missing_bindings_type("TensorColorInt16")
    TensorColorInt32 = _missing_bindings_type("TensorColorInt32")
    TensorColorInt8 = _missing_bindings_type("TensorColorInt8")
    TensorColorUInt16 = _missing_bindings_type("TensorColorUInt16")
    TensorColorUInt8 = _missing_bindings_type("TensorColorUInt8")
    TensorTimestampInt32 = _missing_bindings_type("TensorTimestampInt32")

HAS_BINDINGS = _HAS_BINDINGS

def __getattr__(name: str):  # noqa: D401
    """Lazy import optional utilities."""
    if name == "TORCH_INSTALLED":
        from .core.utils import TORCH_INSTALLED as _ti

        return _ti
    if name in {
        "ModelPackageSpec",
        "PipelinePackageEntry",
        "PipelineZooPackageSpec",
        "SUPPORTED_EXECUTION_MODES",
        "configure_litert_inference_operator",
        "create_litert_model_json",
        "load_pipeline_zoo_manifest",
        "validate_pipeline_zoo_manifest",
        "write_pipeline_zoo_package",
    }:
        from . import pipeline_zoo as _pipeline_zoo

        return getattr(_pipeline_zoo, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")

__version__ = "0.0.1"
__all__ = [
    "BaseType",
    "EDataType",
    "EOperatorType",
    "OperatorFactory",
    "Pipeline",
    "Task",
    "ThreadPool2",
    "Tensor",
    "TensorFactory",
    "TensorMat",
    "TensorPoint2Double",
    "TensorPoint2Float",
    "TensorPoint2Int",
    "TensorPoint3Double",
    "TensorPoint3Float",
    "TensorPoint3Int",
    "TensorScalarDouble",
    "TensorScalarFloat",
    "TensorScalarInt16",
    "TensorScalarInt32",
    "TensorScalarInt8",
    "TensorScalarUInt16",
    "TensorScalarUInt8",
    "TensorSliceInt16",
    "TensorSliceInt32",
    "TensorSliceInt8",
    "TensorSliceUInt16",
    "TensorSliceUInt8",
    "TensorColorDouble",
    "TensorColorFloat",
    "TensorColorInt16",
    "TensorColorInt32",
    "TensorColorInt8",
    "TensorColorUInt16",
    "TensorColorUInt8",
    "TensorTimestampInt32",
    "HAS_BINDINGS",
    "TORCH_INSTALLED",
    "ModelPackageSpec",
    "PipelinePackageEntry",
    "PipelineZooPackageSpec",
    "SUPPORTED_EXECUTION_MODES",
    "configure_litert_inference_operator",
    "create_litert_model_json",
    "load_pipeline_zoo_manifest",
    "validate_pipeline_zoo_manifest",
    "write_pipeline_zoo_package",
]
