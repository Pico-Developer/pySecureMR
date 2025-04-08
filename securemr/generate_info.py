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

"""Module for generating and managing model information in SecureMR."""

import json
import os
from typing import Dict, List, Optional, Union

import numpy as np

__all__ = ["QNNModelInfo", "main"]


class Dtype(object):
    """Enumeration of supported data types."""

    def __init__(self, dtype: Union[str, np.dtype]):
        if isinstance(dtype, str):
            if dtype == "fp32":
                dtype = "float32"
            if dtype == "fp16":
                dtype = "float16"
            if dtype == "fp64":
                dtype = "float64"
        try:
            self.dtype = np.dtype(dtype)
        except TypeError:
            self.dtype = dtype
        if dtype in ["str", "string"]:
            self.dtype = "string"

    def __str__(self):
        return str(self.dtype)

    def to_np(self):
        return np.dtype(str(self))

    def short_str(self):
        if str(self) == "float32":
            return "fp32"
        if str(self) == "float16":
            return "fp16"
        if str(self) == "float64":
            return "fp64"
        return str(self)

    def encoding_type(self):
        if str(self) == "float32":
            return "FP32"
        if str(self) == "float16":
            return "FP16"
        if str(self) == "float64":
            return "FP64"
        return str(self)


class TensorInfo(object):
    """Container for tensor information."""

    def __init__(self, name="", shape=(), dtype="unknown", doc_string=""):
        self.name = name
        self.shape = shape
        self.dtype = Dtype(dtype)
        self.doc_string = doc_string

    def __str__(self):
        return f"TensorInfo(name={self.name}, shape={self.shape}, dtype={self.dtype}, doc_string={self.doc_string})"


class QNNRuntime:
    """A class to represent the runtime of a qnn model."""

    RuntimeEnum = ("HTP_FIXED8_TF", "CPU_FLOAT32", "GPU_FLOAT16")

    def __init__(self, runtime: str):
        self.runtime = runtime.upper()
        if self.runtime not in self.RuntimeEnum:
            if self.runtime == "HTP":
                self.runtime = "HTP_FIXED8_TF"
            elif self.runtime == "CPU":
                self.runtime = "CPU_FLOAT32"
            elif self.runtime == "GPU":
                self.runtime = "GPU_FLOAT16"
            else:
                raise ValueError(f"Unknown runtime {self.runtime}")

    def __str__(self):
        return self.runtime


# TODO: parsing DATATYPE in $model_net.json
_DATATYPE_MAP = {
    # if the precision is int8, the result will be 1032
    # see qti/aisw/quantization_checker/utils/Op.py
    1032: "fp32",
    # if the precision is fp32, the result will be 562
    562: "fp32",
}


def _tensor_to_dict(t: TensorInfo, alias_name: str) -> Dict:
    return {"name": t.name, "shape": t.shape, "encoding_type": t.dtype.encoding_type(), "alias_name": alias_name}


def get_file_basename(filename: str, remove_ext=True) -> str:
    """Get the basename of a file without extension.

    Args:
        file_path: Path to the file.

    Returns:
        Basename of the file without extension.

    Example:
        >>> path = '/opt/tiger/a.md'
        >>> get_file_basename(path) # return a
    """
    basename = os.path.basename(filename)
    if not remove_ext:
        return basename
    return os.path.splitext(basename)[0]


class QNNModelInfo(object):
    """Class representing information about a QNN model."""

    def __init__(self, qnn_model_json_file: str, qnn_model_file: str):
        """Initialize QNNModelInfo object.

        Args:
            qnn_model_json_file: Path to the QNN model JSON file.
            qnn_model_file: Path to the QNN model file.
        """
        self.model_json = qnn_model_json_file
        self.model_path = qnn_model_file
        self.model_info = json.loads(open(self.model_json, "r").read())
        self.model_name = get_file_basename(self.model_info["model.cpp"], remove_ext=True)

    def get_total_params(self) -> float:
        """Get the total number of parameters in the model.

        Returns:
            Total number of parameters.
        """
        return self.model_info["Total parameters"]

    def get_total_macs(self) -> str:
        """Get the total number of MACs (multiply-accumulate operations) in the model.

        Returns:
            Total number of MACs.
        """
        return self.model_info["Total MACs per inference"]

    def get_input_infos(self) -> List[TensorInfo]:
        """Get information about model inputs.

        Returns:
            List of input tensor information.
        """
        tensors: Dict = self.model_info["graph"]["tensors"]
        inputs = []
        for name, info in tensors.items():
            if info["type"] == 0:
                shape = info["dims"]
                dtype = np.float32
                inputs.append(TensorInfo(name=name, shape=shape, dtype=dtype))
        return inputs

    def get_output_infos(self) -> List[TensorInfo]:
        """Get information about model outputs.

        Returns:
            List of output tensor information.
        """
        tensors: Dict = self.model_info["graph"]["tensors"]
        outputs = []
        for name, info in tensors.items():
            if info["type"] == 1:
                shape = info["dims"]
                dtype = np.float32
                outputs.append(TensorInfo(name=name, shape=shape, dtype=dtype))
        return outputs

    def to_json(
        self,
        runtime: Union[str, List[str]] = "cpu",
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        input_alias_names: Optional[List[str]] = None,
        output_alias_names: Optional[List[str]] = None,
        output_filename: Optional[str] = None,
        enable_dynamic_runtime: bool = False,
        **custom_model_info,
    ) -> str:
        """Convert model information to JSON format.

        Returns:
            JSON representation of model information.
        """
        model_name = model_name or self.model_name
        path_to_zoo = model_path or get_file_basename(self.model_path, remove_ext=False)
        model = {
            "model_name": model_name,
            "path_to_zoo": path_to_zoo,
            "engine_type": "qnn",
        }
        input_infos = self.get_input_infos()
        output_infos = self.get_output_infos()
        if input_alias_names is None:
            input_alias_names = [t.name for t in input_infos]
        if output_alias_names is None:
            output_alias_names = [t.name for t in output_infos]
        model["input"] = [_tensor_to_dict(t, input_alias_names[i]) for i, t in enumerate(input_infos)]
        model["output"] = [_tensor_to_dict(t, output_alias_names[i]) for i, t in enumerate(output_infos)]
        if isinstance(runtime, str):
            runtime = [runtime]
        runtime = [QNNRuntime(r) for r in runtime]
        model["specific_config"] = {
            "runtime_order": [str(r) for r in runtime],
            "enable_dynamic_runtime": enable_dynamic_runtime,
        }
        model.update(**custom_model_info)

        json_str = json.dumps(model, indent=4)
        if output_filename:
            with open(output_filename, "w") as f:
                f.write(json_str)
        return json_str


def main(
    model_net_json: str,
    context_binary: str,
    output: str = "./",
):
    """Generate model info from qnn."""
    outdir = os.path.dirname(os.path.abspath(context_binary))
    output = os.path.join(outdir, output, "0")
    os.makedirs(output, exist_ok=True)
    os.system(f"cp {context_binary} {output}/")

    model = QNNModelInfo(model_net_json, context_binary)
    output_filename = os.path.join(output, "model_info.json")
    model.to_json(
        runtime="HTP",
        output_filename=output_filename,
    )
