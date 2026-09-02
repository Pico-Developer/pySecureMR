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

"""LiteRT Python runtime bridge for pySpatialML commands."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from pyspatialml.litert_tools import LiteRTCli, LiteRTToolError, litert_python_for_cli, resolve_litert_cli


class LiteRTRuntimeError(RuntimeError):
    """Raised when LiteRT Python runtime work cannot be completed."""


def inspect_model(model: Path, *, signature_index: int = 0) -> dict[str, Any]:
    """Return LiteRT/TFLite model input and output metadata."""
    try:
        return inspect_model_in_process(model, signature_index=signature_index)
    except Exception as in_process_exc:  # noqa: BLE001
        try:
            cli = resolve_litert_cli(ensure=True)
            return inspect_model_subprocess(
                litert_python_for_cli(cli),
                model,
                signature_index=signature_index,
            )
        except LiteRTToolError as tool_exc:
            raise LiteRTRuntimeError(f"{in_process_exc}; {tool_exc}") from in_process_exc
        except Exception as subprocess_exc:  # noqa: BLE001
            raise LiteRTRuntimeError(f"{in_process_exc}; {subprocess_exc}") from in_process_exc


def inspect_model_in_process(model: Path, *, signature_index: int) -> dict[str, Any]:
    """Inspect a model using the active Python interpreter."""
    from ai_edge_litert.interpreter import Interpreter

    try:
        interpreter = Interpreter(model_path=str(model), num_threads=1)
        interpreter.allocate_tensors()
        signature_key = _interpreter_signature_key(interpreter, signature_index)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    except LiteRTRuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise LiteRTRuntimeError(f"Failed to inspect model {model}: {exc}") from exc

    return _model_info_payload(
        model=model,
        signature_index=signature_index,
        signature_key=signature_key,
        input_details={str(detail.get("name")): detail for detail in input_details},
        output_details={str(detail.get("name")): detail for detail in output_details},
    )


def inspect_model_subprocess(python: Path, model: Path, *, signature_index: int) -> dict[str, Any]:
    """Inspect a model using a LiteRT-managed Python interpreter."""
    result = subprocess.run(
        [str(python), "-c", _INSPECT_MODEL_CODE, str(model), str(signature_index)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return json.loads(result.stdout)


def run_model(
    *,
    model_path: Path,
    inputs: Mapping[str, np.ndarray],
    output_names: Sequence[str],
    output_shapes: Optional[Sequence[tuple]] = None,
    output_dtypes: Optional[Sequence[Any]] = None,
    litert_cli: Optional[LiteRTCli] = None,
) -> dict[str, np.ndarray]:
    """Run a LiteRT model, falling back to the resolved LiteRT interpreter when needed."""
    try:
        return run_model_in_process(
            model_path=model_path,
            inputs=inputs,
            output_names=output_names,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
        )
    except (ImportError, ModuleNotFoundError):
        cli = litert_cli or resolve_litert_cli(ensure=True)
        return run_model_subprocess(
            python=litert_python_for_cli(cli),
            model_path=model_path,
            inputs=inputs,
            output_names=output_names,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
        )


def run_model_in_process(
    *,
    model_path: Path,
    inputs: Mapping[str, np.ndarray],
    output_names: Sequence[str],
    output_shapes: Optional[Sequence[tuple]] = None,
    output_dtypes: Optional[Sequence[Any]] = None,
) -> dict[str, np.ndarray]:
    """Run a LiteRT model using the active Python interpreter."""
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    for details in input_details:
        model_input_name = details.get("name")
        value = _resolve_model_input(model_input_name, inputs)
        if value is None:
            raise LiteRTRuntimeError(f"Missing model input tensor: {model_input_name}")
        shape = details.get("shape", list(np.asarray(value).shape))
        shape = tuple(int(dim) for dim in np.asarray(shape).reshape(-1).tolist())
        dtype = np.dtype(details.get("dtype", np.float32))
        array = np.asarray(value, dtype=dtype)
        if shape and tuple(array.shape) != shape and array.size == int(np.prod(shape)):
            array = array.reshape(shape)
        interpreter.set_tensor(int(details["index"]), array)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    model_output_names = [str(detail.get("name")) for detail in output_details]
    output_by_name = {str(detail.get("name")): detail for detail in output_details}
    results = {}
    for index, pipeline_output_name in enumerate(output_names):
        model_output_name = _resolve_model_output_name(index, pipeline_output_name, model_output_names)
        if model_output_name not in output_by_name:
            raise LiteRTRuntimeError(f"Model output tensor not found: {model_output_name}")
        details = output_by_name[model_output_name]
        array = np.asarray(interpreter.get_tensor(int(details["index"])))
        if output_shapes and index < len(output_shapes):
            requested_shape = output_shapes[index]
            if requested_shape and array.shape != tuple(requested_shape) and array.size == int(np.prod(requested_shape)):
                array = array.reshape(requested_shape)
        if output_dtypes and index < len(output_dtypes):
            array = array.astype(np.dtype(output_dtypes[index]), copy=False)
        results[pipeline_output_name] = array
    return results


def run_model_subprocess(
    *,
    python: Path,
    model_path: Path,
    inputs: Mapping[str, np.ndarray],
    output_names: Sequence[str],
    output_shapes: Optional[Sequence[tuple]] = None,
    output_dtypes: Optional[Sequence[Any]] = None,
) -> dict[str, np.ndarray]:
    """Run a LiteRT model using a LiteRT-managed Python interpreter."""
    with tempfile.TemporaryDirectory(prefix="pyspatialml-litert-run-") as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "inputs.npz"
        output_path = tmp_path / "outputs.npz"
        np.savez(input_path, **{name: np.asarray(value) for name, value in inputs.items()})
        request = {
            "model_path": str(model_path),
            "input_path": str(input_path),
            "output_path": str(output_path),
            "output_names": list(output_names),
            "output_shapes": [list(shape) if shape is not None else None for shape in (output_shapes or [])],
            "output_dtypes": [str(np.dtype(dtype)) for dtype in (output_dtypes or [])],
        }
        request_path = tmp_path / "request.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        try:
            subprocess.run(
                [str(python), "-c", _RUN_MODEL_CODE, str(request_path)],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as exc:
            details = (exc.stderr or exc.stdout or str(exc)).strip()
            raise LiteRTRuntimeError(f"LiteRT subprocess model run failed: {details}") from exc
        with np.load(output_path) as loaded_outputs:
            return {name: loaded_outputs[name] for name in loaded_outputs.files}


def _model_info_payload(
    *,
    model: Path,
    signature_index: int,
    signature_key: str,
    input_details: Mapping[str, Mapping[str, Any]],
    output_details: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "model": str(model),
        "signature_index": signature_index,
        "signature_key": signature_key,
        "inputs": [_tensor_detail(name, detail) for name, detail in input_details.items()],
        "outputs": [_tensor_detail(name, detail) for name, detail in output_details.items()],
    }


def _tensor_detail(name: str, detail: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": detail.get("name", name),
        "index": detail.get("index"),
        "dtype": _serialize_dtype(detail.get("dtype")),
        "shape": _serialize_shape(detail.get("shape", [])),
    }


def _interpreter_signature_key(interpreter: Any, signature_index: int) -> str:
    signatures = interpreter.get_signature_list()
    if not signatures:
        if signature_index == 0:
            return "<placeholder signature>"
        raise LiteRTRuntimeError(f"Signature index {signature_index} not found")
    keys = list(signatures)
    if signature_index < 0 or signature_index >= len(keys):
        raise LiteRTRuntimeError(f"Signature index {signature_index} not found")
    return str(keys[signature_index])


def _serialize_dtype(dtype: Any) -> str:
    try:
        return np.dtype(dtype).name
    except TypeError:
        return str(dtype)


def _serialize_shape(shape: Any) -> list[int]:
    return [int(dim) for dim in np.asarray(shape, dtype=np.int64).reshape(-1).tolist()]


def _resolve_model_input(name: str, inputs: Mapping[str, np.ndarray]) -> Optional[np.ndarray]:
    if name in inputs:
        return inputs[name]
    if len(inputs) == 1:
        return next(iter(inputs.values()))
    return None


def _resolve_model_output_name(index: int, pipeline_output_name: str, model_output_names: Sequence[str]) -> str:
    if pipeline_output_name in model_output_names:
        return pipeline_output_name
    if index < len(model_output_names):
        return model_output_names[index]
    return pipeline_output_name


_INSPECT_MODEL_CODE = r'''
import json
import sys
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter

model = Path(sys.argv[1])
signature_index = int(sys.argv[2])
interpreter = Interpreter(model_path=str(model), num_threads=1)
interpreter.allocate_tensors()
signatures = interpreter.get_signature_list()
if signatures:
    keys = list(signatures)
    if signature_index < 0 or signature_index >= len(keys):
        raise RuntimeError(f"Signature index {signature_index} not found")
    signature_key = str(keys[signature_index])
else:
    if signature_index != 0:
        raise RuntimeError(f"Signature index {signature_index} not found")
    signature_key = "<placeholder signature>"
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def tensor_detail(detail):
    return {
        "name": detail.get("name"),
        "index": detail.get("index"),
        "dtype": np.dtype(detail.get("dtype")).name,
        "shape": [int(dim) for dim in np.asarray(detail.get("shape", []), dtype=np.int64).reshape(-1).tolist()],
    }

print(json.dumps({
    "model": str(model),
    "signature_index": signature_index,
    "signature_key": signature_key,
    "inputs": [tensor_detail(detail) for detail in input_details],
    "outputs": [tensor_detail(detail) for detail in output_details],
}))
'''


_RUN_MODEL_CODE = r'''
import json
import sys
import numpy as np

from ai_edge_litert.interpreter import Interpreter

request = json.load(open(sys.argv[1], "r", encoding="utf-8"))
loaded = np.load(request["input_path"])
inputs = {name: loaded[name] for name in loaded.files}
interpreter = Interpreter(model_path=request["model_path"], num_threads=1)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
for details in input_details:
    model_input_name = details.get("name")
    value = inputs.get(model_input_name)
    if value is None and len(inputs) == 1:
        value = next(iter(inputs.values()))
    if value is None:
        raise RuntimeError(f"Missing model input tensor: {model_input_name}")
    shape = details.get("shape", list(np.asarray(value).shape))
    shape = tuple(int(dim) for dim in np.asarray(shape).reshape(-1).tolist())
    dtype = np.dtype(details.get("dtype", np.float32))
    array = np.asarray(value, dtype=dtype)
    if shape and tuple(array.shape) != shape and array.size == int(np.prod(shape)):
        array = array.reshape(shape)
    interpreter.set_tensor(int(details["index"]), array)

interpreter.invoke()
output_details = interpreter.get_output_details()
model_output_names = [str(detail.get("name")) for detail in output_details]
output_by_name = {str(detail.get("name")): detail for detail in output_details}
results = {}
output_shapes = request.get("output_shapes", [])
output_dtypes = request.get("output_dtypes", [])
for index, pipeline_output_name in enumerate(request["output_names"]):
    model_output_name = pipeline_output_name if pipeline_output_name in model_output_names else model_output_names[index]
    details = output_by_name[model_output_name]
    array = np.asarray(interpreter.get_tensor(int(details["index"])))
    requested_shape = output_shapes[index] if index < len(output_shapes) else None
    if requested_shape and array.shape != tuple(requested_shape) and array.size == int(np.prod(requested_shape)):
        array = array.reshape(requested_shape)
    if index < len(output_dtypes):
        array = array.astype(np.dtype(output_dtypes[index]), copy=False)
    results[pipeline_output_name] = array
np.savez(request["output_path"], **results)
'''
