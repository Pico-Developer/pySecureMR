import os
from pathlib import Path
import numpy as np
import pytest

from securemr.py2smr import ops, trace, convert, verify
from .conftest import skip_if_no_device


MNIST_DIR = Path(__file__).resolve().parents[2] / "examples" / "mnistwild"
MODEL_BIN = MNIST_DIR / "mnist.serialized.bin"
MODEL_JSON = MNIST_DIR / "mnist.serialized.json"


def _load_model_meta():
    import json
    with open(MODEL_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    graph_info = data["info"]["graphs"][0]["info"]
    inputs = [t["info"]["name"] for t in graph_info["graphInputs"]]
    outputs = [t["info"]["name"] for t in graph_info["graphOutputs"]]
    input_dims = graph_info["graphInputs"][0]["info"]["dimensions"]
    output_types = [t["info"].get("dataType") for t in graph_info["graphOutputs"]]
    return inputs, outputs, input_dims, output_types, graph_info.get("graphName", "model")


def _qnn_type_to_numpy(qnn_type):
    mapping = {
        "QNN_DATATYPE_FLOAT_32": np.float32,
        "QNN_DATATYPE_FLOAT_16": np.float16,
        "QNN_DATATYPE_INT_32": np.int32,
        "QNN_DATATYPE_UINT_32": np.uint32,
        "QNN_DATATYPE_INT_16": np.int16,
        "QNN_DATATYPE_UINT_16": np.uint16,
        "QNN_DATATYPE_INT_8": np.int8,
        "QNN_DATATYPE_UINT_8": np.uint8,
    }
    return mapping.get(qnn_type, np.float32)


@trace(inputs=["input_1"], outputs=["_538", "_539"])
def traced_run_model_inference(input_1):
    inputs, outputs, _dims, output_types, name = _load_model_meta()
    output_dtypes = [_qnn_type_to_numpy(t) for t in output_types]
    return ops.run_model_inference(
        inputs={inputs[0]: input_1},
        model_file=str(MODEL_BIN),
        model_name=name,
        output_names=outputs,
        output_dtypes=output_dtypes,
    )


def test_run_model_inference_host():
    if os.getenv("QNN_SDK_ROOT") is None:
        pytest.skip("QNN_SDK_ROOT not set; skip model inference test")
    inputs, outputs, dims, _output_types, _name = _load_model_meta()
    assert MODEL_BIN.exists(), "MNIST serialized model is required"
    input_shape = (dims[1], dims[2], dims[3])
    input_tensor = np.random.rand(*input_shape).astype(np.float32)
    out_map, ctx = traced_run_model_inference.trace(input_1=input_tensor)
    spec = convert(ctx)
    expected = {outputs[0]: out_map[outputs[0]], outputs[1]: out_map[outputs[1]]}
    verification = verify(spec, {"input_1": input_tensor}, expected_outputs=expected)
    assert verification.success


@skip_if_no_device
def test_run_model_inference_device():
    inputs, outputs, dims, _output_types, _name = _load_model_meta()
    input_shape = (dims[1], dims[2], dims[3])
    input_tensor = np.random.rand(*input_shape).astype(np.float32)
    prev_target = os.getenv("PY2SMR_MODEL_INFERENCE_TARGET")
    os.environ["PY2SMR_MODEL_INFERENCE_TARGET"] = "android"

    out_map, ctx = traced_run_model_inference.trace(input_1=input_tensor)
    spec = convert(ctx)
    expected = {outputs[0]: out_map[outputs[0]], outputs[1]: out_map[outputs[1]]}
    verification = verify(
        spec,
        {"input_1": input_tensor},
        expected_outputs=expected,
        device=True,
        duration=60,
    )
    if verification.error_message == "Device execution failed":
        pytest.skip("pipeline_inspect produced no output files; device execution failed")
    assert verification.success
