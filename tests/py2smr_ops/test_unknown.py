import numpy as np
import pytest

from securemr.py2smr import ops, trace
from securemr.py2smr.verifier import run_pipeline_python
from .conftest import run_op_test


@trace(inputs=["tensor"], outputs=["out"])
def traced_unknown(tensor):
    return ops.unknown(tensor, output_name="out")


def test_unknown_host():
    tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result, verification = run_op_test(
        traced_unknown,
        {"tensor": tensor},
        expected_output_name="out",
        test_device=False,
    )
    assert verification.success
    assert np.allclose(result, tensor)


def test_unknown_device_skip():
    pytest.skip("unknown operator type not supported on device")


def test_custom_operator_handler_receives_unknown_operator():
    spec = {
        "tensors": {
            "input": {"dimensions": [3, 1], "channels": 1, "data_type": 6, "is_placeholder": True},
            "output": {"dimensions": [3, 1], "channels": 1, "data_type": 6, "is_placeholder": True},
        },
        "operators": [{"type": "vendor_custom", "inputs": ["input"], "outputs": ["output"],
                         "attrs": ["preserve"]}],
        "inputs": ["input"],
        "outputs": ["output"],
    }
    seen = []

    def handler(op_spec, input_tensors, output_names, tensors):
        seen.append((op_spec["type"], op_spec["attrs"]))
        tensors[output_names[0]] = input_tensors[0] + 1
        return True

    result = run_pipeline_python(
        spec,
        {"input": np.array([[1.0], [2.0], [3.0]], dtype=np.float32)},
        custom_operator_handler=handler,
    )
    assert seen == [("vendor_custom", ["preserve"]) ]
    np.testing.assert_allclose(result["output"], [[2.0], [3.0], [4.0]])


def test_unknown_operator_without_handler_is_rejected():
    spec = {
        "tensors": {},
        "operators": [{"type": "vendor_custom", "inputs": [], "outputs": []}],
        "inputs": [],
        "outputs": [],
    }
    with pytest.raises(ValueError, match="no registered Python custom handler"):
        run_pipeline_python(spec, {})
