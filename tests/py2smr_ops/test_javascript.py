import numpy as np

from securemr.py2smr import ops, trace
from .conftest import run_op_test, skip_if_no_device


JS_CODE = (
    "var out_result; var in_sourceData; "
    "for(var i = 0; i < in_sourceData.length; i++) { "
    "out_result[i] = in_sourceData[i] * 2; }"
)


@trace(inputs=["in_sourceData"], outputs=["out_result"])
def traced_javascript(in_sourceData):
    return ops.javascript(JS_CODE, {"in_sourceData": in_sourceData}, ["out_result"])


def test_javascript_host():
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    result, verification = run_op_test(
        traced_javascript,
        {"in_sourceData": data},
        expected_output_name="out_result",
        test_device=False,
    )
    assert verification.success
    assert np.allclose(result["out_result"], data * 2)


@skip_if_no_device
def test_javascript_device():
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    _, verification = run_op_test(
        traced_javascript,
        {"in_sourceData": data},
        expected_output_name="out_result",
        test_device=True,
    )
    assert verification.success
