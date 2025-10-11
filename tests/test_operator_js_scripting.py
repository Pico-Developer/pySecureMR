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

import numpy as np
import pytest

import securemr as smr


def _js_operator_type() -> smr.EOperatorType:
    """Return the JS scripting operator enum value, falling back to the raw id if unnamed."""
    if hasattr(smr.EOperatorType, "JS_SCRIPTING"):
        return getattr(smr.EOperatorType, "JS_SCRIPTING")
    return smr.EOperatorType(39)


def _create_operator(js_code: str):
    js_op_type = _js_operator_type()
    try:
        return smr.OperatorFactory.create(js_op_type, [js_code])
    except RuntimeError as exc:  # pragma: no cover - defensive to avoid hard failure
        pytest.skip(f"JS scripting operator unavailable: {exc}")


def _squeeze(tensor: smr.TensorMat) -> np.ndarray:
    return np.squeeze(tensor.numpy()).astype(np.float32)


def test_js_scripting_basic_functionality():
    js_code = '''
        var out_result = [];
        var in_sourceData;
        for(var i = 0; i < in_sourceData.length; i++) {
            out_result[i] = in_sourceData[i] * 2;
        }
    '''
    op = _create_operator(js_code)

    assert op.get_operator_type().value == _js_operator_type().value
    assert op.get_operand_cnt() == 1
    assert op.get_results_cnt() == 1

    input_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    input_tensor = smr.TensorMat.from_numpy(input_data)
    output_tensor = smr.TensorMat([1, input_data.shape[1]], 1, smr.EDataType.FLOAT32)

    assert op.verify_compatibility_data_array_as_operand(0, input_tensor)
    assert op.verify_compatibility_data_array_as_result(0, output_tensor)

    op.data_as_operand(input_tensor, 0)
    op.connect_result_to_data_array(0, output_tensor)

    np.testing.assert_allclose(_squeeze(input_tensor), input_data.squeeze())

    op.compute(0)
    result = _squeeze(output_tensor)
    expected = input_data.squeeze() * 2.0
    np.testing.assert_allclose(result, expected)


def test_js_scripting_multiple_inputs_outputs():
    js_code = '''
        var out_sum = [];
        var out_product = [];
        var in_a;
        var in_b;
        for(var i = 0; i < in_a.length; i++) {
            out_sum[i] = in_a[i] + in_b[i];
            out_product[i] = in_a[i] * in_b[i];
        }
    '''
    op = _create_operator(js_code)

    assert op.get_operand_cnt() == 2
    assert op.get_results_cnt() == 2

    data_a = np.array([[2.0, 3.0]], dtype=np.float32)
    data_b = np.array([[4.0, 5.0]], dtype=np.float32)

    tensor_a = smr.TensorMat.from_numpy(data_a)
    tensor_b = smr.TensorMat.from_numpy(data_b)
    sum_tensor = smr.TensorMat([1, data_a.shape[1]], 1, smr.EDataType.FLOAT32)
    product_tensor = smr.TensorMat([1, data_a.shape[1]], 1, smr.EDataType.FLOAT32)

    a_index = op.get_operand_idx_from_name("in_a")
    b_index = op.get_operand_idx_from_name("in_b")
    sum_index = op.get_result_idx_from_name("out_sum")
    product_index = op.get_result_idx_from_name("out_product")

    assert a_index == 0
    assert b_index == 1
    assert sum_index == 0
    assert product_index == 1

    assert op.verify_compatibility_data_array_as_operand(a_index, tensor_a)
    assert op.verify_compatibility_data_array_as_operand(b_index, tensor_b)
    assert op.verify_compatibility_data_array_as_result(sum_index, sum_tensor)
    assert op.verify_compatibility_data_array_as_result(product_index, product_tensor)

    op.data_as_operand(tensor_a, a_index)
    op.data_as_operand(tensor_b, b_index)
    op.connect_result_to_data_array(sum_index, sum_tensor)
    op.connect_result_to_data_array(product_index, product_tensor)

    op.compute(0)

    sum_result = _squeeze(sum_tensor)
    product_result = _squeeze(product_tensor)
    expected_sum = data_a.squeeze() + data_b.squeeze()
    expected_product = data_a.squeeze() * data_b.squeeze()

    np.testing.assert_allclose(sum_result, expected_sum)
    np.testing.assert_allclose(product_result, expected_product)


def test_js_scripting_generate_tensor_without_operands():
    js_code = (
        "var out_frame = [];"
        " var width = 480;"
        " var height = 320;"
        " var channels = 3;"
        " for (var i = 0; i < width * height * channels; i++) {"
        "   out_frame[i] = Math.sin(i);"
        " }"
    )
    op = _create_operator(js_code)

    assert op.get_operand_cnt() == 0
    result_index = op.get_result_idx_from_name("out_frame")
    assert result_index == 0

    output_tensor = smr.TensorMat([320, 480], 3, smr.EDataType.FLOAT32)
    assert op.verify_compatibility_data_array_as_result(result_index, output_tensor)

    op.connect_result_to_data_array(result_index, output_tensor)
    op.compute(0)

    frame = output_tensor.numpy().astype(np.float32)
    assert frame.shape == (320, 480, 3)
    assert np.isfinite(frame).all()
    assert frame.min() >= -1.0 - 1e-5
    assert frame.max() <= 1.0 + 1e-5
    assert not np.allclose(frame, 0.0)
