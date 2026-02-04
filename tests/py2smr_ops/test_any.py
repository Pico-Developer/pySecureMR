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
"""Tests for any operator (ANY)."""

import numpy as np
from securemr.py2smr import trace, ops
from .conftest import run_op_test, skip_if_no_device


class TestAnyOp:
    """Tests for the any operation."""

    def test_any_true(self):
        """Contains non-zero -> 1."""
        @trace(inputs=["x"], outputs=["y"])
        def any_op(x):
            return ops.any(x)

        input_arr = np.array([[0, 0], [3, 0]], dtype=np.int32)
        result, verification = run_op_test(
            any_op,
            inputs={"x": input_arr},
            expected_output_name="y",
        )

        assert verification.success, verification.error_message
        np.testing.assert_array_equal(result, np.array([1], dtype=np.int32))

    def test_any_false(self):
        """All zeros -> 0."""
        @trace(inputs=["x"], outputs=["y"])
        def any_op(x):
            return ops.any(x)

        input_arr = np.array([[0, 0], [0, 0]], dtype=np.int32)
        result, verification = run_op_test(
            any_op,
            inputs={"x": input_arr},
            expected_output_name="y",
        )

        assert verification.success, verification.error_message
        np.testing.assert_array_equal(result, np.array([0], dtype=np.int32))

    @skip_if_no_device
    def test_any_on_device(self):
        """Test any operation on device."""
        @trace(inputs=["x"], outputs=["y"])
        def any_op(x):
            return ops.any(x)

        input_arr = np.array([[0, 2], [0, 0]], dtype=np.int32)
        _, verification = run_op_test(
            any_op,
            inputs={"x": input_arr},
            expected_output_name="y",
            test_device=True,
            device_duration=10,
        )

        assert verification.success, verification.error_message
