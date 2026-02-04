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
"""Tests for elementwise_min operator (ELEMENTWISE_MIN)."""

import numpy as np
import pytest

from securemr.py2smr import trace, ops, convert, verify
from .conftest import run_op_test, skip_if_no_device, DEVICE_AVAILABLE


class TestElementwiseMinOp:
    """Tests for the elementwise_min operation."""

    def test_elementwise_min_basic(self):
        """Test basic elementwise minimum."""
        @trace(inputs=["a", "b"], outputs=["y"])
        def min_op(a, b):
            return ops.elementwise_min(a, b, output_name="y")

        a = np.array([[1.0, 5.0], [3.0, 2.0]], dtype=np.float32)
        b = np.array([[2.0, 3.0], [4.0, 1.0]], dtype=np.float32)

        result, verification = run_op_test(
            min_op,
            inputs={"a": a, "b": b},
            expected_output_name="y",
        )

        assert verification.success, verification.error_message
        expected = np.minimum(a, b)
        np.testing.assert_allclose(result, expected)

    def test_elementwise_min_same_values(self):
        """Test elementwise minimum with same values."""
        @trace(inputs=["a", "b"], outputs=["y"])
        def min_op(a, b):
            return ops.elementwise_min(a, b, output_name="y")

        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        result, verification = run_op_test(
            min_op,
            inputs={"a": a, "b": b},
            expected_output_name="y",
        )

        assert verification.success, verification.error_message
        np.testing.assert_allclose(result, a)

    def test_elementwise_min_negative(self):
        """Test elementwise minimum with negative values."""
        @trace(inputs=["a", "b"], outputs=["y"])
        def min_op(a, b):
            return ops.elementwise_min(a, b, output_name="y")

        a = np.array([[-1.0, 5.0], [-3.0, 2.0]], dtype=np.float32)
        b = np.array([[2.0, -3.0], [4.0, -1.0]], dtype=np.float32)

        result, verification = run_op_test(
            min_op,
            inputs={"a": a, "b": b},
            expected_output_name="y",
        )

        assert verification.success, verification.error_message
        expected = np.minimum(a, b)
        np.testing.assert_allclose(result, expected)

    def test_elementwise_min_3d(self):
        """Test elementwise minimum with 3D tensors."""
        @trace(inputs=["a", "b"], outputs=["y"])
        def min_op(a, b):
            return ops.elementwise_min(a, b, output_name="y")

        a = np.random.randn(4, 4, 3).astype(np.float32)
        b = np.random.randn(4, 4, 3).astype(np.float32)

        result, verification = run_op_test(
            min_op,
            inputs={"a": a, "b": b},
            expected_output_name="y",
        )

        assert verification.success, verification.error_message
        expected = np.minimum(a, b)
        np.testing.assert_allclose(result, expected)

    @skip_if_no_device
    def test_elementwise_min_on_device(self):
        """Test elementwise_min operation on device."""
        @trace(inputs=["a", "b"], outputs=["y"])
        def min_op(a, b):
            return ops.elementwise_min(a, b, output_name="y")

        a = np.array([[1.0, 5.0], [3.0, 2.0]], dtype=np.float32)
        b = np.array([[2.0, 3.0], [4.0, 1.0]], dtype=np.float32)

        result, verification = run_op_test(
            min_op,
            inputs={"a": a, "b": b},
            expected_output_name="y",
            test_device=True,
            device_duration=10,
        )

        assert verification.success, verification.error_message
