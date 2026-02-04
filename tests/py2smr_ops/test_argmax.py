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
"""Tests for argmax operator (ARGMAX)."""

import numpy as np
import pytest

from securemr.py2smr import trace, ops, convert, verify
from .conftest import run_op_test, skip_if_no_device, DEVICE_AVAILABLE


class TestArgmaxOp:
    """Tests for the argmax operation."""

    def test_argmax_basic(self):
        """Test basic argmax along last axis."""
        @trace(inputs=["x"], outputs=["y"])
        def find_max(x):
            return ops.argmax(x, axis=-1)

        input_arr = np.array([
            [1.0, 3.0, 2.0],
            [5.0, 1.0, 4.0],
        ], dtype=np.float32)

        result, verification = run_op_test(
            find_max,
            inputs={"x": input_arr},
            expected_output_name="y",
        )

        assert verification.success, verification.error_message
        expected = np.array([1, 0], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_argmax_single_row(self):
        """Test argmax with single row."""
        @trace(inputs=["x"], outputs=["y"])
        def find_max(x):
            return ops.argmax(x, axis=-1)

        input_arr = np.array([[0.1, 0.7, 0.2]], dtype=np.float32)
        result, verification = run_op_test(
            find_max,
            inputs={"x": input_arr},
            expected_output_name="y",
        )

        assert verification.success, verification.error_message
        expected = np.array([0, 1], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_argmax_classification(self):
        """Test argmax for classification output."""
        @trace(inputs=["logits"], outputs=["predictions"])
        def classify(logits):
            return ops.argmax(logits, axis=-1)

        # Simulate classification logits for 4 samples, 10 classes
        input_arr = np.random.randn(4, 10).astype(np.float32)
        result, verification = run_op_test(
            classify,
            inputs={"logits": input_arr},
            expected_output_name="predictions",
        )

        assert verification.success, verification.error_message
        idx = np.unravel_index(int(np.argmax(input_arr)), input_arr.shape)
        expected = np.array(idx, dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_argmax_3d_tensor(self):
        """Test argmax with 3D tensor."""
        @trace(inputs=["x"], outputs=["y"])
        def find_max(x):
            return ops.argmax(x, axis=-1)

        input_arr = np.random.randn(2, 3, 4).astype(np.float32)
        result, verification = run_op_test(
            find_max,
            inputs={"x": input_arr},
            expected_output_name="y",
        )

        assert verification.success, verification.error_message
        expected = []
        for ch in range(input_arr.shape[2]):
            idx = np.unravel_index(int(np.argmax(input_arr[:, :, ch])), input_arr.shape[:2])
            expected.extend(idx)
        expected = np.array(expected, dtype=np.int32).reshape(input_arr.shape[2], 2)
        np.testing.assert_array_equal(result, expected)

    @skip_if_no_device
    def test_argmax_on_device(self):
        """Test argmax operation on device."""
        @trace(inputs=["x"], outputs=["y"])
        def find_max(x):
            return ops.argmax(x, axis=-1)

        input_arr = np.array([
            [1.0, 3.0, 2.0],
            [5.0, 1.0, 4.0],
        ], dtype=np.float32)

        result, verification = run_op_test(
            find_max,
            inputs={"x": input_arr},
            expected_output_name="y",
            test_device=True,
            device_duration=10,
        )

        assert verification.success, verification.error_message
