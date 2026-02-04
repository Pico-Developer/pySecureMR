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
"""Tests for apply_affine operator (APPLY_AFFINE)."""

import numpy as np
import pytest

from securemr.py2smr import trace, ops
from .conftest import run_op_test, skip_if_no_device


try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    CV2_AVAILABLE = False


skip_if_no_cv2 = pytest.mark.skipif(
    not CV2_AVAILABLE,
    reason="cv2 required for apply_affine",
)


class TestApplyAffineOp:
    """Tests for the apply_affine operation."""

    @skip_if_no_cv2
    def test_apply_affine_identity(self):
        """Test identity affine transform."""
        @trace(inputs=["affine", "image"], outputs=["result"])
        def apply_identity(affine, image):
            return ops.apply_affine(affine, image, output_shape=image.shape[:2])

        image = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        result, verification = run_op_test(
            apply_identity,
            inputs={"affine": affine, "image": image},
            expected_output_name="result",
        )

        assert verification.success, verification.error_message
        expected = cv2.warpAffine(image, affine, (image.shape[1], image.shape[0]))
        np.testing.assert_array_equal(result, expected)

    @skip_if_no_cv2
    def test_apply_affine_scale_down(self):
        """Test affine scaling with smaller output size."""
        @trace(inputs=["affine", "image"], outputs=["result"])
        def apply_scale(affine, image):
            return ops.apply_affine(affine, image, output_shape=(6, 6))

        image = np.random.randint(0, 255, (12, 12, 3), dtype=np.uint8)
        affine = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=np.float32)

        result, verification = run_op_test(
            apply_scale,
            inputs={"affine": affine, "image": image},
            expected_output_name="result",
        )

        assert verification.success, verification.error_message
        expected = cv2.warpAffine(image, affine, (6, 6))
        np.testing.assert_array_equal(result, expected)

    @skip_if_no_cv2
    @skip_if_no_device
    def test_apply_affine_on_device(self):
        """Test apply_affine on device."""
        @trace(inputs=["affine", "image"], outputs=["result"])
        def apply_identity(affine, image):
            return ops.apply_affine(affine, image, output_shape=image.shape[:2])

        image = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        result, verification = run_op_test(
            apply_identity,
            inputs={"affine": affine, "image": image},
            expected_output_name="result",
            test_device=True,
            device_duration=10,
        )

        assert verification.success, verification.error_message
