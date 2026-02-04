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
"""Tests for the NMS (Non-Maximum Suppression) operator."""

import numpy as np
import pytest

from securemr.py2smr import trace, ops
from .conftest import run_op_test, skip_if_no_device


class TestNMS:
    """Tests for NMS operation."""

    def test_basic_nms(self):
        """Test basic NMS with overlapping boxes."""
        @trace(inputs=["boxes", "scores"], outputs=["output"])
        def basic_nms(boxes, scores):
            return ops.nms(boxes, scores, threshold=0.5, output_name="output")

        # Create overlapping boxes
        boxes = np.array([
            [0, 0, 10, 10],    # Box 0
            [1, 1, 11, 11],    # Box 1 - overlaps with 0
            [50, 50, 60, 60],  # Box 2 - no overlap
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

        result, verification = run_op_test(
            basic_nms,
            {"boxes": boxes, "scores": scores},
            "output",
        )

        assert verification.success, verification.error_message
        # Box 0 should be kept (highest score), Box 1 suppressed, Box 2 kept
        assert 0 in result
        assert 2 in result

    def test_no_overlap(self):
        """Test NMS with non-overlapping boxes."""
        @trace(inputs=["boxes", "scores"], outputs=["output"])
        def no_overlap_nms(boxes, scores):
            return ops.nms(boxes, scores, threshold=0.5, output_name="output")

        boxes = np.array([
            [0, 0, 10, 10],
            [20, 20, 30, 30],
            [40, 40, 50, 50],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

        result, verification = run_op_test(
            no_overlap_nms,
            {"boxes": boxes, "scores": scores},
            "output",
        )

        assert verification.success, verification.error_message
        # All boxes should be kept
        assert len(result) == 3

    def test_high_threshold(self):
        """Test NMS with high IoU threshold (keeps more boxes)."""
        @trace(inputs=["boxes", "scores"], outputs=["output"])
        def high_thresh_nms(boxes, scores):
            return ops.nms(boxes, scores, threshold=0.9, output_name="output")

        boxes = np.array([
            [0, 0, 10, 10],
            [1, 1, 11, 11],  # Overlaps but IoU < 0.9
            [2, 2, 12, 12],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

        result, verification = run_op_test(
            high_thresh_nms,
            {"boxes": boxes, "scores": scores},
            "output",
        )

        assert verification.success, verification.error_message

    def test_low_threshold(self):
        """Test NMS with low IoU threshold (suppresses more boxes)."""
        @trace(inputs=["boxes", "scores"], outputs=["output"])
        def low_thresh_nms(boxes, scores):
            return ops.nms(boxes, scores, threshold=0.1, output_name="output")

        boxes = np.array([
            [0, 0, 10, 10],
            [5, 5, 15, 15],  # Partial overlap
            [50, 50, 60, 60],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

        result, verification = run_op_test(
            low_thresh_nms,
            {"boxes": boxes, "scores": scores},
            "output",
        )

        assert verification.success, verification.error_message

    def test_single_box(self):
        """Test NMS with single box."""
        @trace(inputs=["boxes", "scores"], outputs=["output"])
        def single_nms(boxes, scores):
            return ops.nms(boxes, scores, threshold=0.5, output_name="output")

        boxes = np.array([[10, 10, 20, 20]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)

        result, verification = run_op_test(
            single_nms,
            {"boxes": boxes, "scores": scores},
            "output",
        )

        assert verification.success, verification.error_message
        assert len(result) == 1
        assert result[0] == 0

    @skip_if_no_device
    def test_nms_on_device(self):
        """Test NMS operation on device."""
        @trace(inputs=["boxes", "scores"], outputs=["output"])
        def device_nms(boxes, scores):
            return ops.nms(boxes, scores, threshold=0.5, output_name="output")

        boxes = np.array([
            [0, 0, 10, 10],
            [1, 1, 11, 11],
            [50, 50, 60, 60],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

        result, verification = run_op_test(
            device_nms,
            {"boxes": boxes, "scores": scores},
            "output",
            test_device=True,
            device_duration=10,
        )

        assert verification.success, verification.error_message

