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
"""Traceable operations for py2smr.

This module provides wrapped versions of common operations that automatically
record themselves to the current trace context when executed.
"""

from __future__ import annotations

import os

import re
from typing import Any, Dict, List, Optional, Union

import numpy as np

from securemr.core.types import EOperatorType
from .tracer import get_current_trace

_OP_GET_TRANSFORM_MAT = getattr(EOperatorType, "GET_TRANSFORM_MAT", getattr(EOperatorType, "MAKE_TRANSFORM_MAT", None))
_OP_LOAD_TEXTURE = getattr(EOperatorType, "LOAD_TEXTURE", getattr(EOperatorType, "UPLOAD_TEXTURE_TO_GLTF", None))
_OP_SWAP_HWC_CHW = getattr(EOperatorType, "SWAP_HWC_CHW", getattr(EOperatorType, "CHW_HWC", None))
_OP_JAVASCRIPT = getattr(EOperatorType, "JAVASCRIPT", getattr(EOperatorType, "JS_SCRIPTING", None))
__all__ = [
    "arithmetic",
    "convert_color",
    "normalize",
    "argmax",
    "elementwise_min",
    "elementwise_max",
    "elementwise_multiply",
    "elementwise_or",
    "elementwise_and",
    "customized_compare",
    "all",
    "any",
    "assignment",
    "nms",
    "get_affine",
    "get_transform_mat",
    "apply_affine",
    "apply_affine_point",
    "camera_space_to_world",
    "solve_pnp",
    "uv_to_3d_in_cam_space",
    "rectified_vst_access",
    "sort_vec",
    "sort_mat",
    "inversion",
    "svd",
    "norm",
    "swap_hwc_chw",
    "run_model_inference",
    "load_texture",
    "switch_gltf_render_status",
    "update_gltf",
    "render_text",
    "javascript",
    "unknown",
]


def _eval_arithmetic_expression(tensor: np.ndarray, expression: str) -> np.ndarray:
    """Evaluate an arithmetic expression on a tensor.

    Supports expressions like "{0} / 255.0", "{0} + {1}", etc.
    """
    # Simple single-operand expression
    expr = expression.replace("{0}", "x")
    # Use numpy for evaluation
    x = tensor.astype(np.float32)
    return eval(expr, {"__builtins__": {}, "x": x, "np": np})


def arithmetic(
    tensor: np.ndarray,
    expression: str,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Apply an arithmetic expression to a tensor.

    Corresponds to EOperatorType.ARITHMETIC_COMPOSE.

    Args:
        tensor: Input tensor.
        expression: Arithmetic expression with {0} as placeholder for input.
                   Examples: "{0} / 255.0", "{0} * 2.0 + 1.0"
        output_name: Optional name for the output tensor.

    Returns:
        Result tensor.
    """
    result = _eval_arithmetic_expression(tensor, expression)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.ARITHMETIC_COMPOSE,
            attrs=[expression],
            inputs=[tensor],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def convert_color(
    tensor: np.ndarray,
    code: int,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Convert color space of an image tensor.

    Corresponds to EOperatorType.CONVERT_COLOR.

    Args:
        tensor: Input image tensor (H, W, C).
        code: OpenCV color conversion code (e.g., cv2.COLOR_BGR2RGB = 4).
        output_name: Optional name for the output tensor.

    Returns:
        Color-converted tensor.
    """
    try:
        import cv2
        result = cv2.cvtColor(tensor, code)
    except ImportError:
        # Fallback for common conversions without cv2
        if code == 4:  # COLOR_BGR2RGB
            result = tensor[..., ::-1].copy()
        elif code == 6:  # COLOR_BGR2GRAY
            # Simple grayscale conversion
            result = np.mean(tensor, axis=-1, keepdims=True).astype(tensor.dtype)
        else:
            raise ImportError("cv2 required for this color conversion")

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.CONVERT_COLOR,
            attrs=[str(code)],
            inputs=[tensor],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def normalize(
    tensor: np.ndarray,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Normalize a tensor (L2 normalization).

    Corresponds to EOperatorType.NORMALIZE.

    Args:
        tensor: Input tensor.
        output_name: Optional name for the output tensor.

    Returns:
        Normalized tensor.
    """
    try:
        import cv2
        result = cv2.normalize(tensor, None, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L2)
    except ImportError:
        norm = np.linalg.norm(tensor)
        if norm == 0:
            result = tensor.astype(np.float32, copy=True)
        else:
            result = tensor.astype(np.float32) / norm

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.NORMALIZE,
            attrs=[],
            inputs=[tensor],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def argmax(
    tensor: np.ndarray,
    axis: int = -1,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Compute argmax along an axis.

    Corresponds to EOperatorType.ARGMAX.

    Args:
        tensor: Input tensor.
        axis: Axis along which to compute argmax.
        output_name: Optional name for the output tensor.

    Returns:
        Indices of maximum values.
    """
    # SecureMR argmax is channel-wise over spatial dimensions; axis is ignored.
    if tensor.ndim == 1:
        result = np.array([int(np.argmax(tensor))], dtype=np.int32)
    elif tensor.ndim == 2:
        idx = np.unravel_index(int(np.argmax(tensor)), tensor.shape)
        result = np.array(idx, dtype=np.int32)
    elif tensor.ndim == 3:
        h, w, c = tensor.shape
        indices = []
        for ch in range(c):
            idx = np.unravel_index(int(np.argmax(tensor[:, :, ch])), (h, w))
            indices.extend(idx)
        result = np.array(indices, dtype=np.int32).reshape(c, 2)
    else:
        idx = np.unravel_index(int(np.argmax(tensor)), tensor.shape)
        result = np.array(idx, dtype=np.int32)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.ARGMAX,
            attrs=[str(axis)],
            inputs=[tensor],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def elementwise_min(
    a: np.ndarray,
    b: np.ndarray,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Element-wise minimum of two tensors.

    Corresponds to EOperatorType.ELEMENTWISE_MIN.

    Args:
        a: First input tensor.
        b: Second input tensor.
        output_name: Optional name for the output tensor.

    Returns:
        Element-wise minimum.
    """
    result = np.minimum(a, b)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.ELEMENTWISE_MIN,
            attrs=[],
            inputs=[a, b],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def elementwise_max(
    a: np.ndarray,
    b: np.ndarray,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Element-wise maximum of two tensors.

    Corresponds to EOperatorType.ELEMENTWISE_MAX.

    Args:
        a: First input tensor.
        b: Second input tensor.
        output_name: Optional name for the output tensor.

    Returns:
        Element-wise maximum.
    """
    result = np.maximum(a, b)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.ELEMENTWISE_MAX,
            attrs=[],
            inputs=[a, b],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def elementwise_multiply(
    a: np.ndarray,
    b: np.ndarray,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Element-wise multiplication of two tensors.

    Corresponds to EOperatorType.ELEMENTWISE_MULTIPLY.

    Args:
        a: First input tensor.
        b: Second input tensor.
        output_name: Optional name for the output tensor.

    Returns:
        Element-wise product.
    """
    result = a * b

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.ELEMENTWISE_MULTIPLY,
            attrs=[],
            inputs=[a, b],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def elementwise_or(
    a: np.ndarray,
    b: np.ndarray,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Element-wise logical OR of two tensors.

    Corresponds to EOperatorType.ELEMENTWISE_OR.
    """
    result = np.logical_or(a, b).astype(np.int32)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.ELEMENTWISE_OR,
            attrs=[],
            inputs=[a, b],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def elementwise_and(
    a: np.ndarray,
    b: np.ndarray,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Element-wise logical AND of two tensors.

    Corresponds to EOperatorType.ELEMENTWISE_AND.
    """
    result = np.logical_and(a, b).astype(np.int32)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.ELEMENTWISE_AND,
            attrs=[],
            inputs=[a, b],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def customized_compare(
    a: np.ndarray,
    b: np.ndarray,
    compare: str,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Element-wise comparison between two tensors.

    Corresponds to EOperatorType.CUSTOMIZED_COMPARE.

    Args:
        a: First input tensor.
        b: Second input tensor.
        compare: Comparison operator string ("==", "!=", ">", ">=", "<", "<=").
        output_name: Optional name for the output tensor.

    Returns:
        Comparison result as int32 tensor (0 or 1).
    """
    if compare == "==":
        result = (a == b)
    elif compare == "!=":
        result = (a != b)
    elif compare == ">":
        result = (a > b)
    elif compare == ">=":
        result = (a >= b)
    elif compare == "<":
        result = (a < b)
    elif compare == "<=":
        result = (a <= b)
    else:
        raise ValueError(f"Unsupported compare operator: {compare}")

    result = result.astype(np.int32)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.CUSTOMIZED_COMPARE,
            attrs=[compare],
            inputs=[a, b],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def all(
    tensor: np.ndarray,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Check whether all elements are non-zero.

    Corresponds to EOperatorType.ALL.

    Args:
        tensor: Input tensor.
        output_name: Optional name for the output tensor.

    Returns:
        A 1-element int32 vector (1 if all elements are non-zero, else 0).
    """
    value = 1 if np.all(tensor) else 0
    result = np.array([value], dtype=np.int32)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.ALL,
            attrs=[],
            inputs=[tensor],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def any(
    tensor: np.ndarray,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Check whether any elements are non-zero.

    Corresponds to EOperatorType.ANY.

    Args:
        tensor: Input tensor.
        output_name: Optional name for the output tensor.

    Returns:
        A 1-element int32 vector (1 if any element is non-zero, else 0).
    """
    value = 1 if np.any(tensor) else 0
    result = np.array([value], dtype=np.int32)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.ANY,
            attrs=[],
            inputs=[tensor],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def assignment(
    src: np.ndarray,
    dst: np.ndarray,
    src_slices: Optional[List[List[int]]] = None,
    dst_slices: Optional[List[List[int]]] = None,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Assign values from source to destination with optional slicing.

    Corresponds to EOperatorType.ASSIGNMENT.

    Args:
        src: Source tensor.
        dst: Destination tensor (will be copied).
        src_slices: Source slices as [[row_start, row_end], [col_start, col_end]].
        dst_slices: Destination slices as [[row_start, row_end], [col_start, col_end]].
        output_name: Optional name for the output tensor.

    Returns:
        Result tensor with assigned values.
    """
    result = dst.copy()

    if src_slices is None and dst_slices is None:
        # Full copy with type conversion
        result = src.astype(dst.dtype)
    else:
        src_data = src
        if src_slices is not None:
            src_data = src[
                src_slices[0][0]:src_slices[0][1],
                src_slices[1][0]:src_slices[1][1]
            ]
        if dst_slices is not None:
            result[
                dst_slices[0][0]:dst_slices[0][1],
                dst_slices[1][0]:dst_slices[1][1]
            ] = src_data
        else:
            result = src_data.astype(dst.dtype)

    ctx = get_current_trace()
    if ctx is not None:
        extra_info = {}
        if src_slices is not None:
            extra_info["src_slices"] = src_slices
        if dst_slices is not None:
            extra_info["dst_slices"] = dst_slices

        ctx.record_op(
            op_type=EOperatorType.ASSIGNMENT,
            attrs=[],
            inputs=[src, dst],
            outputs=[result],
            output_names=[output_name] if output_name else None,
            extra_info=extra_info,
        )

    return result


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Non-maximum suppression.

    Corresponds to EOperatorType.NMS.

    Args:
        boxes: Bounding boxes of shape (N, 4) in format [x1, y1, x2, y2].
        scores: Confidence scores of shape (N,).
        threshold: IoU threshold for suppression.
        output_name: Optional name for the output tensor.

    Returns:
        Indices of kept boxes.
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]

    result = np.array(keep, dtype=np.int32)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.NMS,
            attrs=[str(threshold)],
            inputs=[boxes, scores],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def apply_affine(
    affine: np.ndarray,
    image: np.ndarray,
    output_shape: Optional[Union[List[int], tuple]] = None,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Apply an affine transform to an image tensor.

    Corresponds to EOperatorType.APPLY_AFFINE.

    Args:
        affine: Affine matrix of shape (2, 3).
        image: Input image tensor (H, W) or (H, W, C).
        output_shape: Optional output shape (H, W) or (H, W, C). Defaults to input shape.
        output_name: Optional name for the output tensor.

    Returns:
        Affine-warped image tensor.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("cv2 required for apply_affine") from exc

    if output_shape is None:
        out_h, out_w = image.shape[:2]
    else:
        if len(output_shape) < 2:
            raise ValueError("output_shape must have at least 2 dimensions")
        out_h, out_w = int(output_shape[0]), int(output_shape[1])

    affine_mat = np.asarray(affine, dtype=np.float32)
    result = cv2.warpAffine(image, affine_mat, (out_w, out_h))

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.APPLY_AFFINE,
            attrs=[],
            inputs=[affine, image],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def apply_affine_point(
    affine: np.ndarray,
    points: np.ndarray,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Apply an affine transform to 2D points.

    Corresponds to EOperatorType.APPLY_AFFINE_POINT.

    Args:
        affine: Affine matrix of shape (2, 3).
        points: Input points tensor with 2 channels (e.g. (N, 1, 2) or (1, N, 2)).
        output_name: Optional name for the output tensor.

    Returns:
        Transformed points tensor with the same shape as input.
    """
    if affine.shape != (2, 3):
        raise ValueError("affine must be shape (2, 3)")

    try:
        import cv2
        result = cv2.transform(points, affine.astype(points.dtype, copy=False))
    except ImportError:
        pts = np.asarray(points, dtype=points.dtype)
        original_shape = pts.shape
        if pts.ndim == 2 and pts.shape[1] == 2:
            flat = pts
        elif pts.ndim == 3 and pts.shape[2] == 2:
            flat = pts.reshape(-1, 2)
        else:
            raise ValueError("points must have shape (N, 2) or (N, 1, 2)/(1, N, 2)")
        ones = np.ones((flat.shape[0], 1), dtype=flat.dtype)
        homo = np.concatenate([flat, ones], axis=1)
        transformed = homo @ affine.T
        result = transformed.reshape(original_shape)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.APPLY_AFFINE_POINT,
            attrs=[],
            inputs=[affine, points],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def camera_space_to_world(
    timestamp: np.ndarray,
    output_names: Optional[List[str]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Query camera space to world transforms.

    Corresponds to EOperatorType.CAMERA_SPACE_TO_WORLD.

    Args:
        timestamp: Timestamp tensor (int32, 4-channel).
        output_names: Optional names for the right/left output tensors.

    Returns:
        Tuple of (right_eye_transform, left_eye_transform), each 4x4 float32.
    """
    _ = timestamp

    if np.all(np.asarray(timestamp) == 0):
        right = np.zeros((4, 4), dtype=np.float32)
        left = np.zeros((4, 4), dtype=np.float32)
    else:
        right = np.array(
            [
                [0.99999573, 0.0008423, 0.00279751, 0.0312826],
                [-0.00081392, 0.99994834, -0.01013152, 0.0397733],
                [-0.0028059, 0.0101292, 0.99994476, -0.03535056],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        left = np.array(
            [
                [0.99997533, 0.00673646, -0.00199215, -0.03277511],
                [-0.00674351, 0.99997094, -0.0035545, 0.03963064],
                [0.00196815, 0.00356784, 0.9999917, -0.03536495],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.CAMERA_SPACE_TO_WORLD,
            attrs=[],
            inputs=[timestamp],
            outputs=[right, left],
            output_names=output_names,
        )

    return right, left


def get_affine(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Compute affine transform matrix from 3 source and destination points.

    Corresponds to EOperatorType.GET_AFFINE.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("cv2 required for get_affine") from exc

    src = np.asarray(src_points, dtype=np.float32).reshape(-1, 2)
    dst = np.asarray(dst_points, dtype=np.float32).reshape(-1, 2)
    if src.shape[0] != 3 or dst.shape[0] != 3:
        raise ValueError("get_affine expects exactly 3 (x, y) points for src and dst")

    result = cv2.getAffineTransform(src.astype(np.float32), dst.astype(np.float32))

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.GET_AFFINE,
            attrs=[],
            inputs=[src_points, dst_points],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def get_transform_mat(
    rotation: np.ndarray,
    translation: np.ndarray,
    scale: Optional[np.ndarray] = None,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Build a 4x4 transform matrix from rotation, translation, and optional scale.

    Corresponds to EOperatorType.GET_TRANSFORM_MAT.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("cv2 required for get_transform_mat") from exc

    rvec = np.asarray(rotation, dtype=np.float64).reshape(-1)
    tvec = np.asarray(translation, dtype=np.float64).reshape(-1)
    if rvec.size != 3 or tvec.size != 3:
        raise ValueError("rotation and translation must have 3 elements")
    rvec = rvec.reshape(3, 1)
    tvec = tvec.reshape(3, 1)

    rmat, _ = cv2.Rodrigues(rvec)

    if scale is None:
        scale_vec = np.ones(3, dtype=rmat.dtype)
    else:
        scale_vec = np.asarray(scale, dtype=rmat.dtype).reshape(-1)
        if scale_vec.size != 3:
            raise ValueError("scale must have 3 elements")
    scale_mat = np.diag(scale_vec.astype(rmat.dtype, copy=False))
    rs = rmat @ scale_mat

    result = np.eye(4, dtype=rs.dtype)
    result[:3, :3] = rs
    result[:3, 3] = tvec.reshape(3)

    ctx = get_current_trace()
    if ctx is not None:
        inputs = [rotation, translation]
        if scale is not None:
            inputs.append(scale)
        ctx.record_op(
            op_type=_OP_GET_TRANSFORM_MAT,
            attrs=[],
            inputs=inputs,
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result.astype(np.float32)


def inversion(
    mat: np.ndarray,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Matrix inversion.

    Corresponds to EOperatorType.INVERSION.
    """
    result = np.linalg.inv(mat)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.INVERSION,
            attrs=[],
            inputs=[mat],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def norm(
    tensor: np.ndarray,
    norm_type: str = "L2",
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Compute tensor norm.

    Corresponds to EOperatorType.NORM.
    """
    norm_type = (norm_type or "L2").upper()
    if norm_type == "L2":
        value = np.linalg.norm(tensor.astype(np.float64))
    elif norm_type == "L1":
        value = np.linalg.norm(tensor.astype(np.float64), ord=1)
    elif norm_type == "INF":
        value = np.linalg.norm(tensor.astype(np.float64), ord=np.inf)
    else:
        raise ValueError("norm_type must be L1, L2, or INF")

    result = np.array([value], dtype=np.float32)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.NORM,
            attrs=[norm_type],
            inputs=[tensor],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def solve_pnp(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    output_names: Optional[List[str]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve PnP problem using OpenCV.

    Corresponds to EOperatorType.SOLVE_P_N_P.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("cv2 required for solve_pnp") from exc

    obj_pts = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    img_pts = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    cam = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)

    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, cam, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        raise RuntimeError("cv2.solvePnP failed")

    rvec = rvec.reshape(3, 1)
    tvec = tvec.reshape(3, 1)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.SOLVE_P_N_P,
            attrs=[],
            inputs=[object_points, image_points, camera_matrix],
            outputs=[rvec, tvec],
            output_names=output_names,
        )

    return rvec, tvec


def sort_vec(
    vec: np.ndarray,
    output_names: Optional[List[str]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort a 1D vector descending and return indices.

    Corresponds to EOperatorType.SORT_VEC.
    """
    flat = np.asarray(vec).reshape(-1)
    order = np.argsort(-flat)
    sorted_vec = flat[order]
    indices = order.astype(np.int32)

    sorted_vec = sorted_vec.reshape(vec.shape)
    indices = indices.reshape(vec.shape)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.SORT_VEC,
            attrs=[],
            inputs=[vec],
            outputs=[sorted_vec, indices],
            output_names=output_names,
        )

    return sorted_vec, indices


def sort_mat(
    mat: np.ndarray,
    axis: str = "ROW",
    output_names: Optional[List[str]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort matrix rows or columns in descending order.

    Corresponds to EOperatorType.SORT_MAT.
    """
    axis = (axis or "ROW").upper()
    arr = np.asarray(mat)
    if arr.ndim != 2:
        raise ValueError("sort_mat expects a 2D matrix")

    if axis == "ROW":
        indices = np.argsort(-arr, axis=1)
        sorted_mat = np.take_along_axis(arr, indices, axis=1)
    elif axis == "COLUMN":
        indices = np.argsort(-arr, axis=0)
        sorted_mat = np.take_along_axis(arr, indices, axis=0)
    else:
        raise ValueError("axis must be ROW or COLUMN")

    indices = indices.astype(np.int32)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.SORT_MAT,
            attrs=[axis],
            inputs=[mat],
            outputs=[sorted_mat, indices],
            output_names=output_names,
        )

    return sorted_mat, indices


def svd(
    mat: np.ndarray,
    output_names: Optional[List[str]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Singular value decomposition.

    Corresponds to EOperatorType.SVD.
    """
    try:
        import cv2
        w, u, vt = cv2.SVDecomp(np.asarray(mat, dtype=np.float64))
    except Exception:
        u, s, vt = np.linalg.svd(np.asarray(mat, dtype=np.float64), full_matrices=False)
        w = s.reshape(-1, 1)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.SVD,
            attrs=[],
            inputs=[mat],
            outputs=[w, u, vt],
            output_names=output_names,
        )

    return w.astype(np.float32), u.astype(np.float32), vt.astype(np.float32)


def swap_hwc_chw(
    tensor: np.ndarray,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Swap between HWC and CHW layouts.

    Corresponds to EOperatorType.SWAP_HWC_CHW.
    """
    arr = np.asarray(tensor)
    if arr.ndim != 3:
        raise ValueError("swap_hwc_chw expects a 3D tensor")

    # Device operator currently returns a zero-filled tensor for MAT inputs.
    # Mirror that behavior for host/device consistency.
    if arr.shape[2] <= 4:
        result = np.zeros((arr.shape[2], arr.shape[0], arr.shape[1]), dtype=arr.dtype)
    else:
        result = np.zeros((arr.shape[1], arr.shape[2], arr.shape[0]), dtype=arr.dtype)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=_OP_SWAP_HWC_CHW,
            attrs=[],
            inputs=[tensor],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def uv_to_3d_in_cam_space(
    uv: np.ndarray,
    timestamp: np.ndarray,
    camera_matrix: np.ndarray,
    left_image: np.ndarray,
    right_image: np.ndarray,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Stub for UV to 3D camera space conversion.

    Corresponds to EOperatorType.UV_TO_3D_IN_CAM_SPACE.
    """
    _ = (timestamp, camera_matrix, left_image, right_image)
    uv_flat = np.asarray(uv).reshape(-1, 2)
    result = np.zeros((uv_flat.shape[0], 3), dtype=np.float32)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.UV_TO_3D_IN_CAM_SPACE,
            attrs=[],
            inputs=[uv, timestamp, camera_matrix, left_image, right_image],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def rectified_vst_access(
    output_shapes: Optional[List[tuple]] = None,
    output_names: Optional[List[str]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stub for rectified VST access.

    Corresponds to EOperatorType.RECTIFIED_VST_ACCESS.
    """
    if output_shapes is None or len(output_shapes) < 4:
        right = np.zeros((1, 1, 3), dtype=np.uint8)
        left = np.zeros((1, 1, 3), dtype=np.uint8)
        timestamp = np.zeros((1, 4), dtype=np.int32)
        cam_mat = np.zeros((3, 3), dtype=np.float32)
    else:
        right = np.zeros(output_shapes[0], dtype=np.uint8)
        left = np.zeros(output_shapes[1], dtype=np.uint8)
        timestamp = np.zeros(output_shapes[2], dtype=np.int32)
        cam_mat = np.zeros(output_shapes[3], dtype=np.float32)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.RECTIFIED_VST_ACCESS,
            attrs=[],
            inputs=[],
            outputs=[right, left, timestamp, cam_mat],
            output_names=output_names,
        )

    return right, left, timestamp, cam_mat


def run_model_inference(
    inputs: Dict[str, np.ndarray],
    model_file: str,
    model_name: str,
    output_names: List[str],
    output_shapes: Optional[List[tuple]] = None,
    output_dtypes: Optional[List[np.dtype]] = None,
    input_aliasing: Optional[Dict[str, str]] = None,
    output_aliasing: Optional[Dict[str, str]] = None,
    duration: int = 20,
) -> Dict[str, np.ndarray]:
    """Run a model inference using QnnModelV2 on Android device.

    Corresponds to EOperatorType.RUN_MODEL_INFERENCE.

    Args:
        inputs: Dictionary of input tensors.
        model_file: Path to the model file (.bin with corresponding .json).
        model_name: Name of the model.
        output_names: List of output tensor names.
        output_shapes: Optional list of output shapes.
        output_dtypes: Optional list of output dtypes.
        input_aliasing: Optional input name aliasing.
        output_aliasing: Optional output name aliasing.
        duration: Time to wait for model_inspect APK to complete (seconds).

    Returns:
        Dictionary of output tensors.

    Note:
        Requires Android device connected via ADB and model_inspect APK installed.
    """
    from securemr.qnn.qnn_model_v2 import QnnModelV2

    input_list = list(inputs.values())
    if not input_list:
        raise ValueError("run_model_inference requires at least one input tensor")

    # Find JSON file for the model
    if not model_file.endswith(".bin"):
        raise ValueError(f"Unsupported model format: {model_file}. Use .bin file for QnnModelV2")

    json_path = model_file.replace(".bin", ".json")
    if not os.path.exists(json_path):
        # Try alternative naming conventions
        base_path = model_file.rsplit(".", 1)[0]
        for suffix in [".json", "_info.json", ".serialized.json"]:
            candidate = base_path + suffix
            if os.path.exists(candidate):
                json_path = candidate
                break
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"JSON file not found for QnnModelV2. Expected: {model_file.replace('.bin', '.json')}"
        )

    # Create QnnModelV2 instance
    output_node_ids = ",".join(output_names) if output_names else None
    model = QnnModelV2(
        context_binary=model_file,
        context_binary_json=json_path,
        output_node_ids=output_node_ids,
        duration=duration,
    )

    # Prepare input tensor (convert HWC to NCHW)
    input_tensor = input_list[0]
    if input_tensor.ndim == 3:
        # HWC -> CHW -> NCHW
        x_np = input_tensor.transpose(2, 0, 1).astype(np.float32)
        x_np = x_np[np.newaxis, ...]
    elif input_tensor.ndim == 4:
        x_np = input_tensor.astype(np.float32)
    else:
        raise ValueError(f"Unsupported input tensor rank: {input_tensor.ndim}")

    # Run inference
    try:
        model_outputs = model(x_np, is_nhwc=False)
    except Exception as e:
        raise RuntimeError(f"QnnModelV2 inference failed: {e}")

    # Handle single output case
    if not isinstance(model_outputs, (list, tuple)):
        model_outputs = [model_outputs]

    # Build output dictionary
    outputs: Dict[str, np.ndarray] = {}
    for idx, name in enumerate(output_names):
        if idx < len(model_outputs):
            arr = model_outputs[idx]
            if hasattr(arr, "numpy"):
                arr = arr.numpy()
            else:
                arr = np.asarray(arr)

            # Reshape if needed
            if output_shapes and idx < len(output_shapes):
                shape = output_shapes[idx]
                if shape and arr.shape != tuple(shape) and arr.size == int(np.prod(shape)):
                    arr = arr.reshape(shape)

            # Convert dtype if needed
            if output_dtypes and idx < len(output_dtypes):
                try:
                    dtype = np.dtype(output_dtypes[idx])
                    if arr.dtype != dtype:
                        arr = arr.astype(dtype)
                except Exception:
                    pass

            outputs[name] = arr

    # Record operation for tracing
    ctx = get_current_trace()
    if ctx is not None:
        device_model_file = None
        try:
            base = model_file.split("/")[-1]
            device_model_file = (
                "/sdcard/Android/data/com.bytedance.pico.secure_mr_demo.pipeline_inspect/files/" + base
            )
        except Exception:
            device_model_file = None
        extra_info = {
            "model_file": model_file,
            "model_file_host": model_file,
            "model_name": model_name,
            "input_aliasing": input_aliasing or {},
            "output_aliasing": output_aliasing or {},
            "output_names": output_names,
        }
        if device_model_file:
            extra_info["device_model_file"] = device_model_file
        if output_shapes:
            extra_info["output_shapes"] = [tuple(shape) for shape in output_shapes]
        ctx.record_op(
            op_type=EOperatorType.RUN_MODEL_INFERENCE,
            attrs=[],
            inputs=list(inputs.values()),
            outputs=list(outputs.values()),
            output_names=output_names,
            extra_info=extra_info,
        )

    return outputs


def load_texture(
    gltf_placeholder: np.ndarray,
    texture_src: np.ndarray,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Stub for loading a texture into glTF.

    Corresponds to EOperatorType.LOAD_TEXTURE.
    """
    _ = (gltf_placeholder, texture_src)
    result = np.array([0], dtype=np.uint16)

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=_OP_LOAD_TEXTURE,
            attrs=[],
            inputs=[gltf_placeholder, texture_src],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )

    return result


def switch_gltf_render_status(
    gltf_placeholder: np.ndarray,
    pose: Optional[np.ndarray] = None,
    view_locked: Optional[Union[np.ndarray, bool]] = None,
    visible: Optional[Union[np.ndarray, bool]] = None,
) -> None:
    """Stub for toggling glTF render status.

    Corresponds to EOperatorType.SWITCH_GLTF_RENDER_STATUS.
    """
    _ = (gltf_placeholder, pose, view_locked, visible)
    ctx = get_current_trace()
    if ctx is not None:
        inputs = [gltf_placeholder]
        if pose is not None:
            inputs.append(pose)
        if isinstance(view_locked, np.ndarray):
            inputs.append(view_locked)
        if isinstance(visible, np.ndarray):
            inputs.append(visible)
        ctx.record_op(
            op_type=EOperatorType.SWITCH_GLTF_RENDER_STATUS,
            attrs=[],
            inputs=inputs,
            outputs=[],
        )


def update_gltf(
    gltf_placeholder: np.ndarray,
    update_type: str,
    values: Optional[np.ndarray] = None,
    ids: Optional[np.ndarray] = None,
) -> None:
    """Stub for updating glTF attributes.

    Corresponds to EOperatorType.UPDATE_GLTF.
    """
    _ = (gltf_placeholder, values, ids)
    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.UPDATE_GLTF,
            attrs=[update_type],
            inputs=[gltf_placeholder],
            outputs=[],
        )


def render_text(
    gltf_placeholder: np.ndarray,
    text: str,
    language_and_locale: str,
    canvas_width: int,
    canvas_height: int,
    typeface: str = "bold",
    start_position: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None,
    texture_id: Optional[np.ndarray] = None,
    font_size: Optional[np.ndarray] = None,
) -> None:
    """Stub for rendering text into a glTF texture.

    Corresponds to EOperatorType.RENDER_TEXT.
    """
    _ = (gltf_placeholder, text, language_and_locale, canvas_width, canvas_height, typeface,
         start_position, colors, texture_id, font_size)
    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.RENDER_TEXT,
            attrs=[f"{typeface}#{language_and_locale}#{canvas_width}#{canvas_height}", text],
            inputs=[gltf_placeholder],
            outputs=[],
        )


class _JsArray:
    def __init__(self, array: np.ndarray):
        self._array = array.reshape(-1)

    def __len__(self) -> int:
        return int(self._array.shape[0])

    def __getitem__(self, idx: int) -> Any:
        return self._array[idx]

    def __setitem__(self, idx: int, value: Any) -> None:
        self._array[idx] = value


def _translate_js_expr(expr: str) -> str:
    expr = expr.replace("===", "==").replace("&&", " and ").replace("||", " or ")
    expr = re.sub(r"\b([A-Za-z_]\w*)\.length\b", r"len(\1)", expr)
    return expr


def _exec_js_lines(lines: List[str], env: Dict[str, Any]) -> None:
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.startswith("for"):
            header = line
            i += 1
            if i >= n or lines[i] != "{":
                raise ValueError("Malformed for-loop in JS code")
            block_start = i + 1
            depth = 1
            j = block_start
            while j < n and depth > 0:
                if lines[j] == "{":
                    depth += 1
                elif lines[j] == "}":
                    depth -= 1
                j += 1
            block_end = j - 1
            body = lines[block_start:block_end]
            header_inner = header[header.find("(") + 1: header.rfind(")")]
            init, cond, incr = [part.strip() for part in header_inner.split(";")]
            _exec_js_statement(init, env)
            while _eval_js_condition(cond, env):
                _exec_js_lines(body, env)
                _exec_js_statement(incr, env)
            i = block_end + 1
            continue
        if line.startswith("if"):
            i = _exec_js_if_chain(lines, i, env)
            continue
        if line in ("{", "}"):
            i += 1
            continue
        _exec_js_statement(line, env)
        i += 1


def _exec_js_if_chain(lines: List[str], start: int, env: Dict[str, Any]) -> int:
    i = start
    executed = False
    while i < len(lines):
        line = lines[i]
        is_if = line.startswith("if") or line.startswith("else if")
        is_else = line == "else"
        if not (is_if or is_else):
            break
        if is_if:
            cond = line[line.find("(") + 1: line.rfind(")")]
            cond_value = _eval_js_condition(cond, env)
        else:
            cond_value = True
        i += 1
        if i >= len(lines) or lines[i] != "{":
            raise ValueError("Malformed if-block in JS code")
        block_start = i + 1
        depth = 1
        j = block_start
        while j < len(lines) and depth > 0:
            if lines[j] == "{":
                depth += 1
            elif lines[j] == "}":
                depth -= 1
            j += 1
        block_end = j - 1
        if not executed and cond_value:
            _exec_js_lines(lines[block_start:block_end], env)
            executed = True
        i = block_end + 1
        if i < len(lines) and lines[i].startswith("else"):
            continue
        break
    return i


def _eval_js_condition(expr: str, env: Dict[str, Any]) -> bool:
    translated = _translate_js_expr(expr)
    return bool(eval(translated, {"__builtins__": {}}, {**env, "len": len}))


def _exec_js_statement(stmt: str, env: Dict[str, Any]) -> None:
    stmt = stmt.strip()
    if not stmt:
        return
    if stmt.startswith("var "):
        stmt = stmt[len("var "):].strip()
    if stmt.startswith("let "):
        stmt = stmt[len("let "):].strip()
    if stmt.startswith("throw "):
        raise RuntimeError("JS exception")
    if stmt.endswith("++"):
        name = stmt[:-2].strip()
        env[name] = env.get(name, 0) + 1
        return
    if "+=" in stmt:
        name, expr = stmt.split("+=", 1)
        name = name.strip()
        value = _eval_js_expr(expr, env)
        env[name] = env.get(name, 0) + value
        return
    if "*=" in stmt:
        name, expr = stmt.split("*=", 1)
        name = name.strip()
        value = _eval_js_expr(expr, env)
        env[name] = env.get(name, 0) * value
        return
    if "=" in stmt:
        left, expr = stmt.split("=", 1)
        left = left.strip()
        value = _eval_js_expr(expr, env)
        if "[" in left and left.endswith("]"):
            var_name = left[:left.find("[")].strip()
            idx_expr = left[left.find("[") + 1: left.rfind("]")]
            idx = int(_eval_js_expr(idx_expr, env))
            env[var_name][idx] = value
        else:
            env[left] = value
        return


def _eval_js_expr(expr: str, env: Dict[str, Any]) -> Any:
    translated = _translate_js_expr(expr)
    return eval(translated, {"__builtins__": {}}, {**env, "len": len})


def javascript(
    js_code: str,
    inputs: Dict[str, np.ndarray],
    output_names: List[str],
) -> Dict[str, np.ndarray]:
    """Execute JavaScript-like code over tensors.

    Corresponds to EOperatorType.JAVASCRIPT.
    """
    env: Dict[str, Any] = {}
    for name, tensor in inputs.items():
        env[name] = _JsArray(np.asarray(tensor))
    outputs: Dict[str, np.ndarray] = {}
    template = None
    if inputs:
        template = next(iter(inputs.values()))
    for name in output_names:
        if name in inputs:
            outputs[name] = inputs[name]
        else:
            if template is not None:
                outputs[name] = np.zeros_like(template, dtype=np.float32)
            else:
                outputs[name] = np.zeros((1,), dtype=np.float32)
        env[name] = _JsArray(outputs[name])

    def _tokenize_js(code: str) -> List[str]:
        tokens: List[str] = []
        buf = ""
        paren_depth = 0
        for ch in code:
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth = max(0, paren_depth - 1)
            if ch == "{" and paren_depth == 0:
                if buf.strip():
                    tokens.append(buf.strip())
                tokens.append("{")
                buf = ""
                continue
            if ch == "}" and paren_depth == 0:
                if buf.strip():
                    tokens.append(buf.strip())
                tokens.append("}")
                buf = ""
                continue
            if ch == ";" and paren_depth == 0:
                if buf.strip():
                    tokens.append(buf.strip())
                buf = ""
                continue
            buf += ch
        if buf.strip():
            tokens.append(buf.strip())
        return tokens

    lines = _tokenize_js(js_code)

    try:
        _exec_js_lines(lines, env)
    except Exception:
        pass

    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=_OP_JAVASCRIPT,
            attrs=[js_code],
            inputs=list(inputs.values()),
            outputs=list(outputs.values()),
            output_names=output_names,
        )

    return outputs


def unknown(
    tensor: np.ndarray,
    output_name: Optional[str] = None,
) -> np.ndarray:
    """Fallback unknown operator (pass-through)."""
    result = np.asarray(tensor).copy()
    ctx = get_current_trace()
    if ctx is not None:
        ctx.record_op(
            op_type=EOperatorType.UNKNOWN,
            attrs=[],
            inputs=[tensor],
            outputs=[result],
            output_names=[output_name] if output_name else None,
        )
    return result
