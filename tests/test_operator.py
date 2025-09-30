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

import cv2
import numpy as np
import os
import math
import pytest
import securemr as smr
from securemr.serialization import Pipeline as SerializablePipeline
from securemr.serialization import DeserializedPipeline
from securemr.serialization import mat_flag, convert_from_dtype

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_cvt_color():
    configs = [str(cv2.COLOR_BGR2GRAY)]
    op = smr.OperatorFactory.create(smr.EOperatorType.CONVERT_COLOR, configs)
    assert op is not None
    assert op.get_operator_type() == smr.EOperatorType.CONVERT_COLOR
    assert op.get_operand_cnt() == 1
    assert op.get_results_cnt() == 1

    img = cv2.imread("tests/data/dog.jpg")
    x = smr.TensorMat.from_numpy(img)
    y = smr.TensorMat(img.shape[:2], 1, smr.EDataType.UINT8)

    op.data_as_operand(x, 0)
    op.connect_result_to_data_array(0, y)
    op.compute(0)

    gray = y.numpy()
    assert gray.ndim == 3
    assert gray.shape == (*img.shape[:2], 1)
    np.testing.assert_array_equal(gray[:,:,0], cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))


def test_affine_image():
    op1 = smr.OperatorFactory.create(smr.EOperatorType.GET_AFFINE)
    op2 = smr.OperatorFactory.create(smr.EOperatorType.APPLY_AFFINE)
    assert op1 and op2

    assert op1.get_operator_type() == smr.EOperatorType.GET_AFFINE
    assert op1.get_operand_cnt() == 2
    assert op1.get_results_cnt() == 1

    assert op2.get_operator_type() == smr.EOperatorType.APPLY_AFFINE
    assert op2.get_operand_cnt() == 2
    assert op2.get_results_cnt() == 1

    image_width = 3248
    image_height = 2464
    crop_x1 = 1444
    crop_y1 = 1332
    crop_x2 = 2045
    crop_y2 = 1933
    crop_width = 224
    crop_height = 224

    src_points = smr.TensorPoint2Float.from_numpy(
        np.array(
            [
                [crop_x1, crop_y1],
                [crop_x2, crop_y1],
                [crop_x2, crop_y2],
            ],
            dtype=np.float32,
        )
    )
    dst_points = smr.TensorPoint2Float.from_numpy(
        np.array(
            [
                [0, 0],
                [crop_width, 0],
                [crop_width, crop_height],
            ],
            dtype=np.float32,
        )
    )
    affine_mat = smr.TensorMat.from_numpy(np.zeros((2, 3), dtype=np.float32))

    op1.data_as_operand(src_points, 0)
    op1.data_as_operand(dst_points, 1)
    op1.connect_result_to_data_array(0, affine_mat)
    op1.compute(0)

    img = cv2.imread("tests/data/number_2.png")
    assert img.shape[:2] == (image_height, image_width)
    x = smr.TensorMat.from_numpy(img)
    y = smr.TensorMat((crop_width, crop_height), 3, smr.EDataType.UINT8)

    op2.data_as_operand(affine_mat, 0)
    op2.data_as_operand(x, 1)
    op2.connect_result_to_data_array(0, y)
    op2.compute(0)

    crop = y.numpy()
    assert crop.shape == (crop_height, crop_width, 3)
    # cv2.imwrite("tmp_crop.png", crop)


def test_assignment():
    op = smr.OperatorFactory.create(smr.EOperatorType.ASSIGNMENT)
    assert op is not None
    assert op.get_operator_type() == smr.EOperatorType.ASSIGNMENT
    assert op.get_operand_cnt() == 5
    assert op.get_results_cnt() == 1

    img = cv2.imread("tests/data/dog.jpg")
    x = smr.TensorMat.from_numpy(img)
    y = smr.TensorMat(img.shape[:2], 3, smr.EDataType.FLOAT32)

    op.data_as_operand(x, 0)
    op.connect_result_to_data_array(0, y)
    op.compute(0)
    assert y.numpy().dtype == np.float32


def test_arithmetic():
    configs = ["{0} / 255.0"]
    op = smr.OperatorFactory.create(smr.EOperatorType.ARITHMETIC_COMPOSE, configs)
    assert op is not None
    assert op.get_operator_type() == smr.EOperatorType.ARITHMETIC_COMPOSE
    assert op.get_operand_cnt() == 10
    assert op.get_results_cnt() == 1

    img = cv2.imread("tests/data/dog.jpg").astype(np.float32)
    x = smr.TensorMat.from_numpy(img)
    y = smr.TensorMat(img.shape[:2], 3, smr.EDataType.FLOAT32)

    op.data_as_operand(x, 0)
    op.connect_result_to_data_array(0, y)
    op.compute(0)

    assert y.numpy().max() == 1.0
    assert y.numpy().min() == 0.0


def _as_mat(np_array, dtype):
    # Use from_numpy only for supported dtypes; otherwise create + load raw bytes
    if dtype == smr.EDataType.UINT8:
        return smr.TensorMat.from_numpy(np_array.astype(np.uint8))
    if dtype == smr.EDataType.FLOAT32:
        return smr.TensorMat.from_numpy(np_array.astype(np.float32))

    if np_array.ndim == 2:
        h, w = int(np_array.shape[0]), int(np_array.shape[1])
        ch = 1
    elif np_array.ndim == 3:
        h, w, ch = int(np_array.shape[0]), int(np_array.shape[1]), int(np_array.shape[2])
    else:
        raise ValueError("np_array must be 2D or 3D for TensorMat")
    t = smr.TensorMat([h, w], ch, dtype)
    # Map dtype to numpy type for raw bytes
    np_dt = {
        smr.EDataType.UINT16: np.uint16,
        smr.EDataType.INT16: np.int16,
        smr.EDataType.INT32: np.int32,
        smr.EDataType.FLOAT64: np.float64,
    }[dtype]
    t.load_from_raw_byte_arrays(np_array.astype(np_dt).tobytes())
    return t


def _empty_mat_like(shape_hw, channels, dtype):
    return smr.TensorMat(shape_hw, channels, dtype)

def test_convert_color_bgr2gray():
    img = cv2.imread(os.path.join(DATA_DIR, "dog.jpg"))
    assert img is not None
    op = smr.OperatorFactory.create(smr.EOperatorType.CONVERT_COLOR, [str(cv2.COLOR_BGR2GRAY)])

    x = _as_mat(img, smr.EDataType.UINT8)
    y = _empty_mat_like(img.shape[:2], 1, smr.EDataType.UINT8)

    op.data_as_operand(x, 0)
    op.connect_result_to_data_array(0, y)
    op.compute(0)

    got = y.numpy()
    exp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Bindings may expose single-channel mats as HxWx1; accept both
    if got.ndim == 3 and got.shape[2] == 1:
        got = got[:, :, 0]
    np.testing.assert_array_equal(got, exp)


def test_get_and_apply_affine_image():
    op_get = smr.OperatorFactory.create(smr.EOperatorType.GET_AFFINE)
    op_apply = smr.OperatorFactory.create(smr.EOperatorType.APPLY_AFFINE)

    crop_w, crop_h = 224, 224
    src_pts = np.array([[0, 0], [511, 0], [0, 511]], dtype=np.float32)
    dst_pts = np.array([[0, 0], [crop_w, 0], [0, crop_h]], dtype=np.float32)

    t_src = smr.TensorPoint2Float.from_numpy(src_pts)
    t_dst = smr.TensorPoint2Float.from_numpy(dst_pts)
    t_aff = _as_mat(np.zeros((2, 3), np.float32), smr.EDataType.FLOAT32)

    op_get.data_as_operand(t_src, 0)
    op_get.data_as_operand(t_dst, 1)
    op_get.connect_result_to_data_array(0, t_aff)
    op_get.compute(0)

    img = cv2.imread(os.path.join(DATA_DIR, "number_2.png"))
    assert img is not None and img.shape[:2] == (2464, 3248)

    x = _as_mat(img, smr.EDataType.UINT8)
    y = _empty_mat_like((crop_h, crop_w), 3, smr.EDataType.UINT8)

    op_apply.data_as_operand(t_aff, 0)
    op_apply.data_as_operand(x, 1)
    op_apply.connect_result_to_data_array(0, y)
    op_apply.compute(0)

    assert y.numpy().shape == (crop_h, crop_w, 3)


def test_apply_affine_point_roundtrip():
    # Build affine from points and apply to point set
    op_get = smr.OperatorFactory.create(smr.EOperatorType.GET_AFFINE)
    op_apply_pt = smr.OperatorFactory.create(smr.EOperatorType.APPLY_AFFINE_POINT)

    src = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    dst = np.array([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75]], dtype=np.float32)
    t_src = smr.TensorPoint2Float.from_numpy(src)
    t_dst = smr.TensorPoint2Float.from_numpy(dst)
    t_aff = _as_mat(np.zeros((2, 3), np.float32), smr.EDataType.FLOAT32)

    op_get.data_as_operand(t_src, 0)
    op_get.data_as_operand(t_dst, 1)
    op_get.connect_result_to_data_array(0, t_aff)
    op_get.compute(0)

    t_out = smr.TensorPoint2Float.from_numpy(np.zeros_like(src))
    op_apply_pt.data_as_operand(t_aff, 0)
    op_apply_pt.data_as_operand(t_src, 1)
    op_apply_pt.connect_result_to_data_array(0, t_out)
    op_apply_pt.compute(0)

    np.testing.assert_allclose(t_out.numpy(), dst, rtol=1e-5, atol=1e-5)


def test_assignment_float_conversion():
    img = cv2.imread(os.path.join(DATA_DIR, "dog.jpg"))
    x = _as_mat(img, smr.EDataType.UINT8)
    y = _empty_mat_like(img.shape[:2], 3, smr.EDataType.FLOAT32)

    op = smr.OperatorFactory.create(smr.EOperatorType.ASSIGNMENT)
    op.data_as_operand(x, 0)
    op.connect_result_to_data_array(0, y)
    op.compute(0)

    assert y.numpy().dtype == np.float32


def test_arithmetic_compose_divide_by_255():
    img = cv2.imread(os.path.join(DATA_DIR, "dog.jpg")).astype(np.float32)
    x = _as_mat(img, smr.EDataType.FLOAT32)
    y = _empty_mat_like(img.shape[:2], 3, smr.EDataType.FLOAT32)

    op = smr.OperatorFactory.create(smr.EOperatorType.ARITHMETIC_COMPOSE, ["{0} / 255.0"])
    op.data_as_operand(x, 0)
    op.connect_result_to_data_array(0, y)
    op.compute(0)

    out = y.numpy()
    assert np.isclose(out.max(), 1.0)
    assert np.isclose(out.min(), 0.0)


def test_elementwise_multiply_min_max_or_and_and_compare():
    # Prepare two float32 2-channel mats
    a = np.dstack([
        np.arange(1, 10, dtype=np.float32).reshape(3, 3),
        np.arange(1, 10, dtype=np.float32).reshape(3, 3),
    ])
    b = np.dstack([
        np.full((3, 3), 0.1, dtype=np.float32),
        np.full((3, 3), 0.2, dtype=np.float32),
    ])
    t0 = _as_mat(a, smr.EDataType.FLOAT32)
    t1 = _as_mat(b, smr.EDataType.FLOAT32)
    out = _empty_mat_like((3, 3), 2, smr.EDataType.FLOAT32)

    # Multiply
    op_mul = smr.OperatorFactory.create(smr.EOperatorType.ELEMENTWISE_MULTIPLY)
    op_mul.data_as_operand(t0, 0)
    op_mul.data_as_operand(t1, 1)
    op_mul.connect_result_to_data_array(0, out)
    op_mul.compute(0)
    np.testing.assert_allclose(out.numpy(), a * b, rtol=1e-5, atol=1e-5)

    # Min/Max clamp
    out32 = _empty_mat_like((3, 3), 1, smr.EDataType.FLOAT32)
    upper = _as_mat(np.full((3, 3, 1), 4.5, np.float32), smr.EDataType.FLOAT32)
    lower = _as_mat(np.full((3, 3, 1), 1.5, np.float32), smr.EDataType.FLOAT32)
    inp = _as_mat(np.array([[26.86762, 11.174697, 39.07748], [47.480686, 25.900314, 15.514498],
                            [44.23027, 46.39083, 26.147209]], dtype=np.float32)[:, :, None], smr.EDataType.FLOAT32)

    op_min = smr.OperatorFactory.create(smr.EOperatorType.ELEMENTWISE_MIN)
    op_min.data_as_operand(inp, 0)
    op_min.data_as_operand(upper, 1)
    op_min.connect_result_to_data_array(0, out32)
    op_min.compute(0)

    op_max = smr.OperatorFactory.create(smr.EOperatorType.ELEMENTWISE_MAX)
    op_max.data_as_operand(out32, 0)
    op_max.data_as_operand(lower, 1)
    op_max.connect_result_to_data_array(0, out32)
    op_max.compute(0)
    np.testing.assert_allclose(out32.numpy()[:, :, 0], np.clip(inp.numpy()[:, :, 0], 1.5, 4.5), rtol=1e-5, atol=1e-5)

    # OR/AND (use uint8 mats for numpy access)
    v0 = _as_mat(np.array([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1]], dtype=np.uint8)[:, :, None],
                 smr.EDataType.UINT8)
    v1 = _as_mat(np.array([[1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.uint8)[:, :, None],
                 smr.EDataType.UINT8)
    vout = _empty_mat_like((3, 5), 1, smr.EDataType.UINT8)

    op_or = smr.OperatorFactory.create(smr.EOperatorType.ELEMENTWISE_OR)
    op_or.data_as_operand(v0, 0)
    op_or.data_as_operand(v1, 1)
    op_or.connect_result_to_data_array(0, vout)
    op_or.compute(0)
    np.testing.assert_array_equal(vout.numpy()[:, :, 0], np.bitwise_or(v0.numpy()[:, :, 0], v1.numpy()[:, :, 0]))

    op_and = smr.OperatorFactory.create(smr.EOperatorType.ELEMENTWISE_AND)
    op_and.data_as_operand(v0, 0)
    op_and.data_as_operand(v1, 1)
    op_and.connect_result_to_data_array(0, vout)
    op_and.compute(0)
    np.testing.assert_array_equal(vout.numpy()[:, :, 0], np.bitwise_and(v0.numpy()[:, :, 0], v1.numpy()[:, :, 0]))

    # Customized compare: ">=" on int32 mat
    cmp_a = np.array([[1, 3, 5], [7, 9, 11]], dtype=np.int32)
    cmp_b = np.array([[12, 10, 8], [6, 4, 2]], dtype=np.int32)
    a_i32 = _as_mat(cmp_a[:, :, None], smr.EDataType.INT32)
    b_i32 = _as_mat(cmp_b[:, :, None], smr.EDataType.INT32)
    out_i32 = _empty_mat_like((2, 3), 1, smr.EDataType.UINT8)
    op_cmp = smr.OperatorFactory.create(smr.EOperatorType.CUSTOMIZED_COMPARE, [">="])
    op_cmp.data_as_operand(a_i32, 0)
    op_cmp.data_as_operand(b_i32, 1)
    op_cmp.connect_result_to_data_array(0, out_i32)
    op_cmp.compute(0)
    np.testing.assert_array_equal(out_i32.numpy()[:, :, 0].astype(bool), (cmp_a >= cmp_b))


def test_nms_small_example():
    # Threshold 0.85
    op = smr.OperatorFactory.create(smr.EOperatorType.NMS, ["0.85"])

    scores = np.array([[0.7527], [0.8331], [0.9862], [0.2276], [0.5752]], dtype=np.float64)
    boxes = np.array([
        [0.0, 0.0, 0.8, 0.8],
        [0.0, 0.0, 0.81, 0.81],
        [0.01, 0.01, 0.81, 0.81],
        [0.5, 0.5, 1.0, 1.0],
        [0.5, 0.51, 1.0, 1.0],
    ], dtype=np.float64)

    t_score = _as_mat(scores[:, :, None], smr.EDataType.FLOAT64)
    t_box = _as_mat(boxes[:, :, None], smr.EDataType.FLOAT64)

    out_box = _empty_mat_like((2, 4), 1, smr.EDataType.FLOAT32)
    out_idx = _empty_mat_like((2, 1), 1, smr.EDataType.UINT8)

    op.data_as_operand(t_score, 0)
    op.data_as_operand(t_box, 1)
    op.connect_result_to_data_array(1, out_box)
    op.connect_result_to_data_array(2, out_idx)
    op.compute(0)

    idx = out_idx.numpy().reshape(-1)[0:2].astype(np.int32)
    # Backend may return indices {1,2} or {2,4} depending on IOU tie-breaks
    assert set(idx.tolist()) in ({2, 4}, {1, 2})


def test_normalize_l1_and_minmax():
    # L1 on float32 mat
    op_l1 = smr.OperatorFactory.create(smr.EOperatorType.NORMALIZE, ["L1"])  # config optional
    mat = _as_mat(np.array([[0.75683555, 82.09314683, 26.06050979],
                            [3.71944931, 2.26043793, 33.85353257],
                            [4.52001966, 16.12689719, 32.90809434],
                            [62.5474935, 59.96486076, 22.26372344],
                            [54.1383077, 31.4957161, 84.84667126]], dtype=np.float32)[:, :, None], smr.EDataType.FLOAT32)
    op_l1.data_as_operand(mat, 0)
    op_l1.connect_result_to_data_array(0, mat)
    op_l1.compute(0)
    # Sum across entire tensor should be 1 within tolerance
    s = mat.numpy().sum()
    assert abs(s - 1.0) < 1e-5

    # MINMAX on vec2 field represented as mat C2
    op_mm = smr.OperatorFactory.create(smr.EOperatorType.NORMALIZE, ["MINMAX"])  # with alpha,beta in operand1
    vec2 = _as_mat(np.array([[24.58577598, 34.33790628], [26.03850459, 3.91164427], [23.37048487, 17.00936702],
                             [3.35316138, 26.34229045], [19.50105471, 31.66600924], [16.86938474, 32.9395935],
                             [15.64208762, 22.4370421], [6.71140535, 11.17409589], [33.89264754, 17.90131646]],
                            dtype=np.float64).reshape(9, 1, 2), smr.EDataType.FLOAT64)
    ab = _as_mat(np.array([[ -2.0, 2.0 ]], dtype=np.float64).reshape(1, 1, 2), smr.EDataType.FLOAT64)
    out = _empty_mat_like((9, 1), 2, smr.EDataType.FLOAT64)
    op_mm.data_as_operand(vec2, 0)
    op_mm.data_as_operand(ab, 1)
    op_mm.connect_result_to_data_array(0, out)
    op_mm.compute(0)
    # All outputs should fall within [-2, 2]
    assert out.numpy().min() >= -2.0 and out.numpy().max() <= 2.0


def test_make_transform_mat_and_inversion():
    # rvec, tvec (float64 3x1) and scale (float32 3x1) -> 4x4 float32
    op = smr.OperatorFactory.create(smr.EOperatorType.MAKE_TRANSFORM_MAT)

    rvec = _as_mat(np.array([[-1.27867382], [-0.27638333], [2.0888375]], dtype=np.float64), smr.EDataType.FLOAT64)
    tvec = _as_mat(np.array([[0.0], [0.3], [-10.0]], dtype=np.float64), smr.EDataType.FLOAT64)
    scale = _as_mat(np.array([[0.3], [0.3], [0.3]], dtype=np.float32), smr.EDataType.FLOAT32)

    out = _empty_mat_like((4, 4), 1, smr.EDataType.FLOAT32)
    op.data_as_operand(rvec, 0)
    op.data_as_operand(tvec, 1)
    op.data_as_operand(scale, 2)
    op.connect_result_to_data_array(0, out)
    op.compute(0)

    m = out.numpy().reshape(4, 4)
    # Invert using INVERSION operator and verify M * M_inv ~= I
    inv = _empty_mat_like((4, 4), 1, smr.EDataType.FLOAT32)
    op_inv = smr.OperatorFactory.create(smr.EOperatorType.INVERSION)
    op_inv.data_as_operand(out, 0)
    op_inv.connect_result_to_data_array(0, inv)
    op_inv.compute(0)

    ident = m @ inv.numpy().reshape(4, 4)
    np.testing.assert_allclose(ident, np.eye(4, dtype=np.float32), rtol=1e-4, atol=1e-4)


def test_solve_pnp_shapes_and_runs():
    # Minimal SolvePnP sanity: no ground truth check, just execution and output shapes
    op = smr.OperatorFactory.create(smr.EOperatorType.SOLVE_P_N_P)

    obj = smr.TensorPoint3Float.from_numpy(
        np.array([
            [0.22960754, -1.38081639, -11.27363128],
            [1.91207856, -0.07446423, -9.16111256],
            [-0.22960754, 1.98081639, -8.72636872],
            [-1.91207856, 0.67446423, -10.83888744],
            [-0.45596303, -3.66931529, -10.73255295],
            [2.34815532, -1.49206169, -7.21168842],
            [-1.22132152, 1.93340601, -6.48711535],
            [-4.02543987, -0.24384759, -10.00797988],
        ], dtype=np.float32)
    )
    img = smr.TensorPoint2Float.from_numpy(
        np.array([
            [0.49321107, 0.54082732], [0.43042772, 0.50270943], [0.50877064, 0.42433598], [0.55880304, 0.47925789],
            [0.51416137, 0.61396218], [0.3914653, 0.56896497], [0.56275627, 0.40065404], [0.63407434, 0.50812177],
        ], dtype=np.float32)
    )
    cam = _as_mat(np.array([[0.33333333, 0.0, 0.5], [0.0, 0.33333333, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32),
                  smr.EDataType.FLOAT32)

    rvec = _empty_mat_like((3, 1), 1, smr.EDataType.FLOAT64)
    tvec = _empty_mat_like((3, 1), 1, smr.EDataType.FLOAT64)

    op.data_as_operand(obj, 0)
    op.data_as_operand(img, 1)
    op.data_as_operand(cam, 2)
    op.connect_result_to_data_array(0, rvec)
    op.connect_result_to_data_array(1, tvec)
    op.compute(0)

    assert rvec.numpy().shape == (3, 1, 1)
    assert tvec.numpy().shape == (3, 1, 1)


def test_sort_mat_by_column():
    mat = np.array([[8.7, 6.5], [9.8, 1.2], [5.5, 8.6], [6.6, 9.7], [1.7, 5.4]], dtype=np.float32)
    t_mat = _as_mat(mat, smr.EDataType.FLOAT32)
    idx = _empty_mat_like((5, 2), 1, smr.EDataType.INT16)

    op = smr.OperatorFactory.create(smr.EOperatorType.SORT_MAT, ["COLUMN"])
    op.data_as_operand(t_mat, 0)
    op.connect_result_to_data_array(0, t_mat)
    op.connect_result_to_data_array(1, idx)
    op.compute(0)

    # Verify sorted order per column descending
    sorted_mat = t_mat.numpy()[:, :, 0]
    assert (sorted_mat[:, 0] == np.sort(mat[:, 0])[::-1]).all()
    assert (sorted_mat[:, 1] == np.sort(mat[:, 1])[::-1]).all()


def test_normalize_div255():
    image = (np.random.randn(480, 320, 3) * 255).astype(np.uint8)
    h0, w0 = image.shape[:2]
    target_h, target_w = h0, w0

    # Build serializable pipeline
    p = SerializablePipeline()

    # Create and query tensors
    lt_img = p.query_local_tensor(p.allocate_placeholder([h0, w0], mat_flag(smr.EDataType.UINT8, 3), "image"))
    lt_y2 = p.query_local_tensor(p.allocate_local_tensor([target_w, target_h], mat_flag(smr.EDataType.UINT8, 3), "letterboxed_rgb"))
    lt_y3 = p.query_local_tensor(p.allocate_local_tensor([target_w, target_h], mat_flag(smr.EDataType.FLOAT32, 3), "letterboxed_float32"))
    lt_y4 = p.query_local_tensor(p.allocate_local_tensor([target_w, target_h], mat_flag(smr.EDataType.FLOAT32, 3), "normalized_tensor"))
    lt_y5 = p.query_local_tensor(p.allocate_placeholder([target_w, target_h], mat_flag(smr.EDataType.FLOAT32, 3), "model_input_tensor"))

    # Operators
    op_cvt_rgb = p.query_operator(p.allocate_operator(smr.EOperatorType.CONVERT_COLOR, [str(cv2.COLOR_BGR2RGB)]))
    op_assign_float = p.query_operator(p.allocate_operator(smr.EOperatorType.ASSIGNMENT))
    op_div255 = p.query_operator(p.allocate_operator(smr.EOperatorType.ARITHMETIC_COMPOSE, ["{0} / 255.0"]))
    op_assign_out = p.query_operator(p.allocate_operator(smr.EOperatorType.ASSIGNMENT))

    # Connect ops
    op_cvt_rgb.data_as_operand(lt_img, 0)
    op_cvt_rgb.connect_result_to_data_array(0, lt_y2)

    op_assign_float.data_as_operand(lt_y2, 0)
    op_assign_float.connect_result_to_data_array(0, lt_y3)

    op_div255.data_as_operand(lt_y3, 0)
    op_div255.connect_result_to_data_array(0, lt_y4)

    op_assign_out.data_as_operand(lt_y4, 0)
    op_assign_out.connect_result_to_data_array(0, lt_y5)

    # IO
    p.set_inputs(["image"])
    p.set_outputs(["model_input_tensor"])  # main output

    # Execute the serialized pipeline via DeserializedPipeline for parity.
    model_input_tensor = DeserializedPipeline(p.spec)(image, timeout=3)
