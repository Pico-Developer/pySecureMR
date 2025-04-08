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

import securemr as smr

import cv2
import numpy as np


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
    assert gray.ndim == 2
    np.testing.assert_array_equal(gray, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    # cv2.imwrite("tmp_gray.png", gray)


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
