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

"""Example implementation of MNIST classification in the wild."""

import os
import pathlib

import securemr as smr

import cv2
import numpy as np


def preprocess(image_path):
    """Preprocess an image for MNIST model inference.

    Args:
        image_path: Path to the input image.

    Returns:
        Preprocessed image tensor.
    """
    img = cv2.imread(image_path)
    x = smr.TensorMat.from_numpy(img)

    op1 = smr.OperatorFactory.create(smr.EOperatorType.GET_AFFINE)
    op2 = smr.OperatorFactory.create(smr.EOperatorType.APPLY_AFFINE)
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

    # crop image
    assert img.shape[:2] == (image_height, image_width)
    y1 = smr.TensorMat((crop_width, crop_height), 1, smr.EDataType.UINT8)
    op2.data_as_operand(affine_mat, 0)
    op2.data_as_operand(x, 1)
    op2.connect_result_to_data_array(0, y1)
    op2.compute(0)

    # to gray
    ConvertColorOp = smr.OperatorFactory.create(smr.EOperatorType.CONVERT_COLOR, [str(cv2.COLOR_BGR2GRAY)])
    y2 = smr.TensorMat((crop_width, crop_height), 1, smr.EDataType.UINT8)
    ConvertColorOp.data_as_operand(y1, 0)
    ConvertColorOp.connect_result_to_data_array(0, y2)
    ConvertColorOp.compute(0)

    # uint8 to float32
    y3 = smr.TensorMat((crop_width, crop_height), 1, smr.EDataType.FLOAT32)
    op3 = smr.OperatorFactory.create(smr.EOperatorType.ASSIGNMENT)
    op3.data_as_operand(y2, 0)
    op3.connect_result_to_data_array(0, y3)
    op3.compute(0)

    # 255.0 -> 1.0
    op4 = smr.OperatorFactory.create(smr.EOperatorType.ARITHMETIC_COMPOSE, ["{0} / 255.0"])
    y4 = smr.TensorMat((crop_width, crop_height), 1, smr.EDataType.FLOAT32)
    op4.data_as_operand(y3, 0)
    op4.connect_result_to_data_array(0, y4)
    op4.compute(0)

    return y4


def main():
    """Run the MNIST wild example.

    This function demonstrates how to use the MNIST model for inference on custom images.
    """
    root = pathlib.Path(__file__).parent.resolve()
    test_image = root / "number_5.png"
    x = preprocess(str(test_image)).numpy()

    context_binary_file = root / "mnist.serialized.bin"
    model = smr.QnnModel(context_binary_file, "host", name="mnistwild_test")
    # # You can also run QnnModel on android device, but root is required
    # model = smr.QnnModel(context_binary_file, "android", name="mnistwild_test")

    x = x[None, :, :, None]  # HxW to NHWC
    score, idx = model(x, is_nhwc=True)
    print("number: ", int(idx.squeeze()))
    print("score: ", score.squeeze())


if __name__ == "__main__":
    main()
