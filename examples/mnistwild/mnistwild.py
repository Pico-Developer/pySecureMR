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

import json
import os
import pathlib
import cv2
import numpy as np
import securemr as smr
from securemr.serialization import (
    Pipeline,
    DeserializedPipeline,
    add_vst_operator,
    add_model_inference_operator,
    mat_flag,
)

ROOT = pathlib.Path(__file__).parent.resolve()
PIPE_JSON = str(ROOT / "mnist_pipeline.json")


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
    convert_color_op = smr.OperatorFactory.create(smr.EOperatorType.CONVERT_COLOR, [str(cv2.COLOR_BGR2GRAY)])
    y2 = smr.TensorMat((crop_width, crop_height), 1, smr.EDataType.UINT8)
    convert_color_op.data_as_operand(y1, 0)
    convert_color_op.connect_result_to_data_array(0, y2)
    convert_color_op.compute(0)

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


def preprocess_pipeline(image_path: str):
    """Preprocess an image for MNIST model inference using a Pipeline.

    This mirrors the logic in `preprocess` but constructs a graph with
    smr.Pipeline, allocates placeholders, and executes it via a Task.

    Args:
        image_path: Path to the input image.

    Returns:
        Preprocessed image tensor (float32 HxWx1, normalized to [0,1]).
    """
    # Load input image and prepare global tensors/constants
    img = cv2.imread(image_path)

    image_width = 3248
    image_height = 2464
    crop_x1 = 1444
    crop_y1 = 1332
    crop_x2 = 2045
    crop_y2 = 1933
    crop_width = 224
    crop_height = 224

    assert img is not None, f"Failed to read image: {image_path}"
    assert img.shape[:2] == (image_height, image_width), (
        f"Unexpected input size {img.shape[:2]}, expected {(image_height, image_width)}"
    )

    # Global tensors backing placeholders
    t_img = smr.TensorMat.from_numpy(img)  # uint8 BGR

    # Create source/destination points using TensorPoint2Float for compatibility
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
    t_affine = smr.TensorMat.from_numpy(np.zeros((2, 3), dtype=np.float32))

    # Use from_numpy to ensure placeholder flags with dtype+channels match
    t_y4 = smr.TensorMat.from_numpy(np.zeros((crop_height, crop_width, 1), dtype=np.float32))
    t_y5 = smr.TensorMat.from_numpy(np.zeros((crop_height, crop_width, 3), dtype=np.uint8))

    # Build pipeline
    p = Pipeline()

    # Flags for POINT_2 tensors: keep POINT_2 bit; backend infers float32
    def _pt2_flag(is_placeholder=False) -> int:
        if is_placeholder:
            return int(smr.BaseType.POINT_2)
        else:
            return int(smr.EDataType.FLOAT32) | int(smr.BaseType.POINT_2)

    # Only inputs/outputs are placeholders; intermediates are local tensors
    ph_img = p.allocate_placeholder(list(img.shape[:2]), mat_flag(smr.EDataType.UINT8, 3), "image")
    # src_points/dst_points should be local tensors (constants)
    ph_src = p.allocate_local_tensor([3], _pt2_flag(), "src_points", value=src_points.numpy())
    ph_dst = p.allocate_local_tensor([3], _pt2_flag(), "dst_points", value=dst_points.numpy())
    # local tensors for intermediate results
    ph_aff = p.allocate_local_tensor([2, 3], mat_flag(smr.EDataType.FLOAT32, 1), "affine")
    ph_y1 = p.allocate_local_tensor([crop_height, crop_width], mat_flag(smr.EDataType.UINT8, 3), "crop_rgb_tensor")
    ph_y2 = p.allocate_local_tensor([crop_height, crop_width], mat_flag(smr.EDataType.UINT8, 1), "crop_gray_tensor")
    ph_y3 = p.allocate_local_tensor([crop_height, crop_width], mat_flag(smr.EDataType.FLOAT32, 1), "crop_float_tensor")
    # output remains a placeholder, so caller can fetch result
    ph_y4 = p.allocate_placeholder([crop_height, crop_width], mat_flag(smr.EDataType.FLOAT32, 1), "normalized_input_tensor")
    ph_y5 = p.allocate_placeholder([crop_height, crop_width], mat_flag(smr.EDataType.UINT8, 3), "cropped_image")

    # Query local tensors for wiring operators
    lt_img = p.query_local_tensor(ph_img)
    lt_src = p.query_local_tensor(ph_src)
    lt_src.load_from_raw_byte_arrays(np.ascontiguousarray(src_points.numpy()).tobytes())
    lt_dst = p.query_local_tensor(ph_dst)
    lt_dst.load_from_raw_byte_arrays(np.ascontiguousarray(dst_points.numpy()).tobytes())
    lt_aff = p.query_local_tensor(ph_aff)
    lt_y1 = p.query_local_tensor(ph_y1)
    lt_y2 = p.query_local_tensor(ph_y2)
    lt_y3 = p.query_local_tensor(ph_y3)
    lt_y4 = p.query_local_tensor(ph_y4)
    lt_y5 = p.query_local_tensor(ph_y5)

    # Operators
    op_get_aff = p.allocate_operator(smr.EOperatorType.GET_AFFINE, )
    op_apply_aff = p.allocate_operator(smr.EOperatorType.APPLY_AFFINE)
    op_cvt_gray = p.allocate_operator(smr.EOperatorType.CONVERT_COLOR, [str(cv2.COLOR_BGR2GRAY)])
    op_assign = p.allocate_operator(smr.EOperatorType.ASSIGNMENT)
    op_div255 = p.allocate_operator(smr.EOperatorType.ARITHMETIC_COMPOSE, ["{0} / 255.0"])
    op_assign_2 = p.allocate_operator(smr.EOperatorType.ASSIGNMENT)

    # Connect: GET_AFFINE(src,dst)->affine
    op_get_aff = p.query_operator(op_get_aff)
    op_get_aff.data_as_operand(lt_src, 0)
    op_get_aff.data_as_operand(lt_dst, 1)
    op_get_aff.connect_result_to_data_array(0, lt_aff)

    # APPLY_AFFINE(affine, image)->y1
    op_apply_aff = p.query_operator(op_apply_aff)
    op_apply_aff.data_as_operand(lt_aff, 0)
    op_apply_aff.data_as_operand(lt_img, 1)
    op_apply_aff.connect_result_to_data_array(0, lt_y1)

    # CONVERT_COLOR(y1 BGR->GRAY)->y2
    op_cvt_gray = p.query_operator(op_cvt_gray)
    op_cvt_gray.data_as_operand(lt_y1, 0)
    op_cvt_gray.connect_result_to_data_array(0, lt_y2)

    # ASSIGNMENT(uint8->float32)->y3
    op_assign = p.query_operator(op_assign)
    op_assign.data_as_operand(lt_y2, 0)
    op_assign.connect_result_to_data_array(0, lt_y3)
    
    # ASSIGNMENT(uint8->uint8, local_tensor to global tensor)
    op_assign_2 = p.query_operator(op_assign_2)
    op_assign_2.data_as_operand(lt_img, 0)
    op_assign_2.connect_result_to_data_array(0, lt_y5)

    # ARITHMETIC_COMPOSE(y3/255.0)->y4
    op_div255 = p.query_operator(op_div255)
    op_div255.data_as_operand(lt_y3, 0)
    op_div255.connect_result_to_data_array(0, lt_y4)

    # Map placeholders to global tensors and run
    # Declare pipeline IO for serialization/deserialization convenience (use names)
    p.set_inputs(["image"])  # only true inputs are placeholders
    p.set_outputs(["normalized_input_tensor", "cropped_image"])    # output as placeholder for retrieval

    # Map only placeholder tensors (inputs/outputs)
    ph_map = {
        int(ph_img): t_img,
        int(ph_y4): t_y4,
        int(ph_y5): t_y5,
    }

    # Print placeholder (key) and tensor (value) data types for debugging
    try:
        for k, v in ph_map.items():
            try:
                k_dtype = p.query_local_tensor(k).get_datatype()
            except Exception:
                k_dtype = "<unknown>"
            try:
                v_dtype = v.get_datatype()
            except Exception:
                v_dtype = "<unknown>"
            print(f"Placeholder {k} dtype={k_dtype} -> Value dtype={v_dtype}")
    except Exception:
        pass
    
    task = smr.Task(p, ph_map, 0, None)
    task.verify_all_place_holder_contained()
    task.setup_place_holder_mapping()

    pool = smr.ThreadPool2()
    pool.enqueue(task)

    # Wait for pipeline execution to complete
    import time as _time
    for _ in range(200):  # up to ~2 seconds
        if not p.cannot_modified():
            break
        _time.sleep(0.01)
    
    return t_y4, p


def main():
    """Run the MNIST wild example.

    This function demonstrates how to use the MNIST model for inference on custom images.
    """
    test_image = ROOT / "number_5.png"

    x, pipeline = preprocess_pipeline(str(test_image))
    x = x.numpy()[:,:,0]
    
    x0 = preprocess(str(test_image)).numpy()[:,:,0]
    assert np.allclose(x, x0, rtol=1e-4, atol=1e-4)

    pipeline.save("/tmp/tmp_pipeline.json")
    restored_pipeline = DeserializedPipeline("/tmp/tmp_pipeline.json")
    x2, _ = restored_pipeline(cv2.imread(str(test_image)))
    x2 = x2.numpy()[:,:,0]
    assert np.allclose(x, x2, rtol=1e-4, atol=1e-4)

    context_binary_file = ROOT / "mnist.serialized.bin"
    model = smr.QnnModel(context_binary_file, "host", name="mnistwild_test")
    # # You can also run QnnModel on android device, but ROOT is required
    # model = smr.QnnModel(context_binary_file, "android", name="mnistwild_test")

    x = x[None, :, :, None]  # HxW to NHWC
    score, idx = model(x, is_nhwc=True)
    print("number: ", int(idx.squeeze()))
    print("score: ", score.squeeze())

    # Add vst and model operator into pipeline
    add_vst_operator(pipeline, ("image", "left_rgb"))
    add_model_inference_operator(
            pipeline,
            context_binary_file=os.path.basename(context_binary_file),
            model_name="mnist",
            model_input=[{"name": "input_1", "tensor": "normalized_input_tensor"}],
            model_output=[
                {"name": "_538", "tensor": "predicted_score"},
                {"name": "_539", "tensor": "predicted_class"}],
            model_output_tensor_info=[
                {"dimensions": [1], "channels": 1, "data_type": np.float32},
                {"dimensions": [1], "channels": 1, "data_type": np.int32}]
            )
    pipeline.save(PIPE2_JSON)


if __name__ == "__main__":
    main()
