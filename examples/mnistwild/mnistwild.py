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
import pathlib
import cv2
import numpy as np
import securemr as smr

ROOT = pathlib.Path(__file__).parent.resolve()
PIPE_JSON = str(ROOT / "mnistwild_preprocess.json")


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
    t_y1 = smr.TensorMat.from_numpy(np.zeros((crop_height, crop_width, 1), dtype=np.uint8))
    t_y2 = smr.TensorMat.from_numpy(np.zeros((crop_height, crop_width, 1), dtype=np.uint8))
    t_y3 = smr.TensorMat.from_numpy(np.zeros((crop_height, crop_width, 1), dtype=np.float32))
    t_y4 = smr.TensorMat.from_numpy(np.zeros((crop_height, crop_width, 1), dtype=np.float32))

    # Build pipeline
    p = smr.Pipeline()

    # Flags matching TensorMat/TensorPoint placeholders (omit channel bits)
    def _mat_flag(dtype: smr.EDataType, channels: int) -> int:
        return int(dtype) | smr.BaseType.MAT | (smr.BaseType.CHANNEL_MASK & channels)

    def _pt2_flag(dtype: smr.EDataType = smr.EDataType.FLOAT32) -> int:
        return int(dtype) | smr.BaseType.POINT_2

    # Placeholders: image, points, affine, and each stage output
    ph_img = p.allocate_placeholder(list(img.shape[:2]), _mat_flag(smr.EDataType.UINT8, 3))
    ph_src = p.allocate_placeholder([3], _pt2_flag())
    ph_dst = p.allocate_placeholder([3], _pt2_flag())
    ph_aff = p.allocate_placeholder([2, 3], _mat_flag(smr.EDataType.FLOAT32, 1))
    ph_y1 = p.allocate_placeholder([crop_height, crop_width], _mat_flag(smr.EDataType.UINT8, 1))
    ph_y2 = p.allocate_placeholder([crop_height, crop_width], _mat_flag(smr.EDataType.UINT8, 1))
    ph_y3 = p.allocate_placeholder([crop_height, crop_width], _mat_flag(smr.EDataType.FLOAT32, 1))
    ph_y4 = p.allocate_placeholder([crop_height, crop_width], _mat_flag(smr.EDataType.FLOAT32, 1))

    # Query local tensors for wiring operators
    lt_img = p.query_local_tensor(ph_img)
    lt_src = p.query_local_tensor(ph_src)
    lt_dst = p.query_local_tensor(ph_dst)
    lt_aff = p.query_local_tensor(ph_aff)
    lt_y1 = p.query_local_tensor(ph_y1)
    lt_y2 = p.query_local_tensor(ph_y2)
    lt_y3 = p.query_local_tensor(ph_y3)
    lt_y4 = p.query_local_tensor(ph_y4)

    # Operators
    op_get_aff = p.allocate_operator(smr.EOperatorType.GET_AFFINE)
    op_apply_aff = p.allocate_operator(smr.EOperatorType.APPLY_AFFINE)
    op_cvt_gray = p.allocate_operator(smr.EOperatorType.CONVERT_COLOR, [str(cv2.COLOR_BGR2GRAY)])
    op_assign = p.allocate_operator(smr.EOperatorType.ASSIGNMENT)
    op_div255 = p.allocate_operator(smr.EOperatorType.ARITHMETIC_COMPOSE, ["{0} / 255.0"])

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

    # ARITHMETIC_COMPOSE(y3/255.0)->y4
    op_div255 = p.query_operator(op_div255)
    op_div255.data_as_operand(lt_y3, 0)
    op_div255.connect_result_to_data_array(0, lt_y4)

    # Map placeholders to global tensors and run
    # AI: print key and value tensor data_type
    ph_map = {
        int(ph_img): t_img,
        int(ph_src): src_points,
        int(ph_dst): dst_points,
        int(ph_aff): t_affine,
        int(ph_y1): t_y1,
        int(ph_y2): t_y2,
        int(ph_y3): t_y3,
        int(ph_y4): t_y4,
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
        # Non-fatal: printing is best-effort for visibility
        pass

    p.serialize_to_json(PIPE_JSON)
    _restore = smr.Pipeline.deserialize_from_json(PIPE_JSON)

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
    return t_y4


def preprocess_pipeline_v2(image_path: str):
    """Preprocess an image using the serialized pipeline configuration.

    The pipeline graph is loaded from ``mnistwild_preprocess.json`` and executed
    with freshly allocated tensors that mirror the placeholders recorded in the
    JSON file.

    Args:
        image_path: Path to the input image.

    Returns:
        Preprocessed image tensor (float32 HxWx1, normalized to [0, 1]).
    """
    pipeline_path = ROOT / "mnistwild_preprocess.json"
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline JSON not found: {pipeline_path}")

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

    try:
        pipeline = smr.Pipeline.deserialize_from_json(str(pipeline_path))
    except RuntimeError as exc:  # pragma: no cover - depends on native runtime
        raise RuntimeError(
            f"Failed to deserialize pipeline: {exc}. Regenerate the JSON by running "
            "`preprocess_pipeline` once and retry."
        ) from exc

    with open(pipeline_path, "r", encoding="utf-8") as f:
        pipeline_spec = json.load(f)

    tensor_ids = pipeline.get_tensor_id_sequence()
    tensors_spec = pipeline_spec.get("tensors", [])

    if len(tensor_ids) != len(tensors_spec):
        raise RuntimeError(
            "Serialized tensor count mismatch between JSON and pipeline instance"
        )

    placeholders = {}
    for entry in tensors_spec:
        if not entry.get("is_placeholder"):
            continue
        idx = entry["index"]
        if idx < 0 or idx >= len(tensor_ids):
            raise RuntimeError(f"Placeholder index {idx} out of range")
        placeholders[idx] = int(tensor_ids[idx])

    if not placeholders:
        raise ValueError("No placeholders found in pipeline JSON")

    def _non_null(values):
        return [v for v in values if v is not None]

    op_type = {int(smr.EOperatorType.GET_AFFINE): None}
    op_type[int(smr.EOperatorType.APPLY_AFFINE)] = None
    op_type[int(smr.EOperatorType.CONVERT_COLOR)] = None
    op_type[int(smr.EOperatorType.ASSIGNMENT)] = None
    op_type[int(smr.EOperatorType.ARITHMETIC_COMPOSE)] = None

    for op in pipeline_spec.get("operators", []):
        if op["type"] in op_type and op_type[op["type"]] is None:
            op_type[op["type"]] = op

    get_affine_op = op_type[int(smr.EOperatorType.GET_AFFINE)]
    apply_affine_op = op_type[int(smr.EOperatorType.APPLY_AFFINE)]
    convert_color_op = op_type[int(smr.EOperatorType.CONVERT_COLOR)]
    assignment_op = op_type[int(smr.EOperatorType.ASSIGNMENT)]
    arithmetic_op = op_type[int(smr.EOperatorType.ARITHMETIC_COMPOSE)]

    if None in (get_affine_op, apply_affine_op, convert_color_op, assignment_op, arithmetic_op):
        missing = [k for k, v in op_type.items() if v is None]
        raise ValueError(f"Missing operators in pipeline JSON for types: {missing}")

    src_idx, dst_idx = _non_null(get_affine_op["operands"])
    (affine_idx,) = _non_null(get_affine_op["results"])

    aff_operands = _non_null(apply_affine_op["operands"])
    if len(aff_operands) < 2:
        raise ValueError("APPLY_AFFINE operator missing operands in JSON")
    affine_operand_idx, image_idx = aff_operands[:2]
    (y1_idx,) = _non_null(apply_affine_op["results"])

    if affine_operand_idx != affine_idx:
        raise ValueError(
            "APPLY_AFFINE operand 0 does not match affine output placeholder"
        )

    (y2_idx,) = _non_null(convert_color_op["results"])
    (y3_idx,) = _non_null(assignment_op["results"])
    (y4_idx,) = _non_null(arithmetic_op["results"])

    ph_src = placeholders[src_idx]
    ph_dst = placeholders[dst_idx]
    ph_affine = placeholders[affine_idx]
    ph_image = placeholders[image_idx]
    ph_y1 = placeholders[y1_idx]
    ph_y2 = placeholders[y2_idx]
    ph_y3 = placeholders[y3_idx]
    ph_y4 = placeholders[y4_idx]

    # Prepare tensors backing the placeholders
    t_img = smr.TensorMat.from_numpy(img)

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

    t_y1 = smr.TensorMat.from_numpy(np.zeros((crop_height, crop_width, 1), dtype=np.uint8))
    t_y2 = smr.TensorMat.from_numpy(np.zeros((crop_height, crop_width, 1), dtype=np.uint8))
    t_y3 = smr.TensorMat.from_numpy(np.zeros((crop_height, crop_width, 1), dtype=np.float32))
    t_y4 = smr.TensorMat.from_numpy(np.zeros((crop_height, crop_width, 1), dtype=np.float32))

    ph_map = {
        ph_image: t_img,
        ph_src: src_points,
        ph_dst: dst_points,
        ph_affine: t_affine,
        ph_y1: t_y1,
        ph_y2: t_y2,
        ph_y3: t_y3,
        ph_y4: t_y4,
    }

    task = smr.Task(pipeline, ph_map, 0, None)
    task.verify_all_place_holder_contained()
    task.setup_place_holder_mapping()

    pool = smr.ThreadPool2()
    pool.enqueue(task)

    import time as _time

    for _ in range(200):  # up to ~2 seconds
        if not pipeline.cannot_modified():
            break
        _time.sleep(0.01)

    return t_y4

def main():
    """Run the MNIST wild example.

    This function demonstrates how to use the MNIST model for inference on custom images.
    """
    test_image = ROOT / "number_5.png"

    x = preprocess_pipeline_v2(str(test_image)).numpy()[:,:,0]
    # x = preprocess_pipeline(str(test_image)).numpy()[:,:,0]
    # x0 = preprocess(str(test_image)).numpy()[:,:,0]
    # assert np.allclose(x, x0, rtol=1e-4, atol=1e-4)

    context_binary_file = ROOT / "mnist.serialized.bin"
    model = smr.QnnModel(context_binary_file, "host", name="mnistwild_test")
    # # You can also run QnnModel on android device, but ROOT is required
    # model = smr.QnnModel(context_binary_file, "android", name="mnistwild_test")

    x = x[None, :, :, None]  # HxW to NHWC
    score, idx = model(x, is_nhwc=True)
    print("number: ", int(idx.squeeze()))
    print("score: ", score.squeeze())


if __name__ == "__main__":
    main()
