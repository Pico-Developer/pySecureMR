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
from securemr.pipeline_zoo import create_litert_model_spec
from securemr.core.utils import convert_from_dtype, mat_flag, type_to_name
from securemr.py2smr import trace, ops, convert
from securemr.py2smr.tracer import get_current_trace
from securemr.py2smr.verifier import run_pipeline_python

ROOT = pathlib.Path(__file__).parent.resolve()
PIPE_JSON = str(ROOT / "mnist_pipeline.json")
MODEL_ASSET = "model/mnist.tflite"

IMAGE_WIDTH = 3248
IMAGE_HEIGHT = 2464
CROP_X1 = 1444
CROP_Y1 = 1332
CROP_X2 = 2045
CROP_Y2 = 1933
CROP_WIDTH = 224
CROP_HEIGHT = 224


def _register_constant(name: str, value: np.ndarray) -> np.ndarray:
    ctx = get_current_trace()
    if ctx is not None:
        ctx.register_tensor(value, name=name)
    return value


def _inject_constant_values(ctx, constants):
    for name, value in constants.items():
        info = ctx.tensors.get(name)
        if info is None:
            ctx.register_tensor(value, name=name)
            info = ctx.tensors[name]
        info.value = value.copy()


@trace(inputs=["image"], outputs=["normalized_input_tensor", "cropped_image"])
def preprocess_py2smr(image: np.ndarray):
    """Preprocess an image for MNIST model inference (py2smr traceable)."""
    src_points = _register_constant(
        "src_points",
        np.array(
            [
                [CROP_X1, CROP_Y1],
                [CROP_X2, CROP_Y1],
                [CROP_X2, CROP_Y2],
            ],
            dtype=np.float32,
        ),
    )
    dst_points = _register_constant(
        "dst_points",
        np.array(
            [
                [0, 0],
                [CROP_WIDTH, 0],
                [CROP_WIDTH, CROP_HEIGHT],
            ],
            dtype=np.float32,
        ),
    )

    affine = ops.get_affine(src_points, dst_points, output_name="affine")
    cropped = ops.apply_affine(
        affine,
        image,
        output_shape=(CROP_HEIGHT, CROP_WIDTH),
        output_name="cropped_image",
    )
    gray = ops.convert_color(cropped, cv2.COLOR_BGR2GRAY, output_name="crop_gray_tensor")
    float_template = np.zeros((CROP_HEIGHT, CROP_WIDTH), dtype=np.float32)
    gray_f32 = ops.assignment(gray, float_template, output_name="crop_float_tensor")
    normalized = ops.arithmetic(gray_f32, "{0} / 255.0", output_name="normalized_input_tensor")
    return normalized, cropped


def preprocess(image_path: str):
    """Run preprocessing without tracing and return the normalized tensor."""
    img = cv2.imread(image_path)
    assert img is not None, f"Failed to read image: {image_path}"
    assert img.shape[:2] == (IMAGE_HEIGHT, IMAGE_WIDTH), (
        f"Unexpected input size {img.shape[:2]}, expected {(IMAGE_HEIGHT, IMAGE_WIDTH)}"
    )
    normalized, _ = preprocess_py2smr(img)
    return normalized


def preprocess_pipeline(image_path: str, output_path: str):
    """Trace preprocessing and export a py2smr pipeline JSON."""
    img = cv2.imread(image_path)
    assert img is not None, f"Failed to read image: {image_path}"
    assert img.shape[:2] == (IMAGE_HEIGHT, IMAGE_WIDTH), (
        f"Unexpected input size {img.shape[:2]}, expected {(IMAGE_HEIGHT, IMAGE_WIDTH)}"
    )

    (normalized, _cropped), ctx = preprocess_py2smr.trace(image=img)
    _inject_constant_values(
        ctx,
        {
            "src_points": np.array(
                [
                    [CROP_X1, CROP_Y1],
                    [CROP_X2, CROP_Y1],
                    [CROP_X2, CROP_Y2],
                ],
                dtype=np.float32,
            ),
            "dst_points": np.array(
                [
                    [0, 0],
                    [CROP_WIDTH, 0],
                    [CROP_WIDTH, CROP_HEIGHT],
                ],
                dtype=np.float32,
            ),
        },
    )

    spec = convert(ctx, output=output_path)
    return normalized, spec


def _add_vst_and_model_ops(
    spec: dict,
    model_path: str,
    model_input_name: str,
    model_output_names,
    model_name: str,
    image_path: str,
) -> dict:
    tensors = spec.setdefault("tensors", {})

    tensors["right_rgb"] = {
        "dimensions": [IMAGE_HEIGHT, IMAGE_WIDTH],
        "channels": 3,
        "data_type": convert_from_dtype(np.uint8),
        "is_placeholder": False,
        "usage": 6,
        "flag": mat_flag(smr.EDataType.UINT8, 3),
    }
    tensors["left_rgb"] = {
        "dimensions": [IMAGE_HEIGHT, IMAGE_WIDTH],
        "channels": 3,
        "data_type": convert_from_dtype(np.uint8),
        "is_placeholder": False,
        "usage": 6,
        "flag": mat_flag(smr.EDataType.UINT8, 3),
    }
    tensors["timestamp_tensor"] = {
        "dimensions": [1],
        "channels": 4,
        "data_type": convert_from_dtype(np.int32),
        "is_placeholder": False,
        "usage": 5,
    }
    tensors["camera_matrix_tensor"] = {
        "dimensions": [3, 3],
        "channels": 1,
        "data_type": convert_from_dtype(np.float32),
        "is_placeholder": False,
        "usage": 6,
        "flag": mat_flag(smr.EDataType.FLOAT32, 1),
    }
    tensors["predicted_score"] = {
        "dimensions": [1],
        "channels": 1,
        "data_type": convert_from_dtype(np.float32),
        "is_placeholder": True,
        "usage": 2,
        "flag": mat_flag(smr.EDataType.FLOAT32, 1),
    }
    tensors["predicted_class"] = {
        "dimensions": [1],
        "channels": 1,
        "data_type": convert_from_dtype(np.int32),
        "is_placeholder": True,
        "usage": 2,
        "flag": mat_flag(smr.EDataType.INT32, 1),
    }

    if "image" in tensors:
        tensors.pop("image", None)
    if "normalized_input_tensor" in tensors:
        tensors["normalized_input_tensor"]["is_placeholder"] = False
        tensors["normalized_input_tensor"]["usage"] = 6

    operators = spec.get("operators", [])
    for op in operators:
        if op.get("type") == type_to_name(smr.EOperatorType.APPLY_AFFINE):
            inputs = op.get("inputs", [])
            if len(inputs) >= 2 and inputs[1] == "image":
                inputs[1] = "left_rgb"

    vst_op = {
        "type": type_to_name(smr.EOperatorType.RECTIFIED_VST_ACCESS),
        "inputs": [],
        "outputs": ["right_rgb", "left_rgb", "timestamp_tensor", "camera_matrix_tensor"],
        "image_path": image_path,
    }
    operators = [vst_op] + operators

    model_op = {
        "type": type_to_name(smr.EOperatorType.RUN_MODEL_INFERENCE),
        "inputs": [{"name": model_input_name, "tensor": "normalized_input_tensor"}],
        "outputs": [
            {"name": model_output_names[0], "tensor": "predicted_score"},
            {"name": model_output_names[1], "tensor": "predicted_class"},
        ],
        "model": create_litert_model_spec(
            model_path,
            model_name,
            input_tensors=[
                {
                    "name": model_input_name,
                    "shape": [1, CROP_HEIGHT, CROP_WIDTH, 1],
                    "encoding_type": "FP32",
                    "alias_name": "normalized_input_tensor",
                }
            ],
            output_tensors=[
                {
                    "name": model_output_names[0],
                    "shape": [1],
                    "encoding_type": "FP32",
                    "alias_name": "predicted_score",
                },
                {
                    "name": model_output_names[1],
                    "shape": [1],
                    "encoding_type": "INT32",
                    "alias_name": "predicted_class",
                },
            ],
        ),
        "model_type": "tflite",
        "model_name": model_name,
    }
    operators.append(model_op)
    spec["operators"] = operators

    spec["inputs"] = []
    spec["outputs"] = ["cropped_image", "predicted_score", "predicted_class"]
    return spec


def main():
    """Run the MNIST wild example.

    This function demonstrates how to use the MNIST model for inference on custom images.
    """
    test_image = ROOT / "number_5.png"

    tmp_pipeline = "/tmp/tmp_pipeline.json"
    x, _ = preprocess_pipeline(str(test_image), tmp_pipeline)
    x0 = preprocess(str(test_image))
    assert np.allclose(x, x0, rtol=1e-4, atol=1e-4)

    with open(tmp_pipeline, "r", encoding="utf-8") as f:
        preprocess_spec = json.load(f)
    x2 = run_pipeline_python(preprocess_spec, {"image": cv2.imread(str(test_image))})["normalized_input_tensor"]
    assert np.allclose(x, x2, rtol=1e-4, atol=1e-4)

    with open(tmp_pipeline, "r", encoding="utf-8") as f:
        full_spec = json.load(f)
    full_spec = _add_vst_and_model_ops(
        full_spec,
        MODEL_ASSET,
        "input",
        ["score", "class"],
        "mnist",
        str(test_image),
    )
    with open(PIPE_JSON, "w", encoding="utf-8") as f:
        json.dump(full_spec, f, indent=2, ensure_ascii=False)
    print(f"Wrote {PIPE_JSON}")


if __name__ == "__main__":
    main()
