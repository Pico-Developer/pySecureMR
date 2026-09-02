from securemr.core.types import EOperatorType
from securemr.py2smr import ops
from securemr.py2smr.verifier import _get_operator_type, run_pipeline_python
import numpy as np


def test_verifier_accepts_documented_operator_aliases():
    assert _get_operator_type("camera_access") == EOperatorType.RECTIFIED_VST_ACCESS
    assert _get_operator_type("cvt_color") == EOperatorType.CONVERT_COLOR
    assert _get_operator_type("arithmetic") == EOperatorType.ARITHMETIC_COMPOSE
    assert _get_operator_type("run_algorithm") == EOperatorType.RUN_MODEL_INFERENCE
    # type_convert is an authoring alias for the assignment enum. The loader
    # selects automatic dtype conversion from the tensor descriptors.
    assert _get_operator_type("type_convert") == EOperatorType.ASSIGNMENT
    assert _get_operator_type("XR_SECURE_MR_OPERATOR_TYPE_MAKE_TRANSFORM_MAT_PICO") == EOperatorType.GET_TRANSFORM_MAT
    assert _get_operator_type("uv2_cam") == EOperatorType.UV_TO_3D_IN_CAM_SPACE
    assert _get_operator_type("transform") == EOperatorType.GET_TRANSFORM_MAT
    assert _get_operator_type("draw_text") == EOperatorType.RENDER_TEXT
    assert _get_operator_type("render_gltf") == EOperatorType.SWITCH_GLTF_RENDER_STATUS


def test_deserialized_operator_types_are_spec_creatable():
    deserialized_operator_creators = {
        "UNKNOWN": "unknown",
        "ARITHMETIC_COMPOSE": "arithmetic",
        "ELEMENTWISE_MIN": "elementwise_min",
        "ELEMENTWISE_MAX": "elementwise_max",
        "ELEMENTWISE_MULTIPLY": "elementwise_multiply",
        "CUSTOMIZED_COMPARE": "customized_compare",
        "ELEMENTWISE_OR": "elementwise_or",
        "ELEMENTWISE_AND": "elementwise_and",
        "ALL": "all",
        "ANY": "any",
        "NMS": "nms",
        "SOLVE_P_N_P": "solve_pnp",
        "GET_AFFINE": "get_affine",
        "APPLY_AFFINE": "apply_affine",
        "APPLY_AFFINE_POINT": "apply_affine_point",
        "UV_TO_3D_IN_CAM_SPACE": "uv_to_3d_in_cam_space",
        "ASSIGNMENT": "assignment",
        "RUN_MODEL_INFERENCE": "run_model_inference",
        "NORMALIZE": "normalize",
        "CAMERA_SPACE_TO_WORLD": "camera_space_to_world",
        "RECTIFIED_VST_ACCESS": "rectified_vst_access",
        "ARGMAX": "argmax",
        "CONVERT_COLOR": "convert_color",
        "SORT_VEC": "sort_vec",
        "INVERSION": "inversion",
        "GET_TRANSFORM_MAT": "get_transform_mat",
        "SORT_MAT": "sort_mat",
        "SWITCH_GLTF_RENDER_STATUS": "switch_gltf_render_status",
        "UPDATE_GLTF": "update_gltf",
        "RENDER_TEXT": "render_text",
        "LOAD_TEXTURE": "load_texture",
        "SVD": "svd",
        "NORM": "norm",
        "SWAP_HWC_CHW": "swap_hwc_chw",
        "SCENEGRAPH_VISIBILITY": "scenegraph_visibility",
        "UPDATE_COMPONENT": "update_component",
        "JAVASCRIPT": "javascript",
        "MICROPHONE": "microphone",
        "SPEAKER": "speaker",
        "DEPTH": "depth",
    }

    missing_enum_members = [name for name in deserialized_operator_creators if not hasattr(EOperatorType, name)]
    missing_creators = [creator for creator in deserialized_operator_creators.values() if not hasattr(ops, creator)]
    assert missing_enum_members == []
    assert missing_creators == []


def test_python_consumer_accepts_schema_comparison_field():
    spec = {
        "tensors": {
            "a": {"dimensions": [1, 3], "channels": 1, "data_type": 6},
            "b": {"dimensions": [1, 3], "channels": 1, "data_type": 6},
            "out": {"dimensions": [1, 3], "channels": 1, "data_type": 5},
        },
        "operators": [{
            "type": "XR_SECURE_MR_OPERATOR_TYPE_CUSTOMIZED_COMPARE_PICO",
            "inputs": ["a", "b"],
            "outputs": ["out"],
            "comparison": "<",
        }],
        "inputs": ["a", "b"],
        "outputs": ["out"],
    }

    result = run_pipeline_python(
        spec,
        {"a": np.array([1.0, 3.0, 2.0], dtype=np.float32),
         "b": np.array([2.0, 2.0, 2.0], dtype=np.float32)},
    )
    np.testing.assert_array_equal(result["out"], [1, 0, 0])
