#!/usr/bin/env sh
set -ex

# pytest -s tests/py2smr_ops/test_all.py
# pytest -s tests/py2smr_ops/test_any.py
# pytest -s tests/py2smr_ops/test_apply_affine_point.py
# pytest -s tests/py2smr_ops/test_apply_affine.py
# pytest -s tests/py2smr_ops/test_argmax.py
# pytest -s tests/py2smr_ops/test_arithmetic.py
# pytest -s tests/py2smr_ops/test_assignment.py
# pytest -s tests/py2smr_ops/test_camera_space_to_world.py      # stub
# pytest -s tests/py2smr_ops/test_convert_color.py
# pytest -s tests/py2smr_ops/test_customized_compare.py
# pytest -s tests/py2smr_ops/test_elementwise_and.py
# pytest -s tests/py2smr_ops/test_elementwise_max.py
# pytest -s tests/py2smr_ops/test_elementwise_min.py
# pytest -s tests/py2smr_ops/test_elementwise_multiply.py
# pytest -s tests/py2smr_ops/test_elementwise_or.py
# pytest -s tests/py2smr_ops/test_get_affine.py
# pytest -s tests/py2smr_ops/test_get_transform_mat.py
# pytest -s tests/py2smr_ops/test_inversion.py
# # pytest -s tests/py2smr_ops/test_javascript.py                   # stub, failed
# pytest -s tests/py2smr_ops/test_load_texture.py                 # stub
# pytest -s tests/py2smr_ops/test_nms.py
# pytest -s tests/py2smr_ops/test_normalize.py
# pytest -s tests/py2smr_ops/test_norm.py
# pytest -s tests/py2smr_ops/test_rectified_vst_access.py         # load image from image file
# pytest -s tests/py2smr_ops/test_render_text.py                  # stub
pytest -s tests/py2smr_ops/test_run_model_inference.py          # using model-inspect running on device, failed
# pytest -s tests/py2smr_ops/test_solve_pnp.py
# pytest -s tests/py2smr_ops/test_sort_mat.py
# pytest -s tests/py2smr_ops/test_sort_vec.py
# pytest -s tests/py2smr_ops/test_svd.py
# pytest -s tests/py2smr_ops/test_swap_hwc_chw.py
# pytest -s tests/py2smr_ops/test_switch_gltf_render_status.py    # stub
# pytest -s tests/py2smr_ops/test_unknown.py
# pytest -s tests/py2smr_ops/test_update_gltf.py                  # stub
# pytest -s tests/py2smr_ops/test_uv_to_3d_in_cam_space.py        # stub
