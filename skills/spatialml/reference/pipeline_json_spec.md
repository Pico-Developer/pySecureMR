# SpatialML Pipeline JSON Spec

This document consolidates how SecureMR pipelines are recorded to JSON by the Python tools and reconstructed by package deserializers.
Use it as a reference when authoring or reviewing pipeline specs.

## Package Manifest

Pipeline packages include a root `manifest.json` that points at pipeline JSON and package assets. Model metadata is carried inline by model inference operators. Use `runtime.supported_modes` to indicate whether the package is valid in XR mode, Spatial mode, or both:

```json
{
  "schema_version": "2",
  "id": "example-package",
  "pipelines": [{ "id": "main", "path": "pipeline/main.json" }],
  "runtime": { "supported_modes": ["xr", "spatial"] }
}
```

Allowed execution mode values are `xr` and `spatial`. Include only the modes supported by the operators and assets in that package.
Schema version 2 removes manifest-level `model` / `models` entries and external model metadata JSON files.

Most operators are valid in both modes. The current mode-specific exceptions are:

| Operator type | XR mode | Spatial mode | Notes |
|---------------|---------|--------------|-------|
| `XR_SECURE_MR_OPERATOR_TYPE_SWITCH_GLTF_RENDER_STATUS_PICO` | yes | no | GLTF rendering path. |
| `XR_SECURE_MR_OPERATOR_TYPE_UPDATE_GLTF_PICO` | yes | no | GLTF rendering path. |
| `XR_SECURE_MR_OPERATOR_TYPE_RENDER_TEXT_PICO` | yes | no | GLTF/text rendering path. |
| `XR_SECURE_MR_OPERATOR_TYPE_LOAD_TEXTURE_PICO` | yes | no | GLTF texture upload path. |
| `XR_SECURE_MR_OPERATOR_TYPE_SCENEGRAPH_VISIBILITY_PICO` | no | yes | Spatial scenegraph path. |
| `XR_SECURE_MR_OPERATOR_TYPE_UPDATE_COMPONENT_PICO` | no | yes | Spatial component update path. |

## Tensor Descriptors

Every tensor entry under `tensors` is an object with the following fields (`securemr/serialization.py:279-320`, `SecureMR_Samples/base/securemr_utils/serialization.cpp:479-499`):

| Key            | Type                 | Notes |
|----------------|---------------------|-------|
| `dimensions`   | array\<int>          | Spatial dimensions; flattened order follows SecureMR expectations. |
| `channels`     | int                  | Number of channels per element. |
| `data_type`    | int                  | Encodes `XrSecureMrTensorDataTypePICO`; see table below. |
| `is_placeholder` | bool              | Placeholders are allocated externally and become pipeline IO. |
| `usage`        | int                  | Raw `XrSecureMrTensorTypePICO` value (e.g. 6 for MAT, 2 for OUTPUT). |
| `flag`         | int (optional)       | Bitmask combining data type with `smr.BaseType` modifiers (`smr.BaseType.MAT`, channel bits, etc.). |
| `data` / `value` | array\<number> (optional) | Flattened tensor contents for preload. `data` and `value` are synonyms (`SecureMR_Samples/base/securemr_utils/serialization.cpp:340-414`). |
| `is_gltf`      | bool (optional)      | Marks GLTF placeholders that skip numeric attributes (`serialization.cpp:483-488`). |
| `tensor_type`  | string (optional)    | Shorthand accepted by package loaders for special tensors. Known values include `timestamp`, `gltf`, `scalar_array`, `point2_array`, `point3_array`, and `rgba_array`. |
| `size`         | int (optional)       | Element count for special tensor shorthand such as `point2_array`, `point3_array`, and `scalar_array`. |
| `asset`        | string (optional)    | Package-relative asset path for `tensor_type: "gltf"` tensors. Package loaders use it to materialize scene graph tensors from GLTF assets. |

Rule: tensors declared with MAT/matrix usage (`usage: 6`, `usage: "matrix"`, or `tensor_type: "matrix"`) must have at least two entries in `dimensions`. Use `[1, N]` or `[N, 1]` for vector-shaped matrix data; use scalar/point tensor usage for true one-dimensional scalar or point arrays.

### Data Type Codes

Python exposes explicit mappings (`securemr/serialization.py:44-73`):

| Code | NumPy dtype | `smr.EDataType` |
|------|-------------|-----------------|
| 1    | `np.uint8`  | `UINT8` |
| 2    | `np.int8`   | `INT8` |
| 3    | `np.uint16` | `UINT16` |
| 4    | `np.int16`  | `INT16` |
| 5    | `np.int32`  | `INT32` |
| 6    | `np.float32`| `FLOAT32` |
| 7    | `np.float64`| `FLOAT64` |

`convert_from_dtype` and `convert_to_dtype` convert between numeric codes, numpy dtypes, and `smr.EDataType`.
Package loaders may also accept string aliases for convenience, including `uint8`, `int8`, `uint16`, `int16`, `int32`, `float32`, `fp32`, `float64`, and `double`.

### Placeholder Semantics

- During save the Python helper normalizes `is_placeholder` so only tensors referenced in `inputs` or `outputs` remain placeholders (`securemr/serialization.py:528-535`).
- The C++ loader instantiates placeholders or allocates locals accordingly and preloads data when provided (`serialization.cpp:479-499`).

## Inline Model Metadata

Model inference operators carry their model metadata inline under the operator `model` key. For LiteRT/TFLite packages, use the shape produced by `securemr.pipeline_zoo.create_litert_model_spec`:

```json
{
  "bin_path": "model/model.tflite",
  "model_name": "main",
  "model_type": "tflite",
  "model_target": "npu",
  "cpu_target_num_threads": 1,
  "input": [
    {
      "name": "image",
      "shape": [1, 256, 256, 3],
      "encoding_type": "FP32",
      "alias_name": "image"
    }
  ],
  "output": [
    {
      "name": "output",
      "shape": [1, 1],
      "encoding_type": "FP32",
      "alias_name": "output"
    }
  ]
}
```

Model metadata fields are:

| Key | Type | Notes |
|-----|------|-------|
| `bin_path` | string | Package-relative model binary path. |
| `model_name` | string | Logical model name referenced by inference operators; package loaders commonly default to `main` when omitted. |
| `model_type` | string | Runtime family. New TFLite packages should use `tflite`. |
| `model_target` | string | Runtime backend. Use `npu` for NPU execution; `cpu` and `gpu` are also recognized where supported. |
| `cpu_target_num_threads` | int (optional) | CPU thread count, only meaningful when `model_target` is `cpu`. Omit it for NPU packages. |
| `input` / `output` | array\<object> | Model tensor metadata arrays. |
| `input[].name` / `output[].name` | string | Model graph node name. |
| `input[].shape` / `output[].shape` | array\<int> | Model tensor shape in model-runtime order. |
| `input[].encoding_type` / `output[].encoding_type` | string | Encoding such as `FP32`, `UINT8`, or other runtime-supported encodings. |
| `input[].alias_name` / `output[].alias_name` | string (optional) | Alias used by package generation and model-node binding helpers. |

## Operator Entries

Common fields for every operator entry:

| Key        | Type          | Notes |
|------------|---------------|-------|
| `type`     | string        | Canonical `XR_SECURE_MR_OPERATOR_TYPE_*_PICO` enumerant name. |
| `inputs`   | array         | Positional tensor references. Elements may be strings or objects containing a `tensor` key; both forms are accepted by the loader (`serialization.cpp:379-425`). |
| `outputs`  | array         | Same rules as `inputs`. |
| `attrs`    | array\<string> (optional) | Raw attribute strings. Python may promote well-known entries to named keys such as `flag`, `expression`, or `threshold` (`securemr/serialization.py:336-353`). |

### Operator Dictionary by `type`

Below lists the supported operators observed in the serializers together with the pipeline helpers that back them. Entries that the default loader does not yet recognize must be handled through `PipelineDeserializationOptions::customOperatorHandler` (until the missing branch is added).

#### `XR_SECURE_MR_OPERATOR_TYPE_RECTIFIED_VST_ACCESS_PICO` (`camera_access`)
- Sets up the rectified VST capture operator; entries are injected by the Python helper (`securemr/serialization.py:554-603`).
- Expect exactly four outputs in order: right RGB, left RGB, timestamp, camera matrix (`serialization.cpp:572-579`).
- Provide placeholder tensors sized per the SecureMR spec; the loader validates all four outputs.
- No inputs, attributes, or alternate keys; the alias `camera_access` maps to `RECTIFIED_VST_ACCESS`.

#### `XR_SECURE_MR_OPERATOR_TYPE_CAMERA_SPACE_TO_WORLD_PICO` (`camera_space_to_world`)
- Wraps `Pipeline::camSpace2XrLocal` and issues `XR_SECURE_MR_OPERATOR_TYPE_CAMERA_SPACE_TO_WORLD_PICO`.
- Supply the timestamp tensor from `camera_access` as `inputs[0]`; no other inputs are used.
- `outputs[0]` returns the right-eye 4×4 transform, `outputs[1]` (optional) returns the left-eye transform.
- Not yet wired into `serialization.cpp`; extend the loader or rely on a custom handler before round-tripping JSON.

#### `XR_SECURE_MR_OPERATOR_TYPE_UV_TO_3D_IN_CAM_SPACE_PICO` (`uv_to_3d_in_cam_space`)
- Implements `Pipeline::uv2Cam` (`XR_SECURE_MR_OPERATOR_TYPE_UV_TO_3D_IN_CAM_SPACE_PICO`) to lift UVs into 3D camera space.
- Requires five inputs ordered as UV coordinates, timestamp, camera matrix, left RGB image, right RGB image.
- `outputs[0]` stores the 3-channel floating-point point cloud; shape must mirror the UV tensor.
- The shipped loader lacks this branch; reuse tensors produced by `camera_access` and handle deserialization manually for now.

#### `XR_SECURE_MR_OPERATOR_TYPE_GET_AFFINE_PICO` (`get_affine`)
- Supports inline arrays (`src_points`/`dst_points`) or tensor inputs to describe the three source and destination points (`serialization.cpp:584-613`).
- `outputs[0]` must be a 2×3 float MAT representing the affine transform.
- The loader resolves point tensors from inlined tensor `value` fields when needed.
- No extra attributes beyond the optional point arrays.

#### `XR_SECURE_MR_OPERATOR_TYPE_APPLY_AFFINE_PICO` (`apply_affine`)
- Mirrors `Pipeline::applyAffine` (`XR_SECURE_MR_OPERATOR_TYPE_APPLY_AFFINE_PICO`) for image warps.
- `inputs[0]` supplies the 2×3 affine matrix and `inputs[1]` the source image tensor (`serialization.cpp:617-622`).
- `outputs[0]` receives the warped image; dimensions and channels must match the source tensor.
- No attributes or auxiliary keys are required.

#### `XR_SECURE_MR_OPERATOR_TYPE_APPLY_AFFINE_POINT_PICO` (`apply_affine_point`)
- Wraps `Pipeline::applyAffinePoint` to transform point arrays with the same affine matrix.
- Provide the affine matrix and the input point tensor as the two inputs; the first output carries transformed points.
- Ensure the result tensor has identical element count and channel layout as the input point tensor.
- Deserialization support is pending; hook it up via a custom handler until `serialization.cpp` gains this case.

#### `XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO` (`assignment`)
- `inputs[0]` is the source tensor and `outputs[0]` the destination (`serialization.cpp:623-762`).
- Use `src_slices` / `dst_slices` arrays for static `[start, end[, step]]` slicing; shape must match tensor rank.
- `src_slices_tensor` / `dst_slices_tensor` name tensors that hold slice descriptors (mutually exclusive with the inline arrays).
- `src_channel_slice` / `dst_channel_slice` narrow channels with 1–3 integers; omit all slice keys for full-tensor copies.

#### `XR_SECURE_MR_OPERATOR_TYPE_CUSTOMIZED_COMPARE_PICO` (`customized_compare`)
- Backed by `Pipeline::compareTo` (`XR_SECURE_MR_OPERATOR_TYPE_CUSTOMIZED_COMPARE_PICO`) to compare two tensors element-wise.
- `inputs[0]` and `inputs[1]` are the left/right operands; they must share the same shape and channel count.
- Provide the comparator via `comparison` (one of `">"`, `"<"`, `"=="`, `">="`, `"<="`, `"!="`) or fall back to `attrs[0]`, matching `securemr_operators.md` §4.
- `outputs[0]` stores the boolean/int result; add loader support before relying on automatic deserialization.

#### `XR_SECURE_MR_OPERATOR_TYPE_ALL_PICO` (`all`)
- Adds `XR_SECURE_MR_OPERATOR_TYPE_ALL_PICO` through `Pipeline::all` to reduce a tensor with logical AND.
- `inputs[0]` accepts any boolean-compatible tensor; `outputs[0]` is a vector whose channel count is either 1 or matches the operand.
- Reduction runs per channel when the result shares the operand’s channel count; otherwise all channels collapse into a single value.
- Currently missing in `serialization.cpp`; require a custom handler for JSON round-tripping.

#### `XR_SECURE_MR_OPERATOR_TYPE_ANY_PICO` (`any`)
- Builds on `Pipeline::any` for `XR_SECURE_MR_OPERATOR_TYPE_ANY_PICO`, reducing with logical OR.
- The wiring mirrors `all`: single operand tensor and a scalar or per-channel result tensor.
- Use this to detect whether any element is non-zero while preserving channel grouping when desired.
- Needs explicit loader support or custom handling because the default deserializer does not parse this type yet.

#### `XR_SECURE_MR_OPERATOR_TYPE_ARGMAX_PICO` (`argmax`)
- Wraps `Pipeline::argMax` to expose `XR_SECURE_MR_OPERATOR_TYPE_ARGMAX_PICO`.
- Single operand at `inputs[0]`; `outputs[0]` stores per-channel indices of the maximal element as integers.
- Result tensors typically use MAT usage with one channel (indices) unless you mirror the operand’s channels.
- Absent in `serialization.cpp`; extend the loader before serializing/deserializing this operator.

#### `XR_SECURE_MR_OPERATOR_TYPE_CONVERT_COLOR_PICO` (`cvt_color`)
- Mirrors `Pipeline::cvtColor` (`XR_SECURE_MR_OPERATOR_TYPE_CONVERT_COLOR_PICO`) to run OpenCV color conversions.
- `inputs[0]` is the source image; `outputs[0]` is the destination (`serialization.cpp:763-769`).
- Provide the OpenCV code via `flag` or `attrs[0]`; the helper casts numeric strings to ints (`securemr/serialization.py:336-344`).
- Ensure tensor shape, channel count, and data type align with the requested conversion.

#### `XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO` (`type_convert`)
- Exposed through `Pipeline::typeConvert` (delegates to assignment) for format changes.
- The loader auto-detects this path when input and output tensor data types differ and no slicing is specified (`serialization.cpp:623-654`).
- Use when only the tensor data type differs; dimensions and channels must still match.
- No attributes or inline configuration are required.

#### `XR_SECURE_MR_OPERATOR_TYPE_NORMALIZE_PICO` (`normalize`)
- Wraps `Pipeline::normalize` (`XR_SECURE_MR_OPERATOR_TYPE_NORMALIZE_PICO`) to apply L1/L2/INF/MINMAX normalization.
- `inputs[0]` is the tensor to normalize and `outputs[0]` the normalized result; shapes must match (`securemr_operators.md` §10).
- Set `normalize_type` (or `attrs[0]`) to one of `"l1"`, `"l2"`, `"inf"`, `"minmax"`; map to `XrSecureMrNormalizeTypePICO`.
- Alpha/Beta tuning mentioned in the operator guide is not exposed by the current pipeline helper; loader support is pending.

#### `XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO` (`arithmetic`)
- Implements the arithmetic-compose operator via `Pipeline::arithmetic` (`serialization.cpp:771-780`).
- All inputs are operands referenced in the expression; the first output receives the evaluation result.
- Use `expression` (or `attrs[0]`) to encode the formula with `{index}` placeholders and `+ - * /`, per `securemr_operators.md` §1.
- Ensure the result tensor usage matches MAT; operands must be MAT or arrays of MAT tensors.

#### `XR_SECURE_MR_OPERATOR_TYPE_ELEMENTWISE_MIN_PICO` / `XR_SECURE_MR_OPERATOR_TYPE_ELEMENTWISE_MAX_PICO` / `XR_SECURE_MR_OPERATOR_TYPE_ELEMENTWISE_MULTIPLY_PICO` / `XR_SECURE_MR_OPERATOR_TYPE_ELEMENTWISE_OR_PICO` / `XR_SECURE_MR_OPERATOR_TYPE_ELEMENTWISE_AND_PICO` (`elementwise_min`, `elementwise_max`, `elementwise_multiply`, `elementwise_or`, `elementwise_and`)
- Share the same wiring through elementwise helpers (`serialization.cpp:781-801`).
- Require two operands and at least one output; operands must have identical shape and channel layout.
- Select the operation by choosing one of the `type` strings listed above.
- Use integer tensors for logical ops (`or`, `and`) and numeric tensors for `min`, `max`, `multiply`.

#### `XR_SECURE_MR_OPERATOR_TYPE_NMS_PICO` (`nms`)
- Uses `Pipeline::nms` (`XR_SECURE_MR_OPERATOR_TYPE_NMS_PICO`) to filter bounding boxes (`serialization.cpp:803-827`).
- `inputs[0]` is scores and `inputs[1]` is boxes; up to three outputs deliver filtered scores, boxes, and indices.
- Provide `threshold` or `attrs[0]` as the IoU cut-off; loader parses numeric strings automatically (`securemr/serialization.py:347-352`).
- Result tensors are optional—supply only the ones you need, leaving others absent or null.

#### `XR_SECURE_MR_OPERATOR_TYPE_SOLVE_P_N_P_PICO` (`solve_p_n_p`)
- Wraps the SolvePnP helper (`serialization.cpp:829-893`) to estimate pose from 2D/3D correspondences.
- Expect three inputs (object points, image points, camera matrix) plus optional outputs for rotation and translation.
- Loader auto-converts tensor data into Point2/Point3 buffers when necessary, storing temporaries in `outResult.auxiliaryTensors`.
- Follows OpenCV semantics; ensure tensors originate from `camera_access` to reuse the correct camera intrinsics.

#### `XR_SECURE_MR_OPERATOR_TYPE_SORT_VEC_PICO` (`sort_vec`)
- Exposes `Pipeline::sortVec` (`XR_SECURE_MR_OPERATOR_TYPE_SORT_VEC_PICO`) to sort 1-D vectors.
- `inputs[0]` is the vector to sort; `outputs[0]` (optional) stores the sorted values, `outputs[1]` (optional) stores the original indices.
- Operates on single-channel tensors; indices output must use an integer data type.
- Add a custom deserializer branch—`serialization.cpp` currently lacks explicit support.

#### `XR_SECURE_MR_OPERATOR_TYPE_SORT_MAT_PICO` (`sort_mat`)
- Uses `Pipeline::sortMatByRow` / `sortMatByColumn` via `XR_SECURE_MR_OPERATOR_TYPE_SORT_MAT_PICO` (`serialization.cpp:894-914`).
- A single matrix input plus optional outputs for sorted values and indices; tensors must be 2-D single-channel mats.
- Configure the orientation with `mode` (`"row"` default) or `attrs[0]` (`"col"`/`"column"` switches to column sort).
- Result tensors, when present, must match the input shape; indices output uses integer types.

#### `XR_SECURE_MR_OPERATOR_TYPE_SVD_PICO` (`svd`)
- Wraps `Pipeline::singularValueDecomposition` for `XR_SECURE_MR_OPERATOR_TYPE_SVD_PICO`.
- Requires one matrix input; outputs for singular values (`w`), left singular vectors (`u`), and right singular vectors (`vt`) are optional.
- Shapes follow OpenCV’s SVD conventions; omit outputs you do not need to save memory.
- Serialization support is not yet implemented; wire it via a custom handler if you need JSON specs today.

#### `XR_SECURE_MR_OPERATOR_TYPE_NORM_PICO` (`norm`)
- Implements `Pipeline::norm` (`XR_SECURE_MR_OPERATOR_TYPE_NORM_PICO`) to compute the vector norm of a tensor.
- Single input tensor; single output scalar (or per-channel scalar) containing the norm result.
- Use when you need L2-style reductions without normalizing the tensor itself.
- Not processed by the default loader; requires a custom branch to deserialize.

#### `XR_SECURE_MR_OPERATOR_TYPE_SWAP_HWC_CHW_PICO` (`swap_hwc_chw`)
- Calls `Pipeline::convertHWC_CHW` (`XR_SECURE_MR_OPERATOR_TYPE_SWAP_HWC_CHW_PICO`) to reorder tensor layout.
- Inputs and outputs must use SecureMR channelized 3D matrix encoding. For an HWC tensor with `dimensions: [H, W]` and `channels: C`, the CHW tensor must be declared as `dimensions: [C, H]` and `channels: W`; the reverse direction uses the inverse mapping.
- Native binding uses the default unary operator names (`operand` / `result`), not `src` / `dst`.
- Handy before invoking neural-network runtimes expecting channel-first tensors.

#### `XR_SECURE_MR_OPERATOR_TYPE_INVERSION_PICO` (`inversion`)
- Uses `Pipeline::inversion` (`XR_SECURE_MR_OPERATOR_TYPE_INVERSION_PICO`) to compute matrix inverses.
- One matrix input; one matrix output storing the inverted result.
- Ensure the operand is square and invertible; data type should be float per SecureMR guidance.
- Needs explicit handling in the deserializer because the default code path does not know this type yet.

#### `XR_SECURE_MR_OPERATOR_TYPE_GET_TRANSFORM_MAT_PICO` (`get_transform_mat`; alias `MAKE_TRANSFORM_MAT`)
- Backs `Pipeline::transform` for building 4×4 transforms.
- Provide rotation and translation tensors; `inputs[2]` may optionally carry scale (omit to assume identity).
- `outputs[0]` becomes a 4×4 float MAT combining the supplied components.

#### `XR_SECURE_MR_OPERATOR_TYPE_LOAD_TEXTURE_PICO` (`load_texture`)
- Wraps `Pipeline::newTextureToGLTF` (`XR_SECURE_MR_OPERATOR_TYPE_LOAD_TEXTURE_PICO`) to inject textures into GLTF assets.
- `inputs[0]` is the GLTF placeholder tensor; `inputs[1]` is the RGB texture data.
- `outputs[0]` returns the generated texture identifier tensor for subsequent render commands.
- Not deserialized automatically yet; rely on a custom handler when recording GLTF texture uploads.

#### `XR_SECURE_MR_OPERATOR_TYPE_SWITCH_GLTF_RENDER_STATUS_PICO` (`switch_gltf_render_status`)
- Toggles rendering for a GLTF placeholder.
- `inputs[0]` is the GLTF placeholder tensor; `inputs[1]` may optionally provide the pose/transform.
- This is available in the Python creation and verification helpers for package specs that include GLTF assets.
- XR mode only.

#### `XR_SECURE_MR_OPERATOR_TYPE_UPDATE_GLTF_PICO` (`update_gltf`)
- Applies a GLTF update command to the referenced placeholder.
- `inputs[0]` is the GLTF placeholder tensor.
- Provide the update command via `update_type` or `attrs[0]`.
- XR mode only.

#### `XR_SECURE_MR_OPERATOR_TYPE_RENDER_TEXT_PICO` (`render_text`)
- Renders text into a texture-like tensor that can be used by GLTF/rendering operators.
- `inputs[0]` is the target placeholder or texture tensor.
- `config`/`attrs[0]` uses `typeface#language#width#height`; `text`/`attrs[1]` provides the rendered text.
- XR mode only.

#### `XR_SECURE_MR_OPERATOR_TYPE_SCENEGRAPH_VISIBILITY_PICO` (`scenegraph_visibility`)
- Toggles Spatial scenegraph visibility.
- `inputs[0]` is the scenegraph/component placeholder.
- Use `visible` or `attrs[0]` as a boolean-like value.
- Spatial mode only.

#### `XR_SECURE_MR_OPERATOR_TYPE_UPDATE_COMPONENT_PICO` (`update_component`)
- Updates a property of a specific entity in a Spatial scene graph.
- `scenegraph` or `inputs[0]` identifies the scene-graph tensor.
- `entity_path` identifies the target entity and must start with `/`; `/` refers to the scene root.
- `property` (or `target_property`) identifies the `SceneGraphProperty`, for example `Transform.Scale`, `Text.Content`, or `PBRMaterials[0].BaseColor`.
- `data` or `inputs[1]` identifies the tensor supplying the new property value.
- This operator has no outputs.
- The legacy `enabled` / `update_type` form is not a valid component update because it does not identify an entity, property, or data tensor.
- Spatial mode only.

#### `XR_SECURE_MR_OPERATOR_TYPE_MICROPHONE_PICO` (`microphone`)
- Captures microphone data into the output tensor.
- No required inputs; `outputs[0]` receives the audio buffer.
- Available in both XR and Spatial modes.

#### `XR_SECURE_MR_OPERATOR_TYPE_SPEAKER_PICO` (`speaker`)
- Sends audio data to the speaker path.
- `inputs[0]` is the audio tensor.
- Available in both XR and Spatial modes.

#### `XR_SECURE_MR_OPERATOR_TYPE_DEPTH_PICO` (`depth`)
- Captures or exposes depth data into the output tensor.
- No required inputs; `outputs[0]` receives the depth buffer.
- Available in both XR and Spatial modes.

#### `XR_SECURE_MR_OPERATOR_TYPE_JAVASCRIPT_PICO` (`javascript`; alias `JS_SCRIPTING`)
- Executes a JavaScript snippet to populate output tensors from named inputs.
- Use `script` or `attrs[0]` for the JavaScript source.
- `inputs` and `outputs` may use object references with `name` and `tensor` to preserve script-visible aliases.

#### `XR_SECURE_MR_OPERATOR_TYPE_UNKNOWN_PICO` (`unknown`)
- Reserved for explicitly unknown or pass-through operators in tests and custom pipelines.
- Package deserializers should treat this like a custom operator unless they intentionally implement a fallback.

#### `XR_SECURE_MR_OPERATOR_TYPE_RUN_MODEL_INFERENCE_PICO` (`run_algorithm`)
- Designed for model execution pipelines (`serialization.cpp:915-996`).
- `inputs` and `outputs` are arrays of `{ "name": alias, "tensor": tensor_name }`; strings default aliases to tensor names.
- For new SpatialML Pipeline Zoo packages, put model metadata inline under `model` with `bin_path`, `model_name`, `model_type: "tflite"`, `model_target` (for example `npu`), and optional `cpu_target_num_threads`.
- Schema version 2 does not use manifest-level model ids, `model_id`, external model metadata JSON files, `model_asset`, or filesystem `model_file` fields for package-authored TFLite operators.
- Python utilities (`add_model_inference_operator`, `convert_python_custom_to_run_algorithm`) populate tensors and metadata automatically (`securemr/serialization.py:621-712`).

#### Custom operators
- Unrecognized `type` values fall back to `PipelineDeserializationOptions::customOperatorHandler` (`serialization.cpp:918-920`). Provide `attrs` or additional keys your handler understands.
- `name_to_type` in Python tolerates numeric enum values and aliases, supporting custom handlers (`securemr/serialization.py:103-206`).

## Inputs and Outputs Arrays

- Besides marking placeholders, the `inputs` / `outputs` lists at the top level preserve the ordering expected by task wrappers (`securemr/serialization.py:432-447`, `serialization.cpp:520-528`).
- When tensors appear in both lists the Python helper converts them to locals to avoid conflicts (`securemr/serialization.py:662-669`).

## Saving and Loading Workflow

1. The extended Python `Pipeline` records every allocation and connection into `self.spec` while user code builds the graph (`securemr/serialization.py:264-390`).
2. Calling `Pipeline.save` writes the spec with UTF-8 encoding and pretty formatting (`securemr/serialization.py:536-540`).
3. The C++ loader validates `tensors` and `operators`, constructs a `Pipeline`, and wires operators using the described keys. Any mismatch produces descriptive errors (`serialization.cpp:461-917`).
