---
name: spatialml
description: Author SpatialML Pipeline Zoo packages using LiteRT/TFLite model assets and SecureMR pipeline JSON. QNN context-binary conversion remains documented only for legacy workflows and should not be used for new packages.
---

# Spatial ML

## Overview

Provide tools to author, inspect, and debug SpatialML/SecureMR pipeline packages for Android(Pico) runtimes and other package deserializers.

## SpatialML Pipeline Zoo package authoring

For new Pipeline Zoo packages, prefer LiteRT/TFLite assets over QNN context binaries.

### Package schema

The package schema expects a directory containing:

- `manifest.json` at the package root.
- One or more pipeline JSON files referenced by `manifest.pipelines[].path`.
- A TFLite model file referenced by `manifest.model.bin_path`.
- A LiteRT model metadata file referenced by `manifest.model.json_path`.
- Optional binary assets such as glTF files referenced by package-relative paths.
- `manifest.runtime.supported_modes` lists where the package is valid: `xr`, `spatial`, or both. Set this explicitly because some operators differ by execution mode.

Use `securemr.pipeline_zoo` to emit schema-compatible fields:

```python
from securemr.pipeline_zoo import (
    ModelPackageSpec,
    PipelinePackageEntry,
    PipelineZooPackageSpec,
    configure_litert_inference_operator,
    create_litert_model_json,
    write_pipeline_zoo_package,
)

pipeline = {
    "tensors": {},
    "operators": [
        configure_litert_inference_operator(
            {"type": "RUN_MODEL_INFERENCE", "inputs": [], "outputs": []},
            model_name="face_detector",
            model_target="npu",
        )
    ],
    "inputs": [],
    "outputs": [],
}

package = PipelineZooPackageSpec(
    package_id="face",
    pipelines=[PipelinePackageEntry("detection", "pipeline/face_detection_pipeline.json")],
    model=ModelPackageSpec("model/face_detector.tflite", "model/model.json"),
    supported_modes=["xr", "spatial"],
    runtime={"detection_tensor": "detections"},
)

write_pipeline_zoo_package(
    "face-mediapipe-pipeline",
    package,
    pipelines={"detection": pipeline},
    model_json=create_litert_model_json("model/face_detector.tflite", "face_detector"),
    assets={"model/face_detector.tflite": "face_detector.tflite"},
)
```

For `RUN_MODEL_INFERENCE` operators in new packages:

- Set `model_type` to `litert`.
- Set `model_target` to the desired backend (`npu` by default).
- Set `cpu_target_num_threads` when CPU fallback is expected.
- Select the model using one of the schema-supported forms:
  - `model: "<manifest-model-id>"` to reference a manifest model by id.
  - `model_id: "<manifest-model-id>"` as an explicit model-id selector.
  - `model: { ... }` to provide an inline model spec with fields such as `bin_path`, `model_name`, `model_type`, and `model_target`.
- Do not use QNN-only `model_asset` or filesystem `model_file` fields in package-authored LiteRT operators.

## Legacy QNN conversion (deprecated)

QNN conversion tools are kept only for existing context-binary workflows. Do not use them for new Pipeline Zoo packages.

### Deprecated conversion workflow

1. Ensure Docker Desktop is running and the model file is present on disk.
2. Run the deprecated conversion script from the directory that contains the input model.
3. Verify the output folder and context binary are generated.

### Script usage

Deprecated script usage:

```bash
./scripts/convert_model_qnn237.sh --input <model_file> [--custom_io <custom_io.yml>]
```

Behavior:
- Accepts: `.onnx`, `.tflite`, `.pb`, `.pt`, `.pth`
- Runs the legacy SecureMRTools QNN container
- Writes output to `<model_name>_output/` in the current working directory
- Context binary file name: `<model_name>.serialized.bin`

### Outputs to verify

- `<model_name>_output/<model_name>.serialized.bin`
- `<model_name>_output/model.json`

### Tips

- MUST make sure input model_file is correct. Fed with input, it can output expected result. If onnx file is converted from pt file manually, MUST check onnx output is same as pt output.
- If input model_file is broken, the coverted serialized.bin is broken too.

## PySecureMR inspect debugging on Android(Pico)

Use `pySecureMR` to sanity-check models/pipelines on device.

### Install & verify

If there is `.venv` in current directory, `source .venv/bin/activate` first.

```bash
python3.10 -m ensurepip
python3.10 -m pip install git+https://github.com/Pico-Developer/pySecureMR.git
python3.10 -c "import securemr"
```

- Requires Python 3.10.x and adb-accessible Android device (developer options + USB debugging).
- Installs bundled inspect APKs automatically unless `--apk` overrides them.

### Model inspector

Run the packaged model inspector, push model/spec to device, pull outputs under `tmp_data/model_inspect_outputs_<timestamp>`:

```bash
python3.10 -m securemr.inspect.model_cli \
  --model model.serialized.bin \
  --json model.serialized.json \
  [--input input.bin] \
  [--output expected.bin|v1,v2,...] \
  [--output-name <tensor_name>] \
  [--duration 20] \
  [--device <adb_id>] \
  [--apk /path/to/model_inspect-debug.apk]
```

### Pipeline inspector

Inspect a SecureMR pipeline JSON; outputs land in `tmp_data/pipeline_inspect_outputs_<timestamp>`:

```bash
python3.10 -m securemr.inspect.pipeline_cli \
  --pipeline pipeline.json \
  [--input input.bin|image.(png|jpg)] \
  [--input-tensor <tensor_name>] \
  [--output expected.bin]... \
  [--duration 30] \
  [--device <adb_id>] \
  [--apk /path/to/pipeline_inspect-debug.apk]
```

### Generate pipeline json from QNN context binary (deprecated)

You can also generate `pipeline.json` directly from a legacy QNN context binary file as a starter for pipeline inspect.

```bash
python3.10 -m securemr.qnn.qnn_to_pipeline /path/to/qnn_context.bin --output /path/to/pipeline.json
```

### Visualize pipeline JSON

```
python3.10 -m securemr.viz.pipeline_viz <path-to-pipeline.json>
```

## Resources

### scripts/

- `scripts/convert_model_qnn220.sh`: Deprecated Docker-based QNN conversion wrapper, target for pico4 ultra.
- `scripts/convert_model_qnn237.sh`: Deprecated Docker-based QNN conversion wrapper, target for next platform.

### reference/

- `pipeline_json_spec.md`: Specification for securemr pipeline json. You should refer this if you need to dump securemr pipeline to json.
- `operator_tips.md`: If you find operator is not supported in pipeline_json_spec.md, read operator_tips.md for advanced tricks to implement it.
