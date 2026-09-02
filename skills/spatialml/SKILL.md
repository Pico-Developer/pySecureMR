---
name: spatialml
description: Author SpatialML packages using LiteRT/TFLite model assets and SecureMR pipeline JSON.
---

# Spatial ML

## Overview

Provide tools to author, inspect, and debug SpatialML/SecureMR pipeline packages for Android(Pico) runtimes and other package deserializers.

## SpatialML package authoring

SpatialML packages use LiteRT/TFLite assets.

### Package schema

The package schema expects a directory containing:

- `manifest.json` at the package root.
- One or more pipeline JSON files referenced by `manifest.pipelines[].path`.
- TFLite model files referenced by inline `RUN_MODEL_INFERENCE` operator model specs.
- Optional binary assets such as glTF files referenced by package-relative paths.
- `manifest.runtime.supported_modes` lists where the package is valid: `xr`, `spatial`, or both. Set this explicitly because some operators differ by execution mode.

Use `securemr.pipeline_zoo` to emit schema-compatible fields:

```python
from securemr.pipeline_zoo import (
    PipelinePackageEntry,
    PipelineZooPackageSpec,
    configure_litert_inference_operator,
    write_pipeline_zoo_package,
)

pipeline = {
    "tensors": {},
    "operators": [
        configure_litert_inference_operator(
            {"type": "RUN_MODEL_INFERENCE", "inputs": [], "outputs": []},
            model_path="model/face_detector.tflite",
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
    supported_modes=["xr", "spatial"],
    runtime={"detection_tensor": "detections"},
)

write_pipeline_zoo_package(
    "face-mediapipe-pipeline",
    package,
    pipelines={"detection": pipeline},
    assets={"model/face_detector.tflite": "face_detector.tflite"},
)
```

For `RUN_MODEL_INFERENCE` operators in new packages:

- Put model metadata inline under `model`; do not use manifest-level `model` / `models`.
- Set `model_type` to `tflite`.
- Set `model_target` to the desired backend (`npu` by default).
- Set `cpu_target_num_threads` when CPU fallback is expected.
- Include `bin_path`, `model_name`, `model_type`, and `model_target` in the inline model spec.
- Do not use `model_asset`, `model_file`, or `model_id` fields in package-authored TFLite operators.

### CLI package creation

Use `pyspatialml package create` for file-based packages. It copies source pipelines
into `pipeline/`, discovers `.tflite` model references from inline
`model.bin_path`, copies models into `model/`, discovers GLTF tensor/operator asset
references, copies them into `gltf/`, and rewrites the packaged pipeline JSON.

```bash
pyspatialml package create \
  --id face-demo \
  --pipeline detection=./detection.json \
  --pipeline display=./display.json \
  --supported-mode spatial \
  --output ./face-demo-package

pyspatialml package validate ./face-demo-package
pyspatialml package inspect ./face-demo-package
```

Pass `--asset-root ./assets` when referenced assets are not beside the source
pipeline and are not relative to the current working directory.

## LiteRT CLI integration

Use the `pyspatialml` CLI for LiteRT tool resolution. It detects `litert` on `PATH`,
honors `PYSPATIALML_LITERT`, or installs a managed copy into the pySpatialML tool
cache when a LiteRT-backed command needs it.

```bash
pyspatialml tools litert status
pyspatialml tools litert install
pyspatialml model run -- --help
pyspatialml model convert -- --help
pyspatialml model quantize -- --help
pyspatialml model benchmark -- --help
pyspatialml visualize model -- --help
```

## pySpatialML inspect debugging on Android(Pico)

Use pySpatialML/securemr helpers to sanity-check pipelines on device.

### Install & verify

If there is `.venv` in current directory, `source .venv/bin/activate` first.

```bash
python3.10 -m ensurepip
python3.10 -m pip install git+https://github.com/Pico-Developer/pySpatialML.git
pyspatialml --version
python3.10 -c "import securemr"
```

- Requires Python 3.10.x and adb-accessible Android device (developer options + USB debugging).
- Installs bundled inspect APKs automatically unless `--apk` overrides them.

### Visualize pipeline JSON

```
python3.10 -m securemr.viz.pipeline_viz <path-to-pipeline.json>
```

## Resources

### reference/

- `pipeline_json_spec.md`: Specification for securemr pipeline json. You should refer this if you need to dump securemr pipeline to json.
- `operator_tips.md`: If you find operator is not supported in pipeline_json_spec.md, read operator_tips.md for advanced tricks to implement it.
