---
name: spatialml
description: Convert ML models (ONNX/TFLite/PB/PyTorch) to QNN context binaries. Use when asked to convert models with convert_model.sh, generate QNN context binaries, or document SecureMRTools model conversion steps.
---

# Spatial ML

## Overview

Provide bunch of debugging tools to help deploy ML model on Android(Pico) device.

## Model Conversion

### Conversion workflow

1. Ensure Docker Desktop is running and the model file is present on disk.
2. Run the conversion script from the directory that contains the input model.
3. Verify the output folder and context binary are generated.

## # Script usage

Use the bundled script to perform the conversion:

```bash
./scripts/convert_model.sh --input <model_file> [--custom_io <custom_io.yml>]
```

Behavior:
- Accepts: `.onnx`, `.tflite`, `.pb`, `.pt`, `.pth`
- Runs the SecureMRTools container (`ghcr.io/pico-developer/securemr_tool:v0`)
- Writes output to `<model_name>_output/` in the current working directory
- Context binary file name: `<model_name>.serialized.bin`

### Outputs to verify

- `<model_name>_output/<model_name>.serialized.bin`
- `<model_name>_output/model.json`

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

### Generate pipeline json from model

You can also generate `pipeline.json` derectly from context binary file to get a pipeline.json as a starter for pipeline inspect.

```bash
python3.10 -m securemr.qnn.qnn_to_pipeline /path/to/qnn_context.bin --output /path/to/pipeline.json
```

### Visualize pipeline JSON

```
python3.10 -m securemr.viz.pipeline_viz <path-to-pipeline.json>
```

## Resources

### scripts/

- `scripts/convert_model.sh`: Docker-based conversion wrapper.

### reference/

- `pipeline_json_spec.md`: Specification for securemr pipeline json. You should refer this if you need to dump securemr pipeline to json.

- `operator_tips.md`: If you find operator is not supported in pipeline_json_spec.md, read operator_tips.md for advanced tricks to implement it.
