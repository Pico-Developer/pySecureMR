# pySpatialML XR Pipeline Runner

This is a small NativeActivity APK project for running SpatialML pipeline packages on a PICO device.
It uses the shared NativeActivity/OpenXR base from:

`external/SpatialML-XR-Utils`

The runner is intended to be installed once, then driven by `adb` or a future
`pyspatialml run device` command by pushing package/input files to app storage.

## Build

```bash
cd pyspatialml/xr_pipeline_runner
./gradlew :app:assembleDebug
```

The debug APK is written under:

```text
app/build/outputs/apk/debug/app-debug.apk
```

The packaged copy used by `pyspatialml run device` is:

```text
pyspatialml/apks/pyspatialml_xr_runner-debug.apk
```

## Device Properties

Set these before launching the activity:

```bash
adb shell setprop debug.pyspatialml.xr_runner.package /sdcard/Android/data/com.bytedance.pico.pyspatialml.xr_runner/files/package
adb shell setprop debug.pyspatialml.xr_runner.output /sdcard/Android/data/com.bytedance.pico.pyspatialml.xr_runner/files/outputs
adb shell setprop debug.pyspatialml.xr_runner.input /sdcard/Android/data/com.bytedance.pico.pyspatialml.xr_runner/files/input.jpg
adb shell setprop debug.pyspatialml.xr_runner.use_vst false
adb shell setprop debug.pyspatialml.xr_runner.loop false
adb shell setprop debug.pyspatialml.xr_runner.interval_ms 50
adb shell setprop debug.pyspatialml.xr_runner.pipelines detection,display
```

If `debug.pyspatialml.xr_runner.pipelines` is empty, the runner uses manifest
pipeline order. If `use_vst=false`, `RECTIFIED_VST_ACCESS` operators are removed
from loaded pipeline JSON so pushed image tensors are not overwritten by device
camera access.

For image input, `input` may point to an image file or to a directory. When it is
a directory, the runner looks for files named after input tensors, such as
`vst_left_image.jpg`, `vst_right_image.jpg`, or `.bin` raw tensor files.

## Launch

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell pm grant com.bytedance.pico.pyspatialml.xr_runner android.permission.CAMERA
adb shell pm grant com.bytedance.pico.pyspatialml.xr_runner com.picovr.permission.SPATIAL_DATA
adb shell am start -n com.bytedance.pico.pyspatialml.xr_runner/android.app.NativeActivity
```

Outputs are read back as raw binary files under the configured output directory,
along with `status.json`.

## Run From Host

The helper script builds/installs the APK if requested, pushes a package and
optional input image, launches the runner, waits, and pulls outputs:

```bash
./scripts/run_xr_pipeline.py /path/to/package-or-package.zip \
  --input /path/to/image.jpg \
  --pipeline detection \
  --output-dir ./xr_runner_outputs \
  --duration 5
```

For VST camera input instead of a pushed image, pass `--use-vst`. For repeated
execution, pass `--loop`; add `--keep-running` if the app should remain running
after the script pulls the first output batch.
