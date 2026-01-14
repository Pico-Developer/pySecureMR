import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .compare_outputs import compare_outputs
from .inspect_utils import (
    AdbError,
    DeviceContext,
    capture_adb,
    capture_adb_binary,
    ensure_screen_on,
    install_apk,
    run_adb,
    select_device,
    start_logcat,
    turn_screen_off,
)


PACKAGE_NAME = "com.bytedance.pico.secure_mr_demo.pipeline_inspect"
COMPONENT = f"{PACKAGE_NAME}/.PipelineInspectActivity"
OUTPUT_PREFIX = "pipeline_inspect_output_"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SecureMR pipeline_inspect on an Android device."
    )
    parser.add_argument("--input", help="Input tensor file (raw tensor bytes).")
    parser.add_argument("--input-tensor", help="Tensor name to map --input into")
    parser.add_argument("--duration", type=int, default=30, help="Run time in seconds")
    parser.add_argument(
        "--output",
        action="append",
        default=[],
        help="Expected output file (repeatable)",
    )
    parser.add_argument("--pipeline", required=True, help="Pipeline JSON file")
    parser.add_argument("--device", help="ADB device id (optional)")
    parser.add_argument(
        "--apk",
        default="",
        help="Path to pipeline_inspect APK (optional)",
    )
    parser.add_argument(
        "--force-install-apk",
        action="store_true",
        help="Force install pipeline_inspect APK even if it already exists",
    )
    return parser.parse_args(argv)


def resolve_apk_path(user_path: str) -> Path:
    if user_path:
        return Path(user_path)
    return Path(__file__).resolve().parent / "apks" / "pipeline_inspect-debug.apk"


def prepare_device(device: DeviceContext, apk_path: Path, force_install: bool = False) -> None:
    if not apk_path.exists():
        raise FileNotFoundError(f"APK not found: {apk_path}")
    print("Installing pipeline_inspect APK...")
    install_apk(device, str(apk_path), PACKAGE_NAME, force=force_install)


def clean_device_dirs(device: DeviceContext, tmp_dir: str, output_dirs: list[str]) -> None:
    run_adb(["shell", "rm", "-rf", tmp_dir], device, check=False)
    run_adb(["shell", "mkdir", "-p", tmp_dir], device)
    for output_dir in output_dirs:
        run_adb(["shell", "rm", "-f", f"{output_dir}/*"], device, check=False)


def device_input_path(input_path: str, tmp_dir: str) -> str:
    ext = os.path.splitext(input_path)[1].lower().lstrip(".")
    if ext in {"png", "jpg", "jpeg"}:
        return f"{tmp_dir}/input.{ext}"
    return f"{tmp_dir}/input.bin"


def pull_outputs(device: DeviceContext, output_dirs: list[str], local_dir: Path) -> bool:
    local_dir.mkdir(parents=True, exist_ok=True)
    outputs_pulled = False
    for directory in output_dirs:
        find_cmd = [
            "shell",
            "find",
            directory,
            "-maxdepth",
            "1",
            "-name",
            f"{OUTPUT_PREFIX}*.bin",
            "-type",
            "f",
        ]
        listing = capture_adb(find_cmd, device, check=False)
        for path in listing.splitlines():
            path = path.strip().replace("\r", "")
            if not path:
                continue
            base = os.path.basename(path)
            data = capture_adb_binary(["shell", "cat", path], device, check=False)
            if not data:
                continue
            with open(local_dir / base, "wb") as file_handle:
                file_handle.write(data)
            outputs_pulled = True
    return outputs_pulled


def run_pipeline_inspect(args: argparse.Namespace) -> int:
    pipeline_file = Path(args.pipeline)
    if not pipeline_file.is_file():
        raise FileNotFoundError(f"Pipeline file not found: {pipeline_file}")
    if args.input and not Path(args.input).is_file():
        raise FileNotFoundError(f"Input tensor file not found: {args.input}")

    device = select_device(args.device)
    apk_path = resolve_apk_path(args.apk)
    prepare_device(device, apk_path, force_install=args.force_install_apk)

    device_tmp_dir = f"/sdcard/Android/data/{PACKAGE_NAME}/files"
    device_pipeline_path = f"{device_tmp_dir}/pipeline.json"
    device_output_dirs = [
        f"{device_tmp_dir}/pipeline_inspect",
        "/data/local/tmp/securemr_pipeline_inspect",
    ]

    clean_device_dirs(device, device_tmp_dir, device_output_dirs)

    print("Pushing pipeline to device...")
    run_adb(["push", str(pipeline_file), device_pipeline_path], device)

    # Also push any .bin files in the same directory as the pipeline file
    pipeline_dir = pipeline_file.parent
    for bin_file in pipeline_dir.glob("*.bin"):
        print(f"Pushing model/binary file {bin_file.name} to {device_tmp_dir}...")
        run_adb(["push", str(bin_file), device_tmp_dir + "/"], device)

    if args.input:
        remote_input = device_input_path(args.input, device_tmp_dir)
        print(f"Pushing input to {remote_input}...")
        run_adb(["push", args.input, remote_input], device)
        run_adb(
            ["shell", "setprop", "debug.securemr.pipeline_inspect.input", remote_input],
            device,
        )
    else:
        run_adb(["shell", "setprop", "debug.securemr.pipeline_inspect.input", "''"], device)

    if args.input_tensor:
        run_adb(
            [
                "shell",
                "setprop",
                "debug.securemr.pipeline_inspect.input_tensor",
                args.input_tensor,
            ],
            device,
        )
    else:
        run_adb(
            ["shell", "setprop", "debug.securemr.pipeline_inspect.input_tensor", "''"],
            device,
        )

    run_adb(
        [
            "shell",
            "setprop",
            "debug.securemr.pipeline_inspect.pipeline",
            device_pipeline_path,
        ],
        device,
    )

    run_adb(
        ["shell", "pm", "grant", PACKAGE_NAME, "android.permission.CAMERA"],
        device,
        check=False,
    )

    ensure_screen_on(device)
    existing_pid = capture_adb(["shell", "pidof", PACKAGE_NAME], device, check=False).strip()
    if existing_pid:
        run_adb(["shell", "am", "force-stop", PACKAGE_NAME], device, check=False)
        time.sleep(3)

    logcat_proc = start_logcat(PACKAGE_NAME, "PipelineInspect", device)
    try:
        run_adb(["shell", "am", "force-stop", "com.bytedance.pico.openmr"], device, check=False)
        time.sleep(2)
        print("Launching app...")
        run_adb(["shell", "am", "start", "-n", COMPONENT], device)
        print(f"Waiting for {args.duration} seconds...")
        time.sleep(args.duration)
        run_adb(["shell", "am", "force-stop", PACKAGE_NAME], device, check=False)
    finally:
        if logcat_proc.poll() is None:
            logcat_proc.terminate()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_output_dir = Path.cwd() / "tmp_data" / f"pipeline_inspect_outputs_{timestamp}"
    outputs_pulled = pull_outputs(device, device_output_dirs, local_output_dir)

    if outputs_pulled:
        print(f"Outputs saved under {local_output_dir}")
    else:
        print("No output files pulled (none found on device).")

    if args.output and outputs_pulled:
        for expected in args.output:
            output_name = ""
            if Path(expected).is_file():
                output_name = Path(expected).name
            compare_outputs(
                expected=expected,
                output_dir=str(local_output_dir),
                output_name=output_name,
                prefix=OUTPUT_PREFIX,
                int32_names=["predicted_class"],
            )

    turn_screen_off(device)
    print("Test completed.")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run_pipeline_inspect(args)
    except (AdbError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
