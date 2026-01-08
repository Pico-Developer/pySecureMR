import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

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


PACKAGE_NAME = "com.bytedance.pico.secure_mr_demo.model_inspect"
COMPONENT = f"{PACKAGE_NAME}/android.app.NativeActivity"
OUTPUT_PREFIX = "model_inspect_output_"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SecureMR model_inspect on an Android device."
    )
    parser.add_argument("--input", help="Input tensor file (NHWC float32).")
    parser.add_argument("--duration", type=int, default=20, help="Run time in seconds")
    parser.add_argument(
        "--output",
        help="Expected output: comma-separated float32 values or a float32 binary file",
    )
    parser.add_argument(
        "--output-name",
        default="",
        help="Model output file name to compare with --output",
    )
    parser.add_argument("--model", required=True, help="Serialized model .bin")
    parser.add_argument("--json", required=True, help="Model spec .json")
    parser.add_argument("--device", help="ADB device id (optional)")
    parser.add_argument(
        "--apk",
        default="",
        help="Path to model_inspect APK (optional)",
    )
    return parser.parse_args(argv)


def resolve_apk_path(user_path: str) -> Path:
    if user_path:
        return Path(user_path)
    return Path(__file__).resolve().parent / "apks" / "model_inspect-debug.apk"


def prepare_device(device: DeviceContext, apk_path: Path) -> None:
    if not apk_path.exists():
        raise FileNotFoundError(f"APK not found: {apk_path}")
    print("Installing model_inspect APK...")
    install_apk(device, str(apk_path), PACKAGE_NAME)


def clean_device_dirs(device: DeviceContext, tmp_dir: str, output_dirs: list[str]) -> None:
    run_adb(["shell", "rm", "-rf", tmp_dir], device, check=False)
    run_adb(["shell", "mkdir", "-p", tmp_dir], device)
    for output_dir in output_dirs:
        run_adb(["shell", "rm", "-f", f"{output_dir}/*"], device, check=False)


def pull_outputs(device: DeviceContext, output_dirs: list[str], local_dir: Path) -> bool:
    local_dir.mkdir(parents=True, exist_ok=True)
    outputs_pulled = False
    for directory in output_dirs:
        listing = capture_adb(["shell", "ls", directory], device, check=False)
        for filename in listing.splitlines():
            filename = filename.strip().replace("\r", "")
            if not filename or not filename.startswith(OUTPUT_PREFIX):
                continue
            if not filename.endswith(".bin"):
                continue
            device_path = f"{directory}/{filename}"
            data = capture_adb_binary(["shell", "cat", device_path], device, check=False)
            if not data:
                continue
            with open(local_dir / filename, "wb") as file_handle:
                file_handle.write(data)
            outputs_pulled = True
    return outputs_pulled


def run_model_inspect(args: argparse.Namespace) -> int:
    model_file = Path(args.model)
    json_file = Path(args.json)
    if not model_file.is_file():
        raise FileNotFoundError(f"Bin file not found: {model_file}")
    if not json_file.is_file():
        raise FileNotFoundError(f"Json file not found: {json_file}")
    if args.input and not Path(args.input).is_file():
        raise FileNotFoundError(f"Input tensor file not found: {args.input}")

    device = select_device(args.device)
    apk_path = resolve_apk_path(args.apk)
    prepare_device(device, apk_path)

    device_tmp_dir = f"/sdcard/Android/data/{PACKAGE_NAME}/files"
    device_input_path = f"{device_tmp_dir}/input.bin"
    device_output_dirs = [
        f"{device_tmp_dir}/model_inspect",
        "/data/local/tmp/securemr_model_inspect",
    ]

    clean_device_dirs(device, device_tmp_dir, device_output_dirs)

    print("Pushing model files to device...")
    run_adb(["push", str(model_file), f"{device_tmp_dir}/model.serialized.bin"], device)
    run_adb(["push", str(json_file), f"{device_tmp_dir}/model.serialized.json"], device)

    if args.input:
        print(f"Pushing input tensor to {device_input_path}...")
        run_adb(["push", args.input, device_input_path], device)
        run_adb(["shell", "setprop", "debug.securemr.model_inspect.input", device_input_path], device)
    else:
        run_adb(["shell", "setprop", "debug.securemr.model_inspect.input", "''"], device)

    run_adb(
        ["shell", "setprop", "debug.securemr.model_inspect.model_dir", device_tmp_dir],
        device,
    )

    ensure_screen_on(device)

    existing_pid = capture_adb(["shell", "pidof", PACKAGE_NAME], device, check=False).strip()
    if existing_pid:
        run_adb(["shell", "am", "force-stop", PACKAGE_NAME], device, check=False)
        time.sleep(3)

    logcat_proc = start_logcat(PACKAGE_NAME, "ModelInspect", device)
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
    local_output_dir = Path.cwd() / "tmp_data" / f"model_inspect_outputs_{timestamp}"
    outputs_pulled = pull_outputs(device, device_output_dirs, local_output_dir)

    if outputs_pulled:
        print(f"Outputs saved under {local_output_dir}")
    else:
        print("No output files pulled (none found on device).")

    if args.output and outputs_pulled:
        compare_outputs(
            expected=args.output,
            output_dir=str(local_output_dir),
            output_name=args.output_name,
            prefix=OUTPUT_PREFIX,
        )

    turn_screen_off(device)
    print("Test completed.")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run_model_inspect(args)
    except (AdbError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
