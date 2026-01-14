import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

from ppadb.client import Client as AdbClient


DEFAULT_DEVICE_ID = os.getenv("ANDROID_DEVICE_ID", "PA9210BGJ3121331D")


class AdbError(RuntimeError):
    pass


@dataclass(frozen=True)
class DeviceContext:
    device_id: str

    def adb_prefix(self) -> List[str]:
        return ["adb", "-s", self.device_id]


def _parse_adb_devices(output: str) -> List[str]:
    devices: List[str] = []
    for line in output.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            devices.append(parts[0])
    return devices


def get_connected_devices() -> List[str]:
    result = subprocess.run(["adb", "devices"], check=True, capture_output=True, text=True)
    return _parse_adb_devices(result.stdout)


def select_device(device_id: Optional[str]) -> DeviceContext:
    devices = get_connected_devices()
    if not devices:
        raise AdbError("No adb devices found. Please connect a device and retry.")
    if device_id:
        if device_id in devices:
            return DeviceContext(device_id)
        raise AdbError(f"Requested device {device_id} not found. Connected: {', '.join(devices)}")
    if len(devices) == 1:
        return DeviceContext(devices[0])
    if DEFAULT_DEVICE_ID in devices:
        return DeviceContext(DEFAULT_DEVICE_ID)
    raise AdbError(
        "Multiple devices connected; specify --device or set ANDROID_DEVICE_ID env. Connected: " + ", ".join(devices)
    )


def run_adb(args: Iterable[str], device: DeviceContext, check: bool = True) -> subprocess.CompletedProcess:
    cmd = device.adb_prefix() + list(args)
    return subprocess.run(cmd, check=check)


def capture_adb(args: Iterable[str], device: DeviceContext, check: bool = True) -> str:
    cmd = device.adb_prefix() + list(args)
    result = subprocess.run(cmd, check=check, capture_output=True, text=True)
    return result.stdout


def capture_adb_binary(args: Iterable[str], device: DeviceContext, check: bool = True) -> bytes:
    cmd = device.adb_prefix() + list(args)
    result = subprocess.run(cmd, check=check, capture_output=True)
    return result.stdout


def screen_is_on(device: DeviceContext) -> bool:
    output = capture_adb(["shell", "dumpsys", "power"], device, check=False)
    for line in output.splitlines():
        if "Display Power" in line or "mWakefulness" in line:
            if "ON" in line or "Awake" in line:
                return True
    return False


def ensure_screen_on(device: DeviceContext) -> None:
    if screen_is_on(device):
        return
    run_adb(["shell", "input", "keyevent", "26"], device, check=False)
    time.sleep(3)


def turn_screen_off(device: DeviceContext) -> None:
    if screen_is_on(device):
        run_adb(["shell", "input", "keyevent", "26"], device, check=False)


def start_logcat(package: str, regex: str, device: DeviceContext) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "securemr.inspect.logcat",
        "-p",
        package,
        "-e",
        regex,
        "--device",
        device.device_id,
    ]
    return subprocess.Popen(cmd)


def install_apk(device: DeviceContext, apk_path: str, package_name: str, force: bool = False) -> None:
    adb_host = os.getenv("ADB_HOST", "host.docker.internal" if os.path.exists("/.dockerenv") else "127.0.0.1")
    adb_port = int(os.getenv("ADB_PORT", "5037"))
    client = AdbClient(host=adb_host, port=adb_port)
    devices = client.devices()
    target = next((item for item in devices if item.serial == device.device_id), None)
    if not target:
        raise AdbError(
            f"Failed to locate device {device.device_id} via ppadb. Connected: "
            + ", ".join([item.serial for item in devices])
        )
    try:
        listed = target.shell("pm list packages")
    except Exception as exc:  # noqa: BLE001
        raise AdbError(f"Failed to query installed packages: {exc}") from exc

    if not force and f"package:{package_name}" in listed:
        print("APK exists, skip installation.")
        return

    try:
        target.install(apk_path, reinstall=True)
        return
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        if "INSTALL_FAILED_UPDATE_INCOMPATIBLE" in message:
            target.uninstall(package_name)
            target.install(apk_path, reinstall=True)
            return
        raise AdbError(f"Failed to install {apk_path}: {message}") from exc
