#!/usr/bin/env python3
import argparse
import glob
import os
import struct
import sys
from typing import Iterable, List, Optional


def load_expected(src: str, as_int: bool) -> List[float]:
    if os.path.isfile(src):
        data = open(src, "rb").read()
        if len(data) % 4 != 0:
            print(
                f"Expected file {src} size {len(data)} is not divisible by 4 bytes (float32).",
                flush=True,
            )
        count = len(data) // 4
        fmt = "<%di" % count if as_int else "<%df" % count
        return list(struct.unpack(fmt, data[: count * 4]))
    if as_int:
        return [int(float(x)) for x in src.split(",") if x.strip()]
    return [float(x) for x in src.split(",") if x.strip()]


def should_use_int32(output_name: str, int32_names: Iterable[str]) -> bool:
    if not output_name:
        return False
    for name in int32_names:
        if name and name in output_name:
            return True
    return False


def compare_outputs(
    expected: str,
    output_dir: str,
    output_name: str,
    prefix: str,
    int32_names: Optional[Iterable[str]] = None,
) -> int:
    int32_names = list(int32_names or ["predicted_class"])
    selected_name = output_name
    if selected_name and not selected_name.startswith(prefix):
        selected_name = prefix + selected_name

    files = sorted(glob.glob(os.path.join(output_dir, f"{prefix}*.bin")))
    if not files:
        print("No output files available for comparison.", flush=True)
        return 0

    if len(files) > 1:
        if not selected_name:
            print("--output-name is not given, but found multiple outputs:")
            for file_path in files:
                print(f"  {os.path.basename(file_path)}")
            return 0
        first = os.path.join(output_dir, selected_name)
        if not os.path.exists(first):
            print(f"{selected_name} not found in {output_dir}", flush=True)
            return 1
    else:
        first = files[0]

    int_mode = should_use_int32(os.path.basename(first), int32_names)
    expected_values = load_expected(expected, int_mode)
    if not expected_values:
        print("No expected values to compare; skipping.", flush=True)
        return 0

    data = open(first, "rb").read()
    need = len(expected_values)
    actual: List[float] = []
    for i in range(need):
        offset = i * 4
        if offset + 4 > len(data):
            break
        if int_mode:
            actual.append(struct.unpack_from("<i", data, offset)[0])
        else:
            actual.append(struct.unpack_from("<f", data, offset)[0])

    print(
        f"Comparing first {len(actual)} values from {os.path.basename(first)} "
        f"against expected ({need} values requested):"
    )
    compared_result_txt = os.path.join(output_dir, "output_diff.txt")
    with open(compared_result_txt, "w") as file_handle:
        for idx, (exp, act) in enumerate(zip(expected_values, actual)):
            diff = abs(exp - act)
            rel = diff / (abs(exp) + 1e-9)
            status = "OK" if diff <= 1e-3 or rel <= 1e-3 else "DIFF"
            ret = (
                f"  idx {idx}: expected={exp:.6g}, actual={act:.6g}, "
                f"abs_diff={diff:.3g}, rel_diff={rel:.3g} -> {status}"
            )
            file_handle.write(ret + "\n")
            if idx < 10:
                print(ret)
            if idx == 10:
                print("  ...")
    print(f"Diff results saved in {compared_result_txt}")

    if len(actual) < need:
        print(
            f"Warning: output had only {len(actual)} values, expected {need}.",
            flush=True,
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare output buffers against expected float32 or int32 values."
    )
    parser.add_argument("expected", help="Comma-separated values or a binary file")
    parser.add_argument("output_dir", help="Directory containing output binaries")
    parser.add_argument("output_name", nargs="?", default="", help="Specific output file name to compare")
    parser.add_argument(
        "--prefix",
        default="model_inspect_output_",
        help="Output filename prefix (default: model_inspect_output_)",
    )
    parser.add_argument(
        "--int32-name",
        action="append",
        default=["predicted_class"],
        help="Substring that marks outputs to be parsed as int32 (repeatable)",
    )
    args = parser.parse_args()
    return compare_outputs(
        expected=args.expected,
        output_dir=args.output_dir,
        output_name=args.output_name,
        prefix=args.prefix,
        int32_names=args.int32_name,
    )


if __name__ == "__main__":
    sys.exit(main())
