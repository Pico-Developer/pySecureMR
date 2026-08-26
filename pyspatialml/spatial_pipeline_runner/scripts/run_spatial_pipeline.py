#!/usr/bin/env python3
"""Push a SpatialML package to the Spatial runner APK, run it, and pull outputs."""

from __future__ import annotations

import sys

from pyspatialml.device_runner_base import SPATIAL_CONFIG, main as run_device_runner


def main(argv: list[str] | None = None) -> int:
    return run_device_runner(SPATIAL_CONFIG, argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
