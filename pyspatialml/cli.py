# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""pySpatialML command line interface."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

from pyspatialml import __version__
from pyspatialml import compare_cli
from pyspatialml import model_cli
from pyspatialml import operator_cli
from pyspatialml import package_cli
from pyspatialml import pipeline_cli
from pyspatialml import run_cli
from pyspatialml.litert_tools import (
    DEFAULT_LITERT_PACKAGE,
    DEFAULT_LITERT_VERSION,
    LiteRTToolError,
    default_tool_cache,
    install_litert_cli,
    managed_litert_bin,
    print_litert_error,
    resolve_litert_cli,
)


_LITERT_MODEL_COMMANDS = {
    "run": "run",
    "convert": "convert",
    "quantize": "quantize",
    "benchmark": "benchmark",
    "visualize": "visualize",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(
        prog="pyspatialml",
        description="SpatialML package and LiteRT/TFLite workflow tools.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_tools_parser(subparsers)
    _add_compare_parser(subparsers)
    _add_model_parser(subparsers)
    _add_visualize_parser(subparsers)
    _add_operator_parser(subparsers)
    _add_pipeline_parser(subparsers)
    _add_package_parser(subparsers)
    _add_run_parser(subparsers)

    return parser


def _add_tools_parser(subparsers: argparse._SubParsersAction) -> None:
    tools = subparsers.add_parser("tools", help="Manage external tool dependencies.")
    tools_subparsers = tools.add_subparsers(dest="tools_command", required=True)
    litert = tools_subparsers.add_parser("litert", help="Resolve or install LiteRT CLI.")
    litert.add_argument(
        "action",
        choices=["status", "install"],
        nargs="?",
        default="status",
        help="Tool action to run.",
    )
    litert.add_argument("--tool-cache", type=Path, help="Managed tool cache directory.")
    litert.add_argument("--package", default=DEFAULT_LITERT_PACKAGE, help="Pip package to install.")
    litert.add_argument("--version", default=DEFAULT_LITERT_VERSION, help="LiteRT package version.")
    _add_json_output_argument(litert)
    litert.set_defaults(func=_run_litert_tool)


def _add_compare_parser(subparsers: argparse._SubParsersAction) -> None:
    compare = subparsers.add_parser("compare", help="Compare tensor output .npy files or directories.")
    compare.add_argument("expected", type=Path, help="Expected .npy file or directory.")
    compare.add_argument("actual", type=Path, help="Actual .npy file or directory.")
    compare.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance.")
    compare.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance.")
    _add_json_output_argument(compare)
    compare.set_defaults(func=_run_compare)


def _add_model_parser(subparsers: argparse._SubParsersAction) -> None:
    model = subparsers.add_parser("model", help="Delegate model-level work to LiteRT CLI.")
    model_subparsers = model.add_subparsers(dest="model_command", required=True)
    model_info = model_subparsers.add_parser("info", help="Inspect LiteRT/TFLite model inputs and outputs.")
    model_info.add_argument("model", type=Path, help="Path to .tflite model.")
    model_info.add_argument("--signature-index", type=int, default=0, help="Signature index to inspect.")
    _add_json_output_argument(model_info)
    model_info.set_defaults(func=_run_model_info)

    for command_name, litert_command in _LITERT_MODEL_COMMANDS.items():
        command = model_subparsers.add_parser(
            command_name,
            help=f"Run `litert {litert_command}` through pySpatialML tool resolution.",
            add_help=False,
        )
        command.add_argument("--tool-cache", type=Path, help="Managed tool cache directory.")
        _add_json_output_argument(command, help="Print pySpatialML JSON wrapper output.")
        command.add_argument(
            "litert_args",
            nargs=argparse.REMAINDER,
            help=f"Arguments passed to `litert {litert_command}`.",
        )
        command.set_defaults(func=_run_litert_model_command, litert_command=litert_command)


def _add_visualize_parser(subparsers: argparse._SubParsersAction) -> None:
    visualize = subparsers.add_parser("visualize", help="Visualization commands.")
    visualize_subparsers = visualize.add_subparsers(dest="visualize_command", required=True)
    visualize_model = visualize_subparsers.add_parser(
        "model",
        help="Run `litert visualize` through pySpatialML tool resolution.",
        add_help=False,
    )
    visualize_model.add_argument("--tool-cache", type=Path, help="Managed tool cache directory.")
    _add_json_output_argument(visualize_model, help="Print pySpatialML JSON wrapper output.")
    visualize_model.add_argument("litert_args", nargs=argparse.REMAINDER, help="Arguments passed to `litert visualize`.")
    visualize_model.set_defaults(func=_run_litert_model_command, litert_command="visualize")


def _add_operator_parser(subparsers: argparse._SubParsersAction) -> None:
    operator = subparsers.add_parser("operator", help="Discover supported SecureMR operators.")
    operator_subparsers = operator.add_subparsers(dest="operator_command", required=True)
    operator_list = operator_subparsers.add_parser("list", help="List supported operators.")
    _add_json_output_argument(operator_list)
    operator_list.set_defaults(func=_run_operator_list)

    operator_describe = operator_subparsers.add_parser("describe-op", help="Describe one operator.")
    operator_describe.add_argument("name", help="Operator enum name, JSON type name, or creator name.")
    _add_json_output_argument(operator_describe)
    operator_describe.set_defaults(func=_run_operator_describe)


def _add_pipeline_parser(subparsers: argparse._SubParsersAction) -> None:
    pipeline = subparsers.add_parser("pipeline", help="Build and inspect pipeline JSON files.")
    pipeline_subparsers = pipeline.add_subparsers(dest="pipeline_command", required=True)

    pipeline_init = pipeline_subparsers.add_parser("init", help="Create an empty pipeline JSON file.")
    pipeline_init.add_argument("pipeline", type=Path, help="Pipeline JSON path.")
    pipeline_init.add_argument("--force", action="store_true", help="Overwrite an existing pipeline file.")
    _add_json_output_argument(pipeline_init)
    pipeline_init.set_defaults(func=_run_pipeline_init)

    add_tensor = pipeline_subparsers.add_parser("add-tensor", help="Add a tensor descriptor.")
    add_tensor.add_argument("pipeline", type=Path, help="Pipeline JSON path.")
    add_tensor.add_argument("name", help="Tensor name.")
    add_tensor.add_argument("--shape", required=True, help="Tensor shape, for example 128,128,3.")
    add_tensor.add_argument("--dtype", required=True, help="Tensor dtype, for example uint8 or float32.")
    add_tensor.add_argument("--usage", default="matrix", help="Tensor usage: matrix, scalar, point, slice, color, timestamp, gltf.")
    add_tensor.add_argument("--input", action="store_true", help="Also mark this tensor as a pipeline input.")
    add_tensor.add_argument("--output", action="store_true", help="Also mark this tensor as a pipeline output.")
    add_tensor.add_argument("--value", help="Optional comma-separated tensor values.")
    _add_json_output_argument(add_tensor)
    add_tensor.set_defaults(func=_run_pipeline_add_tensor)

    add_op = pipeline_subparsers.add_parser("add-op", help="Append an operator.")
    add_op.add_argument("pipeline", type=Path, help="Pipeline JSON path.")
    add_op.add_argument("op_type", help="Operator type or alias, for example assignment or arithmetic.")
    add_op.add_argument("--input", action="append", default=[], help="Input tensor name. Repeatable.")
    add_op.add_argument("--output", action="append", default=[], help="Output tensor name. Repeatable.")
    add_op.add_argument("--attr", action="append", default=[], help="Raw operator attr. Repeatable.")
    add_op.add_argument("--expression", help="Arithmetic expression for arithmetic operators.")
    add_op.add_argument("--dtype", help="Operator dtype hint when supported.")
    add_op.add_argument("--flag", help="Integer flag for operators such as convert_color.")
    add_op.add_argument("--threshold", type=float, help="Threshold for operators such as nms.")
    add_op.add_argument("--model", help="Package-relative .tflite model path for model inference operators.")
    add_op.add_argument("--model-name", help="Logical model name.")
    add_op.add_argument("--model-target", default="npu", help="Model target, defaults to npu.")
    add_op.add_argument("--cpu-target-num-threads", type=int, default=1, help="CPU thread count for CPU target.")
    _add_json_output_argument(add_op)
    add_op.set_defaults(func=_run_pipeline_add_op)

    set_input = pipeline_subparsers.add_parser("set-input", help="Set top-level pipeline inputs.")
    set_input.add_argument("pipeline", type=Path, help="Pipeline JSON path.")
    set_input.add_argument("names", nargs="+", help="Input tensor names.")
    _add_json_output_argument(set_input)
    set_input.set_defaults(func=_run_pipeline_set_input)

    set_output = pipeline_subparsers.add_parser("set-output", help="Set top-level pipeline outputs.")
    set_output.add_argument("pipeline", type=Path, help="Pipeline JSON path.")
    set_output.add_argument("names", nargs="+", help="Output tensor names.")
    _add_json_output_argument(set_output)
    set_output.set_defaults(func=_run_pipeline_set_output)

    validate = pipeline_subparsers.add_parser("validate", help="Validate a pipeline JSON file.")
    validate.add_argument("pipeline", type=Path, help="Pipeline JSON path.")
    _add_json_output_argument(validate)
    validate.set_defaults(func=_run_pipeline_validate)

    inspect = pipeline_subparsers.add_parser("inspect", help="Print a pipeline summary.")
    inspect.add_argument("pipeline", type=Path, help="Pipeline JSON path.")
    _add_json_output_argument(inspect)
    inspect.set_defaults(func=_run_pipeline_inspect)

    trace = pipeline_subparsers.add_parser("trace", help="Trace a Python function into pipeline JSON.")
    trace.add_argument("source", type=Path, help="Python source file containing a @trace-decorated function.")
    trace.add_argument("--function", required=True, help="Function name to trace.")
    trace.add_argument("--input", action="append", default=[], help="Trace input in name=path format. Repeatable.")
    trace.add_argument("--output", type=Path, required=True, help="Output pipeline JSON path.")
    _add_json_output_argument(trace)
    trace.set_defaults(func=_run_pipeline_trace)


def _add_package_parser(subparsers: argparse._SubParsersAction) -> None:
    package = subparsers.add_parser("package", help="Create, validate, and inspect SpatialML packages.")
    package_subparsers = package.add_subparsers(dest="package_command", required=True)

    package_create = package_subparsers.add_parser("create", help="Create a package directory or zip.")
    package_create.add_argument("--id", required=True, dest="package_id", help="Package id.")
    package_create.add_argument("--pipeline", action="append", default=[], help="Pipeline in id=path format. Repeatable.")
    package_create.add_argument("--output", type=Path, required=True, help="Output package directory or .zip path.")
    package_create.add_argument("--supported-mode", action="append", default=[], help="Supported mode: xr or spatial. Repeatable.")
    package_create.add_argument("--asset-root", type=Path, action="append", default=[], help="Additional root for resolving assets. Repeatable.")
    package_create.add_argument("--zip", action="store_true", help="Also create a zip archive; implied when --output ends with .zip.")
    package_create.add_argument("--force", action="store_true", help="Overwrite existing output without prompting.")
    package_create.add_argument("--yes", action="store_true", help="Answer yes to overwrite prompts.")
    _add_json_output_argument(package_create)
    package_create.set_defaults(func=_run_package_create)

    package_validate = package_subparsers.add_parser("validate", help="Validate a package directory or zip.")
    package_validate.add_argument("package", type=Path, help="Package directory or .zip path.")
    _add_json_output_argument(package_validate)
    package_validate.set_defaults(func=_run_package_validate)

    package_inspect = package_subparsers.add_parser("inspect", help="Print package summary.")
    package_inspect.add_argument("package", type=Path, help="Package directory or .zip path.")
    _add_json_output_argument(package_inspect)
    package_inspect.set_defaults(func=_run_package_inspect)


def _add_run_parser(subparsers: argparse._SubParsersAction) -> None:
    run = subparsers.add_parser("run", help="Run pipelines and packages.")
    run_subparsers = run.add_subparsers(dest="run_command", required=True)
    run_host = run_subparsers.add_parser("host", help="Run a pipeline or package on the host Python executor.")
    run_host.add_argument("target", type=Path, help="Pipeline JSON, package directory, or package zip.")
    run_host.add_argument("--pipeline", action="append", default=[], help="Pipeline id to run from a package. Repeatable.")
    run_host.add_argument("--input", action="append", default=[], help="Input tensor in name=path format. Repeatable.")
    run_host.add_argument("--dump", action="append", default=[], help="Tensor name to dump, or 'all'. Repeatable.")
    run_host.add_argument("--output-dir", type=Path, help="Directory for .npy output tensors.")
    _add_json_output_argument(run_host)
    run_host.set_defaults(func=_run_host)

    run_device = run_subparsers.add_parser("device", help="Run a package on a device through the XR runner APK.")
    run_device.add_argument("target", type=Path, help="Package directory or package zip.")
    run_device.add_argument(
        "--input",
        action="append",
        default=[],
        help="Image/raw input path or tensor=path binding. Repeatable.",
    )
    run_device.add_argument("--pipeline", action="append", default=[], help="Pipeline id to run from a package. Repeatable.")
    run_device.add_argument("--output-dir", type=Path, help="Local directory for pulled device outputs.")
    run_device.add_argument("--dump", action="append", default=[], help="Use 'all' to dump every readable numeric device tensor.")
    run_device.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="Max seconds to wait for one-shot output; fixed run time for loop/keep-running.",
    )
    run_device.add_argument("--loop", action="store_true", help="Run the pipeline repeatedly on device.")
    run_device.add_argument("--keep-running", action="store_true", help="Leave the runner app running after pulling outputs.")
    run_device.add_argument("--use-vst", action="store_true", help="Use device VST camera input instead of --input.")
    run_device.add_argument(
        "--backend",
        choices=["npu", "gpu", "cpu"],
        help="Override model backend for device run without changing the package.",
    )
    run_device.add_argument("--interval-ms", type=int, default=50, help="Loop interval in milliseconds.")
    run_device.add_argument("--apk", type=Path, help="Runner APK path.")
    run_device.add_argument("--device", help="ADB device serial.")
    _add_json_output_argument(run_device)
    run_device.set_defaults(func=_run_device)


def _add_json_output_argument(parser: argparse.ArgumentParser, *, help: str = "Print JSON output.") -> None:
    parser.add_argument("--json", action="store_true", help=help)
    parser.add_argument(
        "--format",
        choices=["json"],
        dest="output_format",
        help="Output format. Currently only 'json' is supported.",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run pySpatialML CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "output_format", None) == "json":
        args.json = True
    try:
        return args.func(args)
    except compare_cli.CompareCliError as exc:
        if _json_requested(args):
            _print_error_json("PSM_COMPARE", "compare", str(exc), exit_code=1)
            return 1
        compare_cli.print_compare_error(exc)
        return 1
    except LiteRTToolError as exc:
        if _json_requested(args):
            _print_error_json("PSM_LITERT_UNAVAILABLE", "litert_tool", str(exc), exit_code=2)
            return 2
        print_litert_error(exc)
        return 2
    except model_cli.ModelCliError as exc:
        if _json_requested(args):
            _print_error_json("PSM_MODEL", "model", str(exc), exit_code=1)
            return 1
        model_cli.print_model_error(exc)
        return 1
    except operator_cli.OperatorCliError as exc:
        if _json_requested(args):
            _print_error_json("PSM_OPERATOR", "operator", str(exc), exit_code=1)
            return 1
        operator_cli.print_operator_error(exc)
        return 1
    except package_cli.PackageCliError as exc:
        if _json_requested(args):
            _print_error_json("PSM_PACKAGE", "package", str(exc), exit_code=1)
            return 1
        package_cli.print_package_error(exc)
        return 1
    except pipeline_cli.PipelineCliError as exc:
        if _json_requested(args):
            _print_error_json("PSM_PIPELINE", "pipeline", str(exc), exit_code=1)
            return 1
        pipeline_cli.print_pipeline_error(exc)
        return 1
    except run_cli.RunCliError as exc:
        if _json_requested(args):
            _print_error_json("PSM_RUN", "run", str(exc), exit_code=1)
            return 1
        run_cli.print_run_error(exc)
        return 1
    except subprocess.CalledProcessError as exc:
        if _json_requested(args):
            _print_error_json(
                "PSM_SUBPROCESS",
                "subprocess",
                str(exc),
                exit_code=exc.returncode or 3,
                stdout=exc.stdout,
                stderr=exc.stderr,
            )
            return exc.returncode or 3
        if exc.stdout:
            print(exc.stdout, end="")
        if exc.stderr:
            print(exc.stderr, end="", file=sys.stderr)
        return exc.returncode or 3


def _run_litert_tool(args: argparse.Namespace) -> int:
    if args.action == "install":
        cli = install_litert_cli(
            cache_dir=args.tool_cache,
            package=args.package,
            version=args.version,
        )
        if args.json:
            _print_json(
                {
                    "ok": True,
                    "command": "tools litert install",
                    "path": str(cli.path),
                    "managed": cli.managed,
                    "package": args.package,
                    "version": args.version,
                }
            )
            return 0
        print(f"LiteRT CLI installed: {cli.path}")
        return 0

    cli = resolve_litert_cli(cache_dir=args.tool_cache)
    source = "managed" if cli.managed else "system"
    version = cli.version()
    if args.json:
        _print_json(
            {
                "ok": True,
                "command": "tools litert status",
                "path": str(cli.path),
                "source": source,
                "managed": cli.managed,
                "version": version,
                "managed_cache": str(args.tool_cache or default_tool_cache()),
                "managed_path": str(managed_litert_bin(args.tool_cache)),
            }
        )
        return 0
    print(f"LiteRT CLI: {cli.path}")
    print(f"Source: {source}")
    print(f"Version: {version}")
    print(f"Managed cache: {args.tool_cache or default_tool_cache()}")
    print(f"Managed path: {managed_litert_bin(args.tool_cache)}")
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    return compare_cli.compare_paths(
        args.expected,
        args.actual,
        rtol=args.rtol,
        atol=args.atol,
        as_json=args.json,
    )


def _run_model_info(args: argparse.Namespace) -> int:
    return model_cli.model_info(
        args.model,
        signature_index=args.signature_index,
        as_json=args.json,
    )


def _run_operator_list(args: argparse.Namespace) -> int:
    return operator_cli.list_operators(as_json=args.json)


def _run_operator_describe(args: argparse.Namespace) -> int:
    return operator_cli.describe_operator(args.name, as_json=args.json)


def _run_litert_model_command(args: argparse.Namespace) -> int:
    cli = resolve_litert_cli(ensure=True, cache_dir=args.tool_cache)
    litert_args = _strip_remainder_separator(args.litert_args)
    env = dict(os.environ)
    env.setdefault("UV_SYSTEM_CERTS", "true")
    argv = [str(cli.path), args.litert_command, *litert_args]
    if _json_requested(args):
        result = subprocess.run(argv, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _print_json(
            {
                "ok": result.returncode == 0,
                "command": f"model {args.litert_command}",
                "litert": str(cli.path),
                "argv": argv,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        return result.returncode
    result = subprocess.run(argv, env=env)
    return result.returncode


def _strip_remainder_separator(args: Sequence[str]) -> list[str]:
    values = list(args)
    if values and values[0] == "--":
        return values[1:]
    return values


def _run_pipeline_init(args: argparse.Namespace) -> int:
    return _json_action(
        args,
        "pipeline.init",
        lambda: pipeline_cli.init_pipeline(args.pipeline, force=args.force),
        {"pipeline": str(args.pipeline), "force": args.force},
    )


def _run_pipeline_add_tensor(args: argparse.Namespace) -> int:
    return _json_action(
        args,
        "pipeline.add_tensor",
        lambda: pipeline_cli.add_tensor(
            args.pipeline,
            args.name,
            shape=args.shape,
            dtype=args.dtype,
            usage=args.usage,
            is_input=args.input,
            is_output=args.output,
            value=args.value,
        ),
        {
            "pipeline": str(args.pipeline),
            "tensor": args.name,
            "shape": args.shape,
            "dtype": args.dtype,
            "usage": args.usage,
            "input": args.input,
            "output": args.output,
        },
    )


def _run_pipeline_add_op(args: argparse.Namespace) -> int:
    return _json_action(
        args,
        "pipeline.add_op",
        lambda: pipeline_cli.add_op(
            args.pipeline,
            args.op_type,
            inputs=args.input,
            outputs=args.output,
            attrs=args.attr,
            expression=args.expression,
            dtype=args.dtype,
            flag=args.flag,
            threshold=args.threshold,
            model=args.model,
            model_name=args.model_name,
            model_target=args.model_target,
            cpu_target_num_threads=args.cpu_target_num_threads,
        ),
        {
            "pipeline": str(args.pipeline),
            "op_type": args.op_type,
            "inputs": args.input,
            "outputs": args.output,
        },
    )


def _run_pipeline_set_input(args: argparse.Namespace) -> int:
    return _json_action(
        args,
        "pipeline.set_input",
        lambda: pipeline_cli.set_input(args.pipeline, args.names),
        {"pipeline": str(args.pipeline), "inputs": list(args.names)},
    )


def _run_pipeline_set_output(args: argparse.Namespace) -> int:
    return _json_action(
        args,
        "pipeline.set_output",
        lambda: pipeline_cli.set_output(args.pipeline, args.names),
        {"pipeline": str(args.pipeline), "outputs": list(args.names)},
    )


def _run_pipeline_validate(args: argparse.Namespace) -> int:
    return _json_action(
        args,
        "pipeline.validate",
        lambda: pipeline_cli.validate_pipeline(args.pipeline),
        {"pipeline": str(args.pipeline), "valid": True},
    )


def _run_pipeline_inspect(args: argparse.Namespace) -> int:
    if args.json:
        spec = _read_json(args.pipeline)
        operators = spec.get("operators", [])
        _print_json(
            {
                "ok": True,
                "command": "pipeline.inspect",
                "pipeline": str(args.pipeline),
                "tensors": len(spec.get("tensors", {})),
                "operators": len(operators),
                "inputs": spec.get("inputs", []),
                "outputs": spec.get("outputs", []),
                "operator_summaries": [
                    {
                        "index": index,
                        "type": op.get("type", "<unknown>") if isinstance(op, dict) else "<invalid>",
                        "inputs": op.get("inputs", []) if isinstance(op, dict) else [],
                        "outputs": op.get("outputs", []) if isinstance(op, dict) else [],
                    }
                    for index, op in enumerate(operators)
                ],
            }
        )
        return 0
    return pipeline_cli.inspect_pipeline(args.pipeline)


def _run_pipeline_trace(args: argparse.Namespace) -> int:
    return _json_action(
        args,
        "pipeline.trace",
        lambda: pipeline_cli.trace_pipeline(
            args.source,
            function_name=args.function,
            output=args.output,
            inputs=args.input,
        ),
        {"source": str(args.source), "function": args.function, "output": str(args.output)},
    )


def _run_package_create(args: argparse.Namespace) -> int:
    return _json_action(
        args,
        "package.create",
        lambda: package_cli.create_package(
            package_id=args.package_id,
            pipelines=args.pipeline,
            output=args.output,
            supported_modes=args.supported_mode,
            asset_roots=args.asset_root,
            force=args.force,
            zip_output=args.zip,
            assume_yes=args.yes,
        ),
        {
            "package_id": args.package_id,
            "pipelines": list(args.pipeline),
            "output": str(args.output),
            "zip": args.zip or args.output.suffix == ".zip",
        },
    )


def _run_package_validate(args: argparse.Namespace) -> int:
    return _json_action(
        args,
        "package.validate",
        lambda: package_cli.validate_package(args.package),
        {"package": str(args.package), "valid": True},
    )


def _run_package_inspect(args: argparse.Namespace) -> int:
    if args.json:
        root = package_cli._materialize_package(args.package)
        manifest = package_cli._load_validated_manifest(root)
        _print_json(
            {
                "ok": True,
                "command": "package.inspect",
                "package": str(args.package),
                "root": str(root),
                "manifest": manifest,
                "assets": package_cli._package_assets(root),
            }
        )
        return 0
    return package_cli.inspect_package(args.package)


def _run_host(args: argparse.Namespace) -> int:
    if args.json:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = run_cli.run_host(
                args.target,
                pipeline_ids=args.pipeline,
                inputs=args.input,
                output_dir=args.output_dir,
                dumps=args.dump,
            )
        _print_json(
            {
                "ok": code == 0,
                "command": "run.host",
                "target": str(args.target),
                "pipelines": list(args.pipeline),
                "output_dir": str(args.output_dir) if args.output_dir else None,
                "dumps": list(args.dump),
                "stdout": stdout.getvalue(),
            }
        )
        return code
    return run_cli.run_host(
        args.target,
        pipeline_ids=args.pipeline,
        inputs=args.input,
        output_dir=args.output_dir,
        dumps=args.dump,
    )


def _run_device(args: argparse.Namespace) -> int:
    return run_cli.run_device(
        args.target,
        inputs=args.input,
        pipeline_ids=args.pipeline,
        output_dir=args.output_dir,
        dumps=args.dump,
        duration=args.duration,
        loop=args.loop,
        keep_running=args.keep_running,
        use_vst=args.use_vst,
        backend=args.backend,
        interval_ms=args.interval_ms,
        apk=args.apk,
        device=args.device,
        as_json=args.json,
    )


def _json_requested(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "json", False) or getattr(args, "output_format", None) == "json")


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _print_error_json(
    code: str,
    category: str,
    message: str,
    *,
    exit_code: int,
    stdout: object = None,
    stderr: object = None,
) -> None:
    payload = {
        "ok": False,
        "error": {
            "code": code,
            "category": category,
            "message": message,
            "exit_code": exit_code,
        },
    }
    if stdout:
        payload["stdout"] = stdout
    if stderr:
        payload["stderr"] = stderr
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str), file=sys.stderr)


def _json_action(args: argparse.Namespace, command: str, func, payload: dict) -> int:
    if not _json_requested(args):
        return func()
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        code = func()
    result = {"ok": code == 0, "command": command, **payload}
    captured = stdout.getvalue()
    if captured:
        result["stdout"] = captured
    _print_json(result)
    return code


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise pipeline_cli.PipelineCliError(f"JSON file must contain an object: {path}")
    return payload


if __name__ == "__main__":
    sys.exit(main())
