#!/usr/bin/env python

import json
import os
import subprocess as commands
from pathlib import Path
from typing import List, Optional

import typer

# Config Description:
#
# { "Name":"resnet18", # 任务名
#     "HostRootPath": "resnet18", # 任务路径，就是benchmarks文件下刚刚建好的那个
#     "HostResultsDir":"resnet18/results", # 测试结果存放路径
#     "DevicePath":"/data/local/tmp/snpebm", # 端侧路径
#     "Devices":["PA7H10MGF9180889W"], # 设备编号，可以用 adb devices 查看
#     "HostName": "localhost", # 不用改
#     "Runs":10, # 测试次数，次数越多越准
#
#     "Model": {
#         "Name": "resnet18", # 模型名字
#         "Dlc": "../resnet18/model/resnet18_quan.dlc", # PC端模型的位置
#         "InputList": "./benchmark_image_lst.txt", # 测试用的输入list在PC本机的位置（注意list里面的路径是输入存在端侧的路径，不是本地PC路径）
#         "Data": [
#             "./resnet18/data" # 测试数据在本机的位置。这些数据都会拷贝到端侧的/DevicePath/resnet18/data里面，前面的InputList指定的就是这个路径
#         ]
#     },
#     "Runtimes":["DSP"], # 模型在哪跑，可以指定多个位置，都会测出结果
#     "Measurements": ["timing"] # 测啥。timing是时间，memory是内存，可以指定多个，都会测出结果
# }


def get_device_id_via_adb() -> str:
    """Get device id using adb."""
    cmd = "adb devices"
    (status, output) = commands.getstatusoutput(cmd)
    output = output.split("\n")
    try:
        device_id = output[1].split("\t")[0]
    except IndexError as e:
        print("Please connect a pico device to host machine.")
        raise e
    return device_id


def get_data_path_from_input_list(input_list_path: str) -> str:
    """Get data path from input list file."""
    for line in open(input_list_path, "r"):
        data_path = os.path.dirname(line.strip().split(" ")[0])
        break
    return data_path


def update_input_list_root(
    input_list_path: str,
    device_path: str,
    model_name: str,
) -> str:
    """Update input data path root from host to device."""
    update_list_path = str(input_list_path) + ".device"
    update_fid = open(update_list_path, "w")
    target_root = os.path.join(device_path, model_name)
    with open(input_list_path, "r") as fid:
        for line in fid:
            # tmp_output/fake_raw/000099.raw =>
            # /data/local/tmp/qnn_benchmark/default_model_name/fake_raw/xxx.raw
            new_line = ""
            for filepath in line.split(" "):
                items = filepath.split("/")
                new_filepath = os.path.join(target_root, "/".join(items[-2:]))
                if new_line:
                    new_line += f" {new_filepath}"
                else:
                    new_line = new_filepath
            update_fid.write(f"{new_line}")
    update_fid.close()
    return update_list_path


def generate_benchmark_config(
    model_path: Path,
    input_list_path: Path,
    output_json: Path,
    task_name: str = "default_task_name",
    model_name: str = "default_model_name",
    root_path: Optional[str] = None,
    output: Optional[str] = None,
    device_path: str = "/data/local/tmp/qnn_benchmark",
    device_id: Optional[str] = None,
    runs: int = 100,
    runtimes: List[str] = None,
    measurements: List[str] = None,
    version: str = "qnn",
    cache: bool = False,
):
    """Generate snpe benchmark config json file."""
    if runtimes is None:
        runtimes = ["DSP"]
    if measurements is None:
        measurements = ["timing"]
    
    if not root_path:
        root_path = os.path.dirname(model_path)
    if not output:
        output = os.path.join(root_path, "benchmark_output")
    if not device_id:
        device_id = get_device_id_via_adb()

    config = dict()
    config["Name"] = task_name
    config["HostRootPath"] = root_path
    config["HostResultsDir"] = output
    config["DevicePath"] = device_path
    config["Devices"] = [device_id]
    config["HostName"] = "localhost"
    config["Runs"] = runs
    config["Measurements"] = measurements
    model_config = dict()
    model_config["Name"] = model_name

    if version == "snpe":
        config["Runtimes"] = runtimes
        model_config["Dlc"] = str(model_path)
    elif version == "qnn":
        config["Backends"] = runtimes
        model_config["qnn_model"] = str(model_path)

    model_config["Data"] = [get_data_path_from_input_list(input_list_path)]
    model_config["InputList"] = update_input_list_root(input_list_path, device_path, model_name)
    config["Model"] = model_config

    with open(output_json, "w") as fid:
        json.dump(config, fid, indent=2)
    
    if not cache:
        # remove target path
        target_path = os.path.join(device_path, model_name)
        os.system(f"adb shell rm -r {target_path}")
    
    return output_json


def cli_main(
    model_path: Path,
    input_list_path: Path,
    output_json: Path,
    task_name: Optional[str] = typer.Option("default_task_name", help="task name"),
    model_name: Optional[str] = typer.Option("default_model_name", help="model name"),
    root_path: Optional[str] = typer.Option(None, help="root path"),
    output: Optional[str] = typer.Option("benchmark_result", help="output name"),
    device_path: Optional[str] = typer.Option("/data/local/tmp/qnn_benchmark", help="device path"),
    device_id: Optional[str] = typer.Option(None, help="device id"),
    runs: Optional[int] = typer.Option(100, help="test runs"),
    runtimes: List[str] = typer.Option(["DSP"], help="runtimes"),
    measurements: List[str] = typer.Option(["timing"], help="metric"),
    version: Optional[str] = typer.Option("qnn", help="version"),
    cache: Optional[bool] = typer.Option(False, help="keep cache"),
):
    """Generate snpe benchmark config json file."""
    generate_benchmark_config(
        model_path=model_path,
        input_list_path=input_list_path,
        output_json=output_json,
        task_name=task_name,
        model_name=model_name,
        root_path=root_path,
        output=output,
        device_path=device_path,
        device_id=device_id,
        runs=runs,
        runtimes=runtimes,
        measurements=measurements,
        version=version,
        cache=cache,
    )
    typer.echo(typer.style(f"save to {output_json}", fg=typer.colors.GREEN))
    print("Done.")
    


if __name__ == "__main__":
    typer.run(cli_main)
