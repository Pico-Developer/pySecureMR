# pySecureMR

<div align="center">

[![PyPI][pypi-badge]][pypi-url]
[![MIT licensed][mit-badge]][mit-url]
[![Python Version][python-badge]][python-url]

</div>


<p align="center">
  <a  alt="python version">
      <img src="https://img.shields.io/badge/python-3.10-blue?logo=python" /></a>
  <a> <img src="https://img.shields.io/badge/secure-mr-green" /></a>
  <a> <img src="https://img.shields.io/badge/os-linux-yellow" /></a>
  <a> <img src="https://img.shields.io/badge/os-windows(wsl2)-yellow" /></a>
  <img alt="PyPI - Wheel" src="https://img.shields.io/pypi/wheel/securemr">
</p>

Python bindings for [SecureMR](https://path-to-SecureMR-link) project.

## Table of Contents

  * [Why pySecureMR](#why-pysecuremr)
  * [Supported platform](#supported-platform)
  * [Install](#install)
     * [Pip](#pip)
     * [Manual install](#manual-install)
  * [Run test](#run-test)
  * [Supported operators](#supported-operators)

## Why pySecureMR?

When developing a SecureMR app, it's not very easy to debug pipeline.
You are not allowed to access each operator output directly. `pySecureMR` happens here
to rescue you from complicated and painful debugging time. We bind [most of SecureMR
operators to python](#supported-operators) so you can call each operator and check input and output.

## Supported platform
- Linux (ubuntu22): YES
- Windows (wsl2, ubuntu22): YES
- Mac: NO

## Install

### Manual install
```bash
git clone https://github.com/Pico-Developer/pySecureMR
cd pySecureMR
pip3 install -e "."
```
Check installation:
```bash
python3 -c "import securemr"
```

### Pip

```bash
pip3 install securemr
```

## Run test

```bash
pytest
```
Refer to [test code](./tests) to learn more about the usage.

## Supported operators

| ID  | Name                         | Pybind |
|-----|------------------------------|--------|
| 1   | ARITHMETIC_COMPOSE           | ✅     |
| 4   | ELEMENTWISE_MIN              | ✅     |
| 5   | ELEMENTWISE_MAX              | ✅     |
| 6   | ELEMENTWISE_MULTIPLY         | ✅     |
| 7   | CUSTOMIZED_COMPARE           | ✅     |
| 8   | ELEMENTWISE_OR               | ✅     |
| 9   | ELEMENTWISE_AND              | ✅     |
| 10  | ALL                          | ✅     |
| 11  | ANY                          | ✅     |
| 12  | NMS                          | ✅     |
| 13  | SOLVE_P_N_P                  | ✅     |
| 14  | GET_AFFINE                   | ✅     |
| 15  | APPLY_AFFINE                 | ✅     |
| 16  | APPLY_AFFINE_POINT           | ✅     |
| 17  | UV_TO_3D_IN_CAM_SPACE        | ❌     |
| 18  | ASSIGNMENT                   | ✅     |
| 19  | RUN_MODEL_INFERENCE          | ❌     |
| 21  | NORMALIZE                    | ✅     |
| 22  | CAMERA_SPACE_TO_WORLD        | ❌     |
| 23  | RECTIFIED_VST_ACCESS         | ❌     |
| 24  | ARGMAX                       | ✅     |
| 25  | CONVERT_COLOR                | ✅     |
| 26  | SORT_VEC                     | ✅     |
| 27  | INVERSION                    | ✅     |
| 28  | MAKE_TRANSFORM_MAT           | ✅     |
| 29  | SORT_MAT                     | ✅     |
| 30  | SWITCH_GLTF_RENDER_STATUS    | ❌     |
| 31  | UPDATE_GLTF                  | ❌     |
| 32  | RENDER_TEXT                  | ❌     |
| 33  | UPLOAD_TEXTURE_TO_GLTF       | ❌     |


