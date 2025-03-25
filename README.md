# pySecureMR

Python bindings for [SecureMR](https://path-to-SecureMR-link) project.

## Why pySecureMR?

When developing a SecureMR app, it's not very easy to debug pipeline.
You are not allowed to access each operator output directly. `pySecureMR` happens here
to rescue you from complicated and painful debugging time. We bind most of SecureMR
operators to python so you can call each operator and check input and output.

## Supported platform
- Linux (ubuntu22): YES
- Windows (wsl2, ubuntu22): YES
- Mac: NO

## Supported python
- 3.10

## Install

```bash
git clone [TODO: update when porting to github]
cd pySecureMR
pip3 install -e "."
```
Check installation:
```bash
source ./env
python3 -c "import securemr"
```

## Run test

```bash
source ./env
pytest
```
Refer to [test code](./tests) to learn more about the usage.
