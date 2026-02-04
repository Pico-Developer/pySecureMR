# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test helpers for platform-specific SecureMR bindings."""

import pytest

from securemr.bindings import bindings_available, load_bindings


def _bindings_ready() -> bool:
    if not bindings_available():
        return False
    try:
        load_bindings()
        __import__("securemr._bindings._securemr")
        return True
    except Exception:  # noqa: BLE001
        return False


def pytest_collection_modifyitems(config, items):  # noqa: D103
    if _bindings_ready():
        return
    skip_bindings = pytest.mark.skip(
        reason="SecureMR native bindings are unavailable on this platform."
    )
    for item in items:
        nodeid = str(item.nodeid)
        # Skip tests that don't require native bindings
        if "tests/test_inspect.py" in nodeid:
            continue
        if "tests/test_py2smr.py" in nodeid:
            continue
        if "tests/py2smr_ops/" in nodeid:
            continue
        item.add_marker(skip_bindings)
