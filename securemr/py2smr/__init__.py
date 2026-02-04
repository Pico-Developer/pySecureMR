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
"""
py2smr: Python function to SecureMR operator conversion tool.

This module provides utilities to convert Python functions into SecureMR
pipeline JSON files that can be verified on device using pipeline-inspect.

Usage:
    from securemr.py2smr import trace, ops, convert, verify

    @trace(inputs=["image"], outputs=["result"])
    def preprocess(image):
        normalized = ops.arithmetic(image, "{0} / 255.0")
        return normalized

    # Execute with tracing
    test_input = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    result, trace_ctx = preprocess.trace(image=test_input)

    # Convert to pipeline JSON
    convert(trace_ctx, output="pipeline.json")

    # Verify numerical consistency
    verify(
        pipeline="pipeline.json",
        inputs={"image": test_input},
        expected_outputs={"result": result}
    )
"""

from .tracer import trace, TraceContext
from .converter import convert
from .verifier import verify
from . import ops

__all__ = [
    "trace",
    "TraceContext",
    "convert",
    "verify",
    "ops",
]
