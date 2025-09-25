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

import numpy as np
import pytest

import securemr as smr


class TestTensorVec:
    def test_scalar_float_roundtrip_and_indexing(self):
        data = np.array([0.1, 0.2, -5.0], dtype=np.float32)
        tensor = smr.TensorScalarFloat.from_numpy(data)

        assert len(tensor) == data.shape[0]
        np.testing.assert_allclose(tensor.numpy(), data)
        assert tensor[1] == pytest.approx(0.2)

        tensor[0] = 1.5
        tensor[2] = [7.0]
        expected = np.array([1.5, 0.2, 7.0], dtype=np.float32)
        np.testing.assert_allclose(tensor.numpy(), expected)

        with pytest.raises(ValueError):
            tensor[0] = (1.0, 2.0)

    def test_color_uint8_from_numpy_and_negative_index(self):
        data = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)
        tensor = smr.TensorColorUInt8.from_numpy(data)

        assert len(tensor) == 2
        assert tuple(tensor[0]) == (10, 20, 30)

        tensor[1] = (100, 101, 102)
        assert tuple(tensor[-1]) == (100, 101, 102)

        expected = np.array([[10, 20, 30], [100, 101, 102]], dtype=np.uint8)
        np.testing.assert_array_equal(tensor.numpy(), expected)

    def test_slice_int16_shape_validation(self):
        valid = np.array([[1, 2], [3, 4]], dtype=np.int16)
        tensor = smr.TensorSliceInt16.from_numpy(valid)
        np.testing.assert_array_equal(tensor.numpy(), valid)

        invalid = np.array([[1, 2, 3]], dtype=np.int16)
        with pytest.raises(ValueError):
            smr.TensorSliceInt16.from_numpy(invalid)

    def test_slice_uint16_setitem_length_validation(self):
        tensor = smr.TensorSliceUInt16(2)

        tensor[0] = (1, 2)
        assert tuple(tensor[0]) == (1, 2)

        with pytest.raises(ValueError):
            tensor[1] = (3,)
