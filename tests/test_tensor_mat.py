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

import securemr as smr
import cv2
import numpy as np
import pytest

class TestTensorMat:
    @pytest.fixture
    def test_data(self):
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    @pytest.fixture
    def test_shape(self):
        return [100, 100]
        
    class TestShapeVerification:
        """形状验证相关的测试"""
        
        @pytest.mark.skipif(not smr.LD_LIBRARY_PATH_IS_SET, reason="LD_LIBRARY_PATH is not set")
        def test_valid_shapes(self):
            """测试有效的形状"""
            assert smr.TensorMat.verify_shape([100, 100])
            assert smr.TensorMat.verify_shape([50, 50])
            
        @pytest.mark.skipif(not smr.LD_LIBRARY_PATH_IS_SET, reason="LD_LIBRARY_PATH is not set")
        @pytest.mark.parametrize("invalid_shape", [
            [],  # 空形状
            [100],  # dim=1
            [0, 100, 3],  # dim=3
        ])
        def test_invalid_shapes(self, invalid_shape):
            """测试无效的形状"""
            assert not smr.TensorMat.verify_shape(invalid_shape)

    class TestCreation:
        """TensorMat创建相关的测试"""
        
        @pytest.mark.skipif(not smr.LD_LIBRARY_PATH_IS_SET, reason="LD_LIBRARY_PATH is not set")
        def test_create_direct(self):
            """测试直接创建TensorMat"""
            tensor = smr.TensorMat(
                [100, 100],
                3,  # channels
                smr.EDataType.UINT8
            )
            assert tensor is not None
            
        @pytest.mark.skipif(not smr.LD_LIBRARY_PATH_IS_SET, reason="LD_LIBRARY_PATH is not set")
        def test_create_from_numpy(self, test_data):
            """测试从numpy数组创建"""
            tensor = smr.TensorMat.from_numpy(test_data)
            assert tensor is not None
            result = tensor.numpy()
            np.testing.assert_array_equal(result, test_data)

    class TestOperations:
        """TensorMat操作相关的测试"""
            
        @pytest.mark.skipif(not smr.LD_LIBRARY_PATH_IS_SET, reason="LD_LIBRARY_PATH is not set")
        @pytest.mark.parametrize("test_data,expected", [
            (np.zeros((100, 100, 3), dtype=np.uint8), (False, False)),  # 全零数组
            (np.ones((100, 100, 3), dtype=np.uint8), (True, True)),    # 全一数组
        ])
        def test_all_any(self, test_data, expected):
            """测试all和any操作"""
            tensor = smr.TensorMat.from_numpy(test_data)
            assert tensor.any() == expected[0]
            assert tensor.all() == expected[1]

    class TestErrorHandling:
        """错误处理相关的测试"""
        
        @pytest.mark.skipif(not smr.LD_LIBRARY_PATH_IS_SET, reason="LD_LIBRARY_PATH is not set")
        def test_invalid_dtype(self):
            """测试无效的数据类型"""
            invalid_data = np.random.rand(100, 100, 3)  # float类型
            with pytest.raises(Exception):
                smr.TensorMat.from_numpy(invalid_data)
