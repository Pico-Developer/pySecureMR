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

import pytest
import securemr as smr
import numpy as np

class TestTensorFactory:
    """TensorFactory测试类"""
    
    @pytest.mark.skipif(not smr.LD_LIBRARY_PATH_IS_SET, reason="LD_LIBRARY_PATH is not set")
    def test_create_placeholder(self):
        """测试创建placeholder"""
        shape = [100, 100, 3]
        dtype = smr.EDataType.UINT8
        tensor = smr.TensorFactory.create_placeholder(shape, dtype)
        assert tensor is not None
        
    @pytest.mark.skipif(not smr.LD_LIBRARY_PATH_IS_SET, reason="LD_LIBRARY_PATH is not set")
    def test_create_tensor(self):
        """测试创建普通tensor"""
        shape = [100, 100, 3]
        dtype = int(smr.EDataType.UINT8) | smr.BaseType.MAT
        tensor = smr.TensorFactory.create(shape, dtype)
        assert tensor is not None

        shape = [4]
        dtype = int(smr.EDataType.UINT8) | smr.BaseType.POINT_2
        tensor = smr.TensorFactory.create(shape, dtype)
        assert tensor is not None
        
    @pytest.mark.skipif(not smr.LD_LIBRARY_PATH_IS_SET, reason="LD_LIBRARY_PATH is not set")
    def test_memory_usage(self):
        """测试内存使用统计"""
        initial_usage = smr.TensorFactory.get_total_memory_usage()
        
        # 创建一些tensor
        shape = [100, 100, 3]
        dtype = int(smr.EDataType.UINT8) | smr.BaseType.MAT
        tensor1 = smr.TensorFactory.create(shape, dtype)
        dtype = int(smr.EDataType.FLOAT32) | smr.BaseType.MAT
        tensor2 = smr.TensorFactory.create(shape, dtype)
        
        # 验证内存使用增加
        current_usage = smr.TensorFactory.get_total_memory_usage()
        assert current_usage > initial_usage
