SecureMR_Samples=/Users/duino/ws/yunhao_publish/SecureMR_Samples

echo "
# Goal（一句话）
根据SecureMR_Samples的operator和openxr，更新pyspatialml的EOperatorType，以及spatialml skill中的pipeline spec.

# Inputs（路径 / repo / 版本）
1. SecureMR_Samples: ${SecureMR_Samples}
2. operator in pipeline: ${SecureMR_Samples}/base/securemr_utils/pipeline.h
3. openxr: ${SecureMR_Samples}/external/openxr/include/openxr/openxr.h
4. EOperatorType: securemr/core/types.py
5. pipeline spec: skills/spatialml/reference/pipeline_json_spec.md
"
