SecureMR_Samples=/Users/duino/ws/yunhao_publish/SecureMR_Samples

echo "
# Goal（一句话）
根据SecureMR_Samples samples里实现的各个例子，结合现有的spatialml skill，总结
SecureMR_Samples samples中还有哪些实现operator的小技巧，特别是用多个operator组合
完成一个新功能的技巧，补充到operator_tips.md文档中

# Inputs（路径 / repo / 版本）
1. SecureMR_Samples: ${SecureMR_Samples}
2. spatialml skill: ./skills/spatialml/SKILL.md
3. operator_tips.md: ./skills/spatialml/reference/operator_tips.md

# Non-goals（明确不做什么）
1. 不要重复添加已经有的tips

# Steps（必须先给 plan）
1. 理解现有的operator_tips.md
2. 仔细理解SecureMR_Samples samples里各个cpp实现的pipeline
3. 整理新的tips，补充到operator_tips.md

# Done（可自动验证）

"
