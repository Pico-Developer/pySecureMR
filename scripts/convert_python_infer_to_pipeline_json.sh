python_code="infer_cards_onnx.py"
pysecuremr="/Users/duino/code/securemr/pySecureMR"

echo "
# Goal（一句话）
把${python_code}里面描述的模型pipeline流程，转成securemr pipeline，
并保存json，并在设备上通过一致性测试。
务必使用spatialml skill

# Inputs（路径 / repo / 版本）
1. pysecuremr: ${pysecuremr}
2. mnistwild.py: ${pysecuremr}/examples/mnistwild/mnistwild.py
3. spatialml skill: ${pysecuremr}/skills/spatialml/SKILL.md
4. 使用.venv环境运行python程序

# Non-goals（明确不做什么）
1. pipeline.json不要遗漏后处理，即使后处理非常复杂，也需要实现到securemr pipeline

# Steps（必须先给 plan）
1. 理解${python_code}的pipeline流程，包括前处理、模型infer、后处理。
2. 利用spatialml skill，理解securemr operators
3. 参考mnistwild.py，实现一个yolov8_convert.py，把python实现的逻辑，转为securemr operators实现，并保存json
4. 利用spatialml skill，在设备上测试生成的pipeline json
5. 如果log报错，fix，直到log正常跑没报错
6. 给定和python相同的input，对比output的数值一致性。如果不一致，可以尝试分段检查。
    先检查前处理的输出是否一致，
    再检查模型infer的输出是否一致，
    最后检查后处理的输出是否一致。
7. 如果数值不一致，fix，直到数值一致。
8. 可视化pipeline json保存的结果，和python结果做对比。

# Done（可自动验证）
1. python ${python_code} 运行的结果，和利用pipeline inspect工具(python -m securemr.inspect.pipeline_cli)得到的结果，数值一致。
2. 保存两个可视化结果。

确保直到done条件完成了，才可以停止任务。
确保直到done条件完成了，才可以停止任务。
确保直到done条件完成了，才可以停止任务。
"
