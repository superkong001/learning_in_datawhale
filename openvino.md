> https://docs.openvino.ai/2024/index.html
>

酷睿Ultra(带神经处理单元NPU)用于AI推理

安装环境配置

```bash
# 创建虚拟环境
conda create -n openvino python=3.10
conda activate openvino

# 将环境添加到Jupyter
python -m ipykernel install --user --name openvino --display-name "Python (openvino)"
jupyter kernelspec list # 检查现有的 Jupyter kernel
# 删除无效的 kernel
jupyter kernelspec uninstall openvino

# 安装OpenVINO和其它工具包
# pip install openvino==2024.4.0
# 安装了与生成式 AI 模型相关的扩展包以及 Hugging Face Optimum 库的 OpenVINO 支持模块
pip install openvino-genai==2024.4.0 optimum[openvino]

pip install opencv-python jupyter notebook openai appbuilder-sdk qianfan
# Verify that the Package Is Installed
python -c "from openvino import Core; print(Core().available_devices)"
```

[下载Qwen2-7B-Instruct开源大模型](https://huggingface.co/Qwen/Qwen2-7B-Instruct)

```bash
# 将原生大模型量化压缩为INT4的OpenVINO IR模型
!optimum-cli export openvino \
             --model "Qwen/Qwen2-7B-Instruct" \
             --task text-generation-with-past \
             --weight-format int4 \
             --group-size 128 \
             --ratio 0.8 \
             "Qwen2-7B-Instruct-int4-ov"
```

- [optimum库介绍](https://www.bilibili.com/video/BV1SQpceiEMh)
- [optimum-cli参数说明](https://huggingface.co/docs/optimum/intel/openvino/export)
- [或者直接下载转换好的OpenVINO INT4 IR模型文件](https://www.modelscope.cn/models/snake7gun/Qwen2-7B-Instruct-int4-ov)

