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
# python -m pip install optimum
# 安装了与生成式 AI 模型相关的扩展包以及 Hugging Face Optimum 库的 OpenVINO 支持模块
pip install openvino-genai==2024.4.0 optimum[openvino]
pip install einops

pip install opencv-python jupyter notebook openai appbuilder-sdk qianfan
# Verify that the Package Is Installed
python -c "from openvino import Core; print(Core().available_devices)"
```

[下载Qwen2-7B-Instruct开源大模型](https://huggingface.co/Qwen/Qwen2-7B-Instruct)

```bash
# 将原生大模型量化压缩为INT4的OpenVINO IR模型
# 使用 Hugging Face 的 optimum-cli 工具将一个基于 PyTorch 的生成式 AI 模型导出为 OpenVINO 格式，以便在 OpenVINO 框架中运行
optimum-cli export openvino --model "D:\github\internlm2_5-1_8b-chat" --task text-generation-with-past --weight-format int4 --group-size 128 --ratio 0.8 --trust-remote-code "D:\github\internlm2_5-1_8b-chat-int4-ov"

# 参数解释
optimum-cli export openvino \
  --model "D:\github\internlm2_5-1_8b-chat" \
  --task text-generation-with-past \ # 指定任务类型，这里是生成式文本任务
  --weight-format int4 \ # 指定模型权重的精度为 int4，即 4 位整数精度。
  --group-size 128 \ # 指定量化分组的大小，即每 128 个参数被分成一个组进行处理
  --ratio 0.8 \ # 量化比例，0.8 表示保持 80% 的权重信息
  --trust-remote-code \
  "D:\github\internlm2_5-1_8b-chat-int4-ov" # 指定输出模型的名称和路径

# 远程拉取保存到当前目录
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

开启大模型对话

```bash
# 导入工具包
import openvino_genai as ov_genai
# 选择计算设备
device = 'CPU' 
# device = 'GPU'
# device = 'NPU'
# 载入OpenVINO IR格式的大模型
pipe = ov_genai.LLMPipeline("D:\github\internlm2_5-1_8b-chat-int4-ov", device)
# 提示词
prompt = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n什么是OpenVINO？<|im_end|>\n<|im_start|>assistant\n"
# 大模型推理预测
result = pipe.generate(prompt)
result.texts
print(result.texts[0])
```

```bash
函数：大模型对话
def chat_qwen_ov(question="什么是OpenVINO？"):
    prompt = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(question)
    result = pipe.generate(prompt)
    return result.texts[0]
result = chat_qwen_ov('什么是OpenVINO？')
print(result)
```

