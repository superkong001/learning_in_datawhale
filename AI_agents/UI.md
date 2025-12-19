# Gradio vs Streamlit
[Gradio](https://www.gradio.app/)有 输入输出组件、控制组件、布局组件 几个基础模块，其中
- 输入输出组件 用于展示内容和获取内容，如： Textbox 文本、 Image 图像
- 布局组件 用于更好地规划组件的布局，如： Column （把组件放成一列）、 Row （把组件放成一行）
  - 推荐使用 gradio.Blocks() 做更多丰富交互的界面， gradio.Interface() 只支持单个函数交互
- 控制组件 用于直接调用函数，无法作为输入输出使用，如： Button （按钮）、 ClearButton （清除按钮）

Gradio的设计哲学是将输入和输出组件与布局组件分开。输入组件（如 Textbox 、 Slider 等）用于接收用户输入，输出组件（如 Label 、 Image 等）用于显示函数的输出结果。而布局组件（如 Tabs 、 Columns 、 Row 等）则用于组织和排列这些输入和输出组件，以创建结构化的用户界面。

如果想了解更多组件详情，可查看[官方文档](https://www.gradio.app/guides/quickstart)；

另外，如果想设计更复杂的界面风格，还可以查看学习[官方文档：主题](https://www.gradio.app/guides/theming-guide)

[Streamlit](https://streamlit.io/) 基础概念

[Streamlit官方的视频演示](https://datawhale-business.oss-cn-hangzhou.aliyuncs.com/dashboard/dipwap/1763316509295/hero-video.mp4)，可以直接输入markdown格式的文本，网页即可渲染好。

Streamlit中没有gradio的输入和输出概念，也没有布局组件的概念。

Gradio和Streamlit中的【文件组件】对比如下图：

<img width="300" height="300" alt="d2c6d7ed0d253943bb09e9a8e652d7b4_62d8505a-3a42-47e1-b327-fc4aae0f2b4b" src="https://github.com/user-attachments/assets/c3969f83-b2dc-4993-850a-7c08cc5a3cd5" />
<img width="300" height="300" alt="f89b90f0c50dbf2347f7a8141c7a397e_fa54dc19-1739-47c1-b141-e0635ae2a4d3" src="https://github.com/user-attachments/assets/1c5495b9-2c4c-4ee4-b704-3fa151cf5bc1" />

Streamlit每个组件都是独立的，需要用什么直接查看官方文档即可，大致有如下几种组件：

页面元素
- 文本
- 数据表格
- 图标绘制（柱状图，散点图等）
- 输入（文本框，按钮，下拉框，滑块，复选框，文件上传，等）
- 多媒体（图片，音频，视频）
- 布局和容器
- Chat（聊天对话控件）
- 状态（进度条，加载中，等等元素）
- 第三方组件（提供了更加丰富的组件）

应用逻辑
- 导航和页面（可以切换页面）
- 执行流程
- 缓存和状态
- 连接和加密（可连接数据库，也可以对内容进行加密处理）
- 自定义组件
- 公共组件（用户信息存储，帮助，以及输出html）
- Config（使用配置文件，来定义一些内容）

工具
- 应用测试
- 命令行

# 案例：智能编程助手

```python
# 导入所需的库
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st

# 创建一个标题和一个副标题
st.title("💬 Yuan2.0 智能编程助手")

# 源大模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir='./')

# 定义模型路径
path = './IEITYuan/Yuan2-2B-Mars-hf'

# 定义模型数据类型
torch_dtype = torch.bfloat16 # A10
# torch_dtype = torch.float16 # P100

# 定义一个函数，用于获取模型和tokenizer
@st.cache_resource
def get_model():
    print("Creat tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
    tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)

    print("Creat model...")
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch_dtype, trust_remote_code=True).cuda()

    return tokenizer, model

# 加载model和tokenizer
tokenizer, model = get_model()

# 初次运行时，session_state中没有"messages"，需要创建一个空列表
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 每次对话时，都需要遍历session_state中的所有消息，并显示在聊天界面上
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 如果用户在聊天输入框中输入了内容，则执行以下操作
if prompt := st.chat_input():
    # 将用户的输入添加到session_state中的messages列表中
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 在聊天界面上显示用户的输入
    st.chat_message("user").write(prompt)

    # 调用模型
    prompt = "<n>".join(msg["content"] for msg in st.session_state.messages) + "<sep>" # 拼接对话历史
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
    outputs = model.generate(inputs, do_sample=False, max_length=1024) # 设置解码方式和最大生成长度
    output = tokenizer.decode(outputs[0])
    response = output.split("<sep>")[-1].replace("<eod>", '')

    # 将模型的输出添加到session_state中的messages列表中
    st.session_state.messages.append({"role": "assistant", "content": response})

    # 在聊天界面上显示模型的输出
    st.chat_message("assistant").write(response)
```

