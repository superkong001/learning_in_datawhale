<img width="641" height="427" alt="ef752730a6a14c0b16691618955487aa_image" src="https://github.com/user-attachments/assets/25d2d3af-f2ce-412f-8ba8-e9c57818af31" />

<img width="451" height="570" alt="34a2752f5f5de7b5b85476fd64de6ac8_05782c31-bfdb-426f-ba93-0c74f345b358" src="https://github.com/user-attachments/assets/be71c648-fb2e-4ad3-af1d-0a7d8e418e02" />

参考:

> https://www.datawhale.cn/activity/354/learn/200/4430/51/43

> https://www.datawhale.cn/learn/content/192/4320

# 基于《甄嬛传》角色数据，构建了一个完整的端侧智能体解决方案。

代码地址：https://github.com/ditingdapeng/ollama_baseline

## 环境部署

参考：https://www.bilibili.com/opus/1075899615621939205?spm_id_from=333.1387.0.0

### 安装 uv （Windows为例）

UV是一个用 Rust 编写的极速 Python 包和项目管理工具，可替代pip、poetry、pyenv等多款工具，来实现python包的依赖管理。

```
# 1. 安装uv（如果未安装）
pip install uv

> powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
> set Path=C:\Users\kong_\.local\bin;%Path% 
> uv python list

# 2. 创建虚拟环境
uv venv huanhuan_env

# 3. 激活环境
conda activate huanhuan_env

# 4. 使用uv安装依赖（比pip快10-100倍）
uv pip install -r requirements.txt
```

## 数据下载（嬛嬛）

从GitHub获取甄嬛传角色对话数据集，为后续的数据预处理和模型训练提供原始数据。

```
python dataScripts/download_data.py
```

<img width="1075" height="188" alt="913c841a5b8682628e48121e7ee29a74_4ea6c200-224a-42bb-b2ad-08f55f2fe023" src="https://github.com/user-attachments/assets/5e58be7e-6968-4839-9721-4b2274ca571c" />

## 数据预处理

原始数据是一个JSON文件，包含着甄嬛的对话数据，但是训练模型不能直接吃这种大块数据，需要把它切成小块，分成训练集、验证集和测试集，还要转换成JSONL格式，每行是一个JSON对象。

```
# 处理全部数据
python dataScripts/huanhuan_data_prepare.py

# 只处理50条数据（用于快速测试）
python dataScripts/huanhuan_data_prepare.py 50
```

<img width="1021" height="200" alt="3d3ce0844219d71f8b2284fa30fd7f7a_7891db64-c031-4c44-9208-6bcc03e2b18d" src="https://github.com/user-attachments/assets/218f252a-2487-409f-885d-855060e06a45" />

输出的数据内容，每一行为一个JSON：

<img width="754" height="153" alt="1a636023a8b6f000f3e3c12b54c19ec2_4e50697c-50c0-4eac-88ca-b7b25ee03f5f" src="https://github.com/user-attachments/assets/96d2ed7d-93c9-475e-8c2c-bf975012a41d" />

## 模型训练

模型训练模块解决的问题是将预处理后的甄嬛传数据转换为具有甄嬛语言风格的对话模型 。该模块采用LoRA（Low-Rank Adaptation）微调技术，在保持基础模型能力的同时，让模型学会甄嬛的说话方式和语言特色。

<img width="2051" height="1100" alt="1261a94b860161b0bd531f4da600a610_1336696f-1dbb-45ce-b635-50742ead9d8c" src="https://github.com/user-attachments/assets/c44dad17-db91-4873-a9cc-e1a6184fbb0f" />

四个组件：基础模型、分词器、LoRA适配器、训练参数。

- 这里的基础模型，我们使用的是Qwen2.5-0.5B，该模型提供了基础的语言理解和生成能力，同时Ollama也原生支持，后续部署时可以直接通过ollama来拉取和加载；

- 分词器，可以理解为语言翻译，它会将文本转成向量，又能将模型输出的向量转回为可读文本；

- LoRA适配器，做个性化训练，保持原来的模型参数不变，通过在注意力层添加小型矩阵，让模型学会甄嬛的风格，做到角色塑造；

- 训练参数，提供的是学习策略，控制了学习的快慢、每次看多少内容、内容学几遍等；

```
# 开始训练
python training/huanhuan_train.py
```

<img width="1921" height="634" alt="a764ba530f4cf9cfe9ee66334b809098_886309e0-b468-469a-bf58-41629cc284b6" src="https://github.com/user-attachments/assets/d58d2a60-e5ac-4e29-a57c-6d2c2749c07a" />

## 训练监控

训练监控模块的实现，解决的问题是实时监控训练过程中的系统资源使用情况，并生成详细的监控报告，满足比赛中提供监控报告的要求。

```
# 启动监控（建议在训练开始前启动），在其他窗口同时启动监控（推荐）
python training/sync_monitor.py

# 监控会自动：
# 1. 检测GPU类型（NVIDIA/Apple Silicon）
# 2. 查找训练相关进程
# 3. 每5秒收集系统资源数据
# 4. 实时显示监控信息
# 5. 记录数据到JSONL文件
# 6. 按Ctrl+C停止时生成报告
```

<img width="1602" height="346" alt="50448548092df0007e9fef98c71c7490_52bdac9c-4122-441d-88b2-98c5ee485e28" src="https://github.com/user-attachments/assets/0c6f8da9-e9d6-4722-96b3-4c0ab49cd65d" />

## 模型部署-Ollama

olloma
常用命令：
ollama list                    # 列出所有已下载模型
ollama pull <model>           # 下载/更新模型
ollama rm <model>             # 删除模型
ollama show <model>           # 显示模型信息
ollama run <model>            # 交互式运行
ollama run <model> "问题"     # 单次问答
ollama run <model> --verbose  # 显示性能信息
ollama serve                  # 启动服务
ollama create <name> -f <file> # 创建自定义模型

系统要求：

- 磁盘空间：至少预留10GB空间（用于后续模型下载）

- 内存要求：根据模型大小而定

- 7B模型：至少8GB内存

- 13B模型：至少16GB内存

- 33B模型：至少32GB内存

### 安装和下载ollama

```
# 启动 Ollama 服务（需要保持运行）
ollama serve

# 新开终端验证服务状态
curl http://localhost:11434/api/tags

> curl http://127.0.0.1:11434
# 预期返回：Ollama is running

# 指定端口启动（如果11434端口被占用）
OLLAMA_HOST=0.0.0.0:11435 ollama serve

# 检查现有模型
ollama list

# 拉取 Qwen2.5-0.5B 基础模型（项目使用）
ollama pull qwen2.5:0.5b

# 验证基础模型下载成功
ollama list 
```

### 创建Ollama模型

```
# 进入deployment目录
cd deployment

# 使用现有Modelfile创建ollama模型
ollama create huanhuan-qwen -f Modelfile.huanhuan
```

### 模型验证和测试

```
# 检查模型详细信息
ollama show huanhuan-qwen

# 启动交互式对话
ollama run huanhuan-qwen

# 进入对话模式后可以：
# - 直接输入问题进行对话
# - 使用 Ctrl+D 或 /bye 退出
# - 使用 /clear 清除上下文

# 单次问答（不进入交互模式）
ollama run huanhuan-qwen "你是谁？"

# 多行输入
ollama run huanhuan-qwen """
给我讲个故事，
要求包含：
1. 主角是甄嬛
2. 故事简短
"""

# 查看模型执行效率详情
ollama run huanhuan-qwen --verbose "介绍一下你自己"

# 输出信息包括：
# - total duration: 总运行时间
# - load duration: 模型加载时间
# - prompt eval count: 提示词token数量
# - prompt eval duration: 提示词处理时间
# - eval count: 响应token数量
# - eval duration: 响应生成时间
```

## Web应用模块-Streamlit

基于Streamlit构建的甄嬛角色对话Web应用，提供友好的用户界面和实时对话功能。支持模型选择、流式对话、连接状态监控、参数调节、对话历史管理等完整功能。

Streamlit是一个基于Python的开源框架，专门用于快速构建数据科学和机器学习的Web应用，无需前端开发（html、css、js）经验，只使用纯Python代码就能构建美观的Web应用。

```
pip install streamlit

# 1. 确保Ollama服务运行
ollama serve

# 2. 确保甄嬛模型已部署
ollama list

# 3. 启动Web应用
streamlit run application/huanhuan_web.py

# 4. 访问Web界面
# 浏览器自动打开: http://localhost:8501
```

<img width="800" height="600" alt="2fe45cfd49d75b48f7282f656c840e52_16ead059-0860-40c0-b1f9-357b8712ee17" src="https://github.com/user-attachments/assets/4280ac47-f918-49d0-a654-0881a30f50d7" />

文件结构：

```
application/
├── huanhuan_web.py          # 主应用文件
├── chat_history/            # 对话历史存储目录
│   └── huanhuan_chat_*.json # 历史记录文件
└── __pycache__/            # Python缓存目录
```

## agent应用-MCP

MCP模块解决的问题是为甄嬛模型提供标准化的工具接口，让外部应用能够通过统一的协议与模型进行交互。通过标准化的MCP协议，将本地部署的甄嬛模型集成到Claude Desktop中，实现AI助手扩展。

### 创建一个uv-managed 的 mcp-server-demo项目工程

```
> mkdir mcp-server-demo & cd mcp-server-demo
> uv init mcp-server-demo

# MCP 依赖添加
> cd mcp-server-demo
> uv add "mcp[cli]"
```

修改main.py写MCP Server

```
# 方式1: 直接运行模块
python -m mcp_server

# 方式2: 运行主文件
python mcp-server-demo/server.py
```

Claude Desktop配置文件(claude_desktop_config.json)

```
{
  "mcpServers": {
    "huanhuan-chat": {
      "command": "python",
      "args": ["-m", "mcp_server"],
      "cwd": "/Users/dapeng/Code/study/ollamaALL/ollama_文档",
      "env": {
        "OLLAMA_HOST": "http://localhost:11434",
        "HUANHUAN_MODEL": "huanhuan_fast"
      }
    }
  }
}
```

配置客户端（以Cherry Studio为例）

<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/2b45f4a4-156b-4336-b734-8bf06bb976c1" />

- stdio:

<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/2a69bc5c-ded8-4fde-a858-77ded4f113bd" />

- SSE:

<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/a5c647c3-0820-4ec6-bcc9-d4a8baa2da32" />

## 基于魔搭搭建MCP

1. 复制魔搭创空间

https://www.modelscope.cn/studios/minmin1023/luner_info1

<img width="540" height="480" alt="52839406be84031e13ffa88555e80198_image" src="https://github.com/user-attachments/assets/50fddf24-ea4a-482e-9795-57fbcaf52936" />

<img width="540" height="480" alt="237103ef901ef33a424a92f2f279c5d6_image" src="https://github.com/user-attachments/assets/a5bd4dea-5315-4cc8-8573-02576b6eb66e" />

2. 复制 MCP Server 配置

点击【通过API使用】，获取【MCP】页的 MCP Server 安装命令

<img width="640" height="480" alt="e644d370e1f4f534fa5cdd73620c4d8c_image" src="https://github.com/user-attachments/assets/5b40b28e-bea2-4f88-a744-49629b32e5ce" />

<img width="640" height="480" alt="75ede2d7d70625fc9a0c7c4d42817bef_image" src="https://github.com/user-attachments/assets/49cb7820-efba-4ce5-b96a-2657a20184ce" />

3. 使用魔搭社区官方提供的 [mcp-inspector](https://modelscope.cn/studios/modelscope/mcp-inspector) 进行测试。

<img width="640" height="480" alt="44a6cc751dbac09e7f1ad1c86be3fb21_image" src="https://github.com/user-attachments/assets/5442d7f1-080c-4fbb-bf72-65504a8babf3" />

<img width="640" height="480" alt="76a820c6532d078eb6a87c000680b2e6_image" src="https://github.com/user-attachments/assets/78db549c-bece-4395-a9c2-9f684b6fd382" />

<img width="640" height="480" alt="826bbe19993e233ea3f06e27176a0e87_image" src="https://github.com/user-attachments/assets/8c12493e-d92f-4bd9-a5aa-fbaa30212dfa" />

4. 发布MCP Server

https://www.modelscope.cn/mcp/servers/create

- 填写MCP Server信息

<img width="640" height="480" alt="3d1cf93bc51cd17aba78f25e193db9f2_image" src="https://github.com/user-attachments/assets/7a686e81-dac6-44dd-a4c0-3d8d8170bbd5" />

- 粘贴创空间链接

<img width="640" height="480" alt="c549fe38d0d144f894d745f4a3bcfa4e_image" src="https://github.com/user-attachments/assets/caae0231-1f2a-4eef-9227-d516f9bbd9b2" />

<img width="640" height="480" alt="c94cf1c8f54e36c92e99659c9116226e_image" src="https://github.com/user-attachments/assets/6536b003-6c98-4aa7-99cf-13addd1a9103" />

- 填写其他基本信息

<img width="640" height="480" alt="1ee6107c3d2ff0e787c7b6e74553a025_image" src="https://github.com/user-attachments/assets/3724e271-1716-44eb-8e83-b38facdedb51" />

- 填写README信息，创建！

```
# luner_info每日黄历
这是一个可以获取某天黄历情况的MCP，你可以输入空或日期获取当日或某日的黄历。日期格式形如：1991年1月1日或1991-01-01。
获取得到的回答如下：
日期 : 2025-07-01 00:00:00 农历 : 二零二五 乙巳[蛇]年 六月大初七 星期 : 星期二 八字 : 乙巳 壬午 辛未 戊子 今日节气: 无 下一节气: ('小暑', (7, 7), 2025) 季节 : 仲夏 生肖冲煞: 羊日冲牛 星座 : 巨蟹座 吉神方位: ['喜神西南', '财神正东', '福神西北', '阳贵东北', '阴贵正南'] 宜 : ['祭祀', '出行', '宴会', '沐浴', '剃头', '修造', '上表章', '上官', '进人口', '竖柱上梁', '经络', '纳财', '扫舍宇', '栽种', '牧养', '破土', '安葬', '祈福', '恤孤茕', '举正直', '裁制', '纳采', '搬移', '招贤', '宣政事', '覃恩', '施恩', '安抚边境', '解除', '求嗣', '整手足甲', '庆赐', '修仓库', '立券交易', '选将', '营建', '上册', '出师', '临政', '纳畜', '缮城郭', '整容', '颁诏', '雪冤'] 忌 : ['畋猎', '取鱼']
```

<img width="640" height="480" alt="51ed46ae0ec5e9a2f68cbf3bd44847bf_image" src="https://github.com/user-attachments/assets/9224a429-4868-474f-b2ee-10326a4dd15e" />

- MCP Server 发布

<img width="640" height="480" alt="a73c1f82726cd8e7eadcc6b3ad6baeaa_image" src="https://github.com/user-attachments/assets/fee05322-da7a-41dc-8cfc-1144cdab6736" />

## Gradio (Gradio能将普通的Python函数“变身”为可被大模型调用的MCP工具)

Gradio 是一个非常适合快速搭建MCP Server的工具。它能将Python函数轻松转化为Web界面，并支持MCP Server模式，代码量少且易于上手。

目前搭建mcp server的方法有很多，可以直接使用Python的mcp库或者fastmcp库，搭建mcp服务后本地调用。

如果需要部署到云，目前可以部署到pipy平台或者云服务器上。但是在pipy平台由于网络问题，不方便国内用户直接使用。

部署在云上可以在自有云或托管云平台，自有云上会面临网络配置等问题，对于新手也不太友好。

有些小伙伴甚至没有自己的云服务器。所以我们推荐把目标放在托管云平台。

使用gradio快速搭建MCP服务，结合魔搭创空间和gradio的奇妙组合，既解决了MCP云服务平台的问题，整体的搭建也不是很复杂。

代码量少而且很好玩，可以在20分钟快速搞定一个自己的mcp。

### 其核心技术栈如下：

- Python： 作为主要的编程语言。

- Gradio： 用于快速构建Web界面和MCP Server。

- MCP相关库： Gradio内置了对MCP协议的支持，你无需额外安装复杂的MCP库。

### Gradio 有 输入输出组件、控制组件、布局组件 几个基础模块

- 输入输出组件 输入组件（如 Textbox 、 Slider 等）用于接收用户输入，输出组件（如 Label 、 Image 等）用于显示函数的输出结果。

  输入输出组件 都有以下三个参数：

  - fn ：绑定的函数，输入参数需与 inputs 列表类型对应
  
  - inputs ：输入组件变量名列表，（例如： [msg, chatbot] ）
  
  - ouputs ：输出组件变量名列表，（例如： [msg, chatbot] ）

  另外不同的 输入输出组件、控制组件 有不同动作可响应（对应一些.方法，如下面的 msg.submit() ）

- 布局组件 用于组织和排列这些输入和输出组件，以创建结构化的用户界面。如： Column （把组件放成一列）、 Row （把组件放成一行）

    推荐使用 gradio.Blocks() 做更多丰富交互的界面， gradio.Interface() 只支持单个函数交互

- 控制组件 用于直接调用函数，无法作为输入输出使用，如： Button （按钮）、 ClearButton （清除按钮）

### 使用Gradio进行MCP Server开发主要有以下要点：

- 明确输入输出： 这是MCP Server设计的核心。你的函数需要清晰地定义接受什么参数（输入）和返回什么结果（输出）。例如，黄历查询的输入是一个日期字符串，输出是黄历信息字符串。

- 准确的MCP声明（Docstring）： 函数的Docstring必须准确、完整地描述其功能、参数类型、描述和返回值。大模型会依赖这些信息来理解和调用你的工具，如果声明不准确，大模型可能无法正确调用。

- 功能实现： 编写实际的Python代码来实现MCP Server的核心逻辑。这可能涉及调用第三方API（使用requests库）、进行数据处理（使用pandas库）、或执行特定计算。

- 异常处理： 考虑各种可能出现的异常情况（如无效输入、网络请求失败、API调用错误），并使用try-except块进行适当的错误处理，以提高MCP Server的稳定性和鲁棒性。

- 依赖管理： 确保你的requirements.txt文件包含了所有必要的Python库（例如gradio, requests, cnlunar等），以便在魔搭创空间部署时能够正确安装。

### 将函数快速变为MCP工具过程主要包括以下几步：

1. 函数定义： 编写一个普通的Python函数，实现你MCP Server的核心功能。

2. MCP声明（Docstring）： 这是最关键的一步！在你的Python函数的Docstring（文档字符串）中，按照特定的格式清晰地描述函数的用途、接受的参数以及返回的结果。大模型会“阅读”这些Docstring来理解你的工具，并决定何时以及如何调用它。

  示例 Docstring 结构：

  ```
  [函数的功能描述]
  
  Args:
      [参数名1]: [参数类型] [参数描述]
      [参数名2]: [参数类型] [参数描述]
      ...
  
  Returns:
      [返回值类型] [返回值描述]
  ```

3. Gradio接口封装： 使用gradio.Interface将你的Python函数封装成一个Gradio应用。它会自动根据你的函数签名和Docstring生成用户界面。

  ```
  import gradio as gr
  # ... 你的函数定义 ...
  demo = gr.Interface(
      fn=你的函数名,
      inputs=["text"], # 根据你的函数输入类型调整
      outputs="text",  # 根据你的函数输出类型调整
      title="你的MCP Server名称",
      description="你的MCP Server描述"
  )
  ```

4. 启动MCP Server模式： 在启动Gradio应用时，设置mcp_server=True，Gradio就会自动暴露符合MCP协议的接口。

  ```
  demo.launch(mcp_server=True)
  ```



