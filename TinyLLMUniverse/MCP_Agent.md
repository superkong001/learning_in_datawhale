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

<img width="1907" height="953" alt="2fe45cfd49d75b48f7282f656c840e52_16ead059-0860-40c0-b1f9-357b8712ee17" src="https://github.com/user-attachments/assets/4280ac47-f918-49d0-a654-0881a30f50d7" />

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

<img width="1068" height="678" alt="image" src="https://github.com/user-attachments/assets/2b45f4a4-156b-4336-b734-8bf06bb976c1" />

- stdio:

<img width="814" height="558" alt="image" src="https://github.com/user-attachments/assets/2a69bc5c-ded8-4fde-a858-77ded4f113bd" />

- SSE:

<img width="833" height="529" alt="image" src="https://github.com/user-attachments/assets/a5c647c3-0820-4ec6-bcc9-d4a8baa2da32" />



