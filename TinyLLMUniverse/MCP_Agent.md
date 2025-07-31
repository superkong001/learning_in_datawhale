
<img width="451" height="570" alt="34a2752f5f5de7b5b85476fd64de6ac8_05782c31-bfdb-426f-ba93-0c74f345b358" src="https://github.com/user-attachments/assets/be71c648-fb2e-4ad3-af1d-0a7d8e418e02" />

参考：基于《甄嬛传》角色数据，构建了一个完整的端侧智能体解决方案。

代码地址：https://github.com/ditingdapeng/ollama_baseline

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

# 检查现有模型
ollama list

# 拉取 Qwen2.5-0.5B 基础模型（项目使用）
ollama pull qwen2.5:0.5b

# 验证基础模型下载成功
ollama list 

# 进入deployment目录
cd deployment

# 使用现有Modelfile创建ollama模型
ollama create huanhuan-qwen -f Modelfile.huanhuan

# 检查模型详细信息
ollama show huanhuan-qwen

# 启动交互式对话
ollama run huanhuan-qwen

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


> curl http://127.0.0.1:11434
# 预期返回：Ollama is running

sudo apt update && sudo apt install nginx -y
sudo systemctl enable --now nginx

参考：https://www.bilibili.com/opus/1075899615621939205?spm_id_from=333.1387.0.0
安装 uv （Windows为例）,UV是一个用 Rust 编写的极速 Python 包和项目管理工具，可替代pip、poetry、pyenv等多款工具，来实现python包的依赖管理。
> powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
> set Path=C:\Users\kong_\.local\bin;%Path% 
> uv python list

created a uv-managed project
> uv init mcp-server-demo

MCP 依赖添加
> cd mcp-server-demo
> uv add "mcp[cli]"

修改main.py写MCP服务器

<img width="641" height="427" alt="ef752730a6a14c0b16691618955487aa_image" src="https://github.com/user-attachments/assets/25d2d3af-f2ce-412f-8ba8-e9c57818af31" />


