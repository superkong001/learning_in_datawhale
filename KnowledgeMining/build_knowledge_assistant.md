# 参考

- 茴香豆

> [茴香豆Github](https://github.com/InternLM/HuixiangDou/tree/main)
> [茴香豆web搭建](https://github.com/InternLM/HuixiangDou/tree/main/web)
> [茴香豆本地搭建实战](https://github.com/InternLM/Tutorial/blob/camp3/docs/L2/Huixiangdou/readme.md)
> [茴香豆零编程接入微信](https://zhuanlan.zhihu.com/p/686579577)
> [茴香豆web](https://openxlab.org.cn/apps/detail/tpoisonooo/huixiangdou-web)

- 医疗知识问答

> [以疾病为中心的一定规模医药领域知识图谱，并以该知识图谱完成自动问答与分析服务](https://github.com/TommyZihao/QASystemOnMedicalKG/blob/master/README.md)

- Coze搭建A1客服助手

# 基于茴香豆

## web 版茴香豆

<img width="528" alt="image" src="https://github.com/user-attachments/assets/c3c957f3-5857-4306-a78d-4c5a958c348e">

<img width="435" alt="image" src="https://github.com/user-attachments/assets/a55a7ac4-3a06-4e1a-bc89-9887370b96ac">

三阶段 Pipeline （前处理、拒答、响应）:

<img width="470" alt="image" src="https://github.com/user-attachments/assets/d2ce9137-2b25-418e-8fe9-22331c1ae3e2">

## 茴香豆本地标准版搭建

参考： [茴香豆Github](https://github.com/InternLM/HuixiangDou/blob/main/README_zh.md)

硬件要求

以下是不同特性所需显存，区别仅在**配置选项是否开启**。

|                     配置示例                     | 显存需求 |                                                                                 描述                                                                                 |                             Linux 系统已验证设备                              |
| :----------------------------------------------: | :------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------: |
|         [config-cpu.ini](./config-cpu.ini)         |   -    | 用 [siliconcloud](https://siliconflow.cn/) API <br/>仅检索文本 | ![](https://img.shields.io/badge/x86-passed-blue?style=for-the-badge) |
|         [config-2G.ini](./config-2G.ini)         |   2GB    | 用 openai API（如 [kimi](https://kimi.moonshot.cn)、[deepseek](https://platform.deepseek.com/usage) 和 [stepfun](https://platform.stepfun.com/)）<br/>仅检索文本 | ![](https://img.shields.io/badge/1660ti%206G-passed-blue?style=for-the-badge) |
| [config-multimodal.ini](./config-multimodal.ini) |   10GB   |                                                                    用 openai API 做 LLM，图文检索                                                                    | ![](https://img.shields.io/badge/3090%2024G-passed-blue?style=for-the-badge)  |
|       【标准版】[config.ini](./config.ini)       |   19GB   |                                                                         本地部署 LLM，单模态                                                                         | ![](https://img.shields.io/badge/3090%2024G-passed-blue?style=for-the-badge)  |
|   [config-advanced.ini](./config-advanced.ini)   |   80GB   |                                                                本地 LLM，指代消歧，单模态，微信群实用                                                                | ![](https://img.shields.io/badge/A100%2080G-passed-blue?style=for-the-badge)  |

### 纯 CPU 版

没有 GPU，使用 siliconcloud API 完成模型推理。以 docker miniconda+Python3.11 为例，安装 cpu 依赖，运行：

```bash
# 启动容器
docker run  -v /path/to/huixiangdou:/huixiangdou  -p 7860:7860 -p 23333:23333  -it continuumio/miniconda3 /bin/bash
# 装依赖
apt update
apt install python-dev libxml2-dev libxslt1-dev antiword unrtf poppler-utils pstotext tesseract-ocr flac ffmpeg lame libmad0 libsox-fmt-mp3 sox libjpeg-dev swig libpulse-dev
python3 -m pip install -r requirements-cpu.txt
# 建立知识库
python3 -m huixiangdou.service.feature_store  --config_path config-cpu.ini
# 问答测试
python3 -m huixiangdou.main --standalone --config_path config-cpu.ini
# gradio UI
python3 -m huixiangdou.gradio_ui --config_path config-cpu.ini
```

```bash
# 1. 下载Docker镜像
# 在能够连接到外网的机器上，下载必要的Docker镜像：
docker pull continuumio/miniconda3

# 2. 导出Docker镜像
# 将下载的Docker镜像导出为一个文件：
docker save continuumio/miniconda3 > miniconda3.tar

# 3. 下载系统依赖
# 下载所需的系统依赖包。可以使用apt命令配合--download-only选项来下载但不安装它们：
apt-get update
apt-get install --download-only python-dev libxml2-dev libxslt1-dev antiword unrtf poppler-utils pstotext tesseract-ocr flac ffmpeg lame libmad0 libsox-fmt-mp3 sox libjpeg-dev swig libpulse-dev
# 这将下载所有依赖包但不安装，然后您可以从/var/cache/apt/archives/目录中找到这些.deb文件。

# 4. 下载Python依赖
# 在一个有外网的环境中，下载所有的Python库：
pip download -r requirements-cpu.txt
# 这将下载所有在requirements-cpu.txt文件中列出的库到当前目录。

# 5. 拷贝文件到外置媒介
# 将上述所有文件（Docker镜像、系统依赖的.deb文件、Python依赖包）拷贝到一个USB驱动器或其他传输设备中。

# 第二步：在内网服务器上部署
# 1. 导入Docker镜像
# 将Docker镜像文件传输到内网服务器后，加载镜像：
docker load < miniconda3.tar

# 2. 安装系统依赖
# 将所有.deb文件传输到服务器后，在服务器上安装它们：
dpkg -i /path/to/debs/*.deb

# 3. 安装Python依赖
# 将下载的Python包上传到服务器后，使用以下命令安装：
pip install --no-index --find-links=/path/to/downloaded/packages -r requirements-cpu.txt
# 这将从本地目录安装所有必需的Python库，而不是从互联网。

# 4. 启动Docker容器
# 使用之前提到的命令来启动Docker容器：
docker run -v /path/to/huixiangdou:/huixiangdou -p 7860:7860 -p 23333:23333 -it continuumio/miniconda3 /bin/bash

# 5. 运行应用
# 在容器内，根据之前提供的步骤运行应用程序：
# 建立知识库
python3 -m huixiangdou.service.feature_store --config_path config-cpu.ini
# 问答测试
python3 -m huixiangdou.main --standalone --config_path config-cpu.ini
# gradio UI
python3 -m huixiangdou.gradio_ui --config_path config-cpu.ini
```

# 基于Coze智能客服

[扣子](https://www.coze.cn/home)

- 人设

你是xxxx有限公司的智能客服;

你是一名擅长民法典、刑法、道路交通安全法、数据安全法相关法律问题咨询专家；

- 提示词

请注意！当业务资料不能直接回答客户提问的时候，你只能回复:抱歉，这个问题我不知道；
任务：根据业务资料【{{knowledge}}】来回答客户的提问【{{input}}】；
要求：输出内容需要你整理一下格式,用emoji表情让内容更加易读；

请注意！当法律资料不能直接回答客户提问的时候，你只能回复:抱歉，这个问题我不知道； 
任务：根据法律资料【{{KDD_output}}】来回答客户的提问【{{USER_INPUT}}】； 
要求：输出内容需要你整理一下格式,用emoji表情让内容更加易读；

Q：老人过世，父母双亡，只留有一妻子和一个女儿，有多个兄弟姐妹，在老人没有留下遗嘱情况下，财产如何继承？

Tips:

- 测试工作流的三个步骤

1) 直接问业务，要求智能客服介绍；
2) 描述需求匹配业务，但不要出现关键词；
3) 问一个原本没有的业务

- 四个锦囊

1) 请注意！当业务资料里没有能回答问题的资料时，一律回答：抱歉，这个问题我不知道
2) 根据情况添加禁止语句
3) 单独加一页概述/目录页
4) 上传到知识库，最好的是Q&A文档

实战：
1. 建立知识库

  创建知识库：
  
   <img width="401" alt="image" src="https://github.com/user-attachments/assets/937bd39f-8352-423a-8808-15f96c3239cb">

  上传文本：

   <img width="469" alt="image" src="https://github.com/user-attachments/assets/0b72ebe2-2ffa-453c-97bb-ad7a05838635">

  分段：

   <img width="769" alt="image" src="https://github.com/user-attachments/assets/d8ad52b7-371a-47b5-98d5-c61e761aaf34">

  手动调整：

   <img width="677" alt="image" src="https://github.com/user-attachments/assets/742ae224-766c-453f-b525-aadeb34fe580">
   
2. 创建工作流

 <img width="835" alt="image" src="https://github.com/user-attachments/assets/bf6d4b82-90fe-4d1a-8813-c209b0aa2a57">

 <img width="835" alt="image" src="https://github.com/user-attachments/assets/2597c354-d8b4-438b-863b-64ecdf5c9706">

3. 创建Bots

  <img width="409" alt="image" src="https://github.com/user-attachments/assets/4bafae28-af41-46fd-8f32-563c18ea3102">

  选择工作流模型，并添加工作流

4. 发布Bots

   <img width="605" alt="image" src="https://github.com/user-attachments/assets/fa51ade1-712a-4615-b105-e0ebbff7fce0">

