内容引用参考： 

- https://github.com/datawhalechina/tiny-universe/tree/main/content/Qwen-blog
- https://github.com/InternLM/Tutorial/blob/main/langchain/readme.md

参考文献

- [When Large Language Models Meet Vector Databases: A Survey](http://arxiv.org/abs/2402.01763)
- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997)
- [Learning to Filter Context for Retrieval-Augmented Generation](http://arxiv.org/abs/2311.08377)
- [In-Context Retrieval-Augmented Language Models](https://arxiv.org/abs/2302.00083)

# 下载 NLTK 相关资源

下载 nltk 资源并解压

```bash
cd /root
git clone https://gitee.com/yzy0612/nltk_data.git  --branch gh-pages
cd nltk_data
mv packages/*  ./
cd tokenizers
unzip punkt.zip
cd ../taggers
unzip averaged_perceptron_tagger.zip
```

# RAG 基础

检索增强生成技术（Retrieval-Augmented Generation，RAG）：RAG 通过在语言模型生成答案之前，先从广泛的文档数据库中检索相关信息，然后利用这些信息来引导生成过程，极大地提升了内容的准确性和相关性。RAG 有效地缓解了幻觉问题，提高了知识更新的速度，并增强了内容生成的可追溯性，使得大型语言模型在实际应用中变得更加实用和可信。

RAG的基本结构：

- 要有一个向量化模块，用来将文档片段向量化。
- 要有一个文档加载和切分的模块，用来加载文档并切分成文档片段。
- 要有一个数据库来存放文档片段和对应的向量表示。
- 要有一个检索模块，用来根据 Query （问题）检索相关的文档片段。
- 要有一个大模型模块，用来根据检索出来的文档回答用户的问题。

![61572dea5f98f67045cb2ef775c5d292_Retrieval-Augmented%20Generation%EF%BC%88RAG-Learning%EF%BC%89_raw=true](https://github.com/superkong001/learning_in_datawhale/assets/37318654/e6664e76-76d5-42f9-93b0-e046806246e0)

 RAG 的流程:

- 索引：将文档库分割成较短的 Chunk，并通过编码器构建向量索引。
- 检索：根据问题和 chunks 的相似度检索相关文档片段。
- 生成：以检索到的上下文为条件，生成问题的回答。

![16d74b949a9268e765ebe7b2810a1461_RAG](https://github.com/superkong001/learning_in_datawhale/assets/37318654/cc4ad35c-4963-47b8-88e2-b584e3232899)

```python
# 获取源代码
git clone https://github.com/datawhalechina/tiny-universe.git
# 安装环境依赖 
pip install -r requirements.txt

# 或者直接用docker
镜像地址：https://www.codewithgpu.com/i/datawhalechina/tiny-universe/tiny-universe-tiny-rag
```

# 向量化

动手实现一个向量化的类

## 创建 Embedding 基类

```python
class BaseEmbeddings:
    """
    Base class for embeddings
    """
    def __init__(self, path: str, is_api: bool) -> None:
        self.path = path
        self.is_api = is_api
    
    def get_embedding(self, text: str, model: str) -> List[float]:
        raise NotImplementedError
    
    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude
```

## 继承BaseEmbeddings类， 编写get_embedding方法

```python
class OpenAIEmbedding(BaseEmbeddings):
    """
    class for OpenAI embeddings
    """
    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            from openai import OpenAI
            self.client = OpenAI()
            self.client.api_key = os.getenv("OPENAI_API_KEY")
            self.client.base_url = os.getenv("OPENAI_BASE_URL")
    
    def get_embedding(self, text: str, model: str = "text-embedding-3-large") -> List[float]:
        if self.is_api:
            text = text.replace("\n", " ")
            return self.client.embeddings.create(input=[text], model=model).data[0].embedding
        else:
            raise NotImplementedError
```

# 文档加载和切分

代码在 RAG/utils.py 文件中, 加载的文件类型有：pdf、md、txt

## 递归读取文件

```python
# 该函数将递归指定文件夹路径，返回其中所有满足条件（即后缀名为 .md 或者 .txt 的文件）的文件路径：
import os 
def get_files(dir_path):
    # args：dir_path，目标文件夹路径
    file_list = []
    for filepath, dirnames, filenames in os.walk(dir_path):
        # os.walk 函数将递归遍历指定文件夹
        for filename in filenames:
            # 通过后缀名判断文件类型是否满足要求
            if filename.endswith(".md"):
                # 如果满足要求，将其绝对路径加入到结果列表
                file_list.append(os.path.join(filepath, filename))
            elif filename.endswith(".txt"):
                file_list.append(os.path.join(filepath, filename))
            elif filename.endswith(".pdf"):
                file_list.append(os.path.join(filepath, filename))
    return file_list
```

## 文档加载

```python
def read_file_content(cls, file_path: str):
    # 根据文件扩展名选择读取方法
    if file_path.endswith('.pdf'):
        return cls.read_pdf(file_path)
    elif file_path.endswith('.md'):
        return cls.read_markdown(file_path)
    elif file_path.endswith('.txt'):
        return cls.read_text(file_path)
    else:
        raise ValueError("Unsupported file type")
```

## 切分&构建向量数据库

### 手搓

#### 分块

- 按 Token 的长度来切分文档。设置一个最大的 Token 长度(如：600)，根据这个最大的 Token 长度来切分文档。
- 切分的时候片段与片段之间最好有一些重叠的内容(如：150)，才能保证检索的时候能够检索到相关的文档片段。还有就是切分文档的时候最好以句子为单位，也就是按 \n 进行粗切分，这样可以基本保证句子内容是完整的。

```python
def get_chunk(cls, text: str, max_token_len: int = 600, cover_content: int = 150):
    chunk_text = []

    curr_len = 0
    curr_chunk = ''

    lines = text.split('\n')  # 假设以换行符分割文本为行

    for line in lines:
        line = line.replace(' ', '')
        line_len = len(enc.encode(line))
        if line_len > max_token_len:
            print('warning line_len = ', line_len)
        if curr_len + line_len <= max_token_len:
            curr_chunk += line
            curr_chunk += '\n'
            curr_len += line_len
            curr_len += 1
        else:
            chunk_text.append(curr_chunk)
            curr_chunk = curr_chunk[-cover_content:]+line
            curr_len = line_len + cover_content

    if curr_chunk:
        chunk_text.append(curr_chunk)

    return chunk_text
```

#### 构建向量数据库

数据库对于最小RAG架构来说，需要实现功能：

- persist：数据库持久化，本地保存
- load_vector：从本地加载数据库
- get_vector：获得文档的向量表示
- query：根据问题检索相关的文档片段

代码可以在 RAG/VectorBase.py 文件中

```python
class VectorStore:
    def __init__(self, document: List[str] = ['']) -> None:
        self.document = document

    def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:
        # 获得文档的向量表示
        pass

    def persist(self, path: str = 'storage'):
        # 数据库持久化，本地保存
        pass

    def load_vector(self, path: str = 'storage'):
        # 从本地加载数据库
        pass

    def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1) -> List[str]:
        # 根据问题检索相关的文档片段
        pass
```

```python
# 仅使用 Numpy 进行加速，存储为array然后根据数据（相似度）从低到高排列，k为返回最相似的个数
def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1) -> List[str]:
    query_vector = EmbeddingModel.get_embedding(query)
    result = np.array([self.get_similarity(query_vector, vector)
                        for vector in self.vectors])
    return np.array(self.document)[result.argsort()[-k:][::-1]].tolist()
```

### 调用工具

引入 LangChain 框架中构建向量数据库。由纯文本对象构建向量数据库，需要先对文本进行分块，接着对文本块进行向量化。

#### 文本进行分块

LangChain 文本分块工具，使用字符串递归分割器，并选择分块大小为 500，块重叠长度为 150

LangChain 文本分块可以参考教程： [《LangChain - Chat With Your Data》](https://github.com/datawhalechina/prompt-engineering-for-developers/blob/9dbcb48416eb8af9ff9447388838521dc0f9acb0/content/LangChain%20Chat%20with%20Your%20Data/1.%E7%AE%80%E4%BB%8B%20Introduction.md)

#### 文本块进行向量化

选用开源词向量模型 [Sentence Transformer](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) 来进行文本向量化。LangChain 提供了直接引入 HuggingFace 开源社区中的模型进行向量化的接口：

```python
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="/root/data/model/sentence-transformer")
```

#### 加载向量数据库

选择 Chroma 作为向量数据库，基于上文分块后的文档以及加载的开源向量化模型，将语料加载到指定路径下的向量数据库。

其他向量数据库工具：llama-index 或者 QAnything (有道的)

#### 调用langchain完整代码

```python
# 首先导入所需第三方库
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import os

# 获取文件路径函数
def get_files(dir_path):
    # args：dir_path，目标文件夹路径
    file_list = []
    for filepath, dirnames, filenames in os.walk(dir_path):
        # os.walk 函数将递归遍历指定文件夹
        for filename in filenames:
            # 通过后缀名判断文件类型是否满足要求
            if filename.endswith(".md"):
                # 如果满足要求，将其绝对路径加入到结果列表
                file_list.append(os.path.join(filepath, filename))
            elif filename.endswith(".txt"):
                file_list.append(os.path.join(filepath, filename))
            elif filename.endswith(".pdf"):
                file_list.append(os.path.join(filepath, filename))
    return file_list

# 加载文件函数
# 使用 LangChain 提供的 Loader 对象来加载目标文件
def get_text(dir_path):
    # args：dir_path，目标文件夹路径
    # 首先调用上文定义的函数得到目标文件路径列表
    file_lst = get_files(dir_path)
    # docs 存放加载之后的纯文本对象
    docs = []
    # 遍历所有目标文件
    for one_file in tqdm(file_lst):
        file_type = one_file.split('.')[-1]
        if file_type == 'pdf':
            loader = UnstructuredPDFLoader(one_file)
        elif file_type == 'md':
            loader = UnstructuredMarkdownLoader(one_file)
        elif file_type == 'txt':
            loader = UnstructuredFileLoader(one_file)
        else:
            # 如果是不符合条件的文件，直接跳过
            continue
        docs.extend(loader.load())
    return docs

# 目标文件夹
tar_dir = [
    "/root/data/InternLM",
    "/root/data/InternLM-XComposer",
    "/root/data/lagent",
    "/root/data/lmdeploy",
    "/root/data/opencompass",
    "/root/data/xtuner"
]

# 加载目标文件
docs = []
for dir_path in tar_dir:
    docs.extend(get_text(dir_path))

# 对文本进行分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=150)
split_docs = text_splitter.split_documents(docs)

# 加载开源词向量模型
embeddings = HuggingFaceEmbeddings(model_name="/root/data/model/sentence-transformer")

# 构建向量数据库
# 定义持久化路径
persist_directory = 'data_base/vector_db/chroma'
# 加载数据库
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory=persist_directory  # 允许将persist_directory目录保存到磁盘上
)
# 将加载的向量数据库持久化到磁盘上
vectordb.persist()
```

定义 Embeddings到持久化到磁盘前的代码得到 vectordb 对象即为已构建的向量数据库对象，该对象可以针对用户的 query 进行语义向量检索，得到与用户提问相关的知识片段。

# 大模型模块

或者用ollama试试？

构建一个大模型基类

```python
class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        pass

    def load_model(self):
        pass
```

基于基类构建InternLM大模型类，因为使用开源模型需要load_model方法。

```python
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Any, List, Optional

class InternLMChat(BaseModel): #如使用langchain则改成LLM，需先导入from langchain.llms.base import LLM
    tokenizer : AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_path : str = '') -> None:
        # model_path: InternLM 模型路径
        # 从本地初始化模型
        super().__init__(model_path)
        self.load_model()

    def chat(self, prompt: str, history: List = [], content: str='') -> str:
        prompt = PROMPT_TEMPLATE['InternLM_PROMPT_TEMPALTE'].format(question=prompt, context=content)
        response, history = self.model.chat(self.tokenizer, prompt, history)
        return response

    def load_model(self):
        print("正在从本地加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.path, torch_dtype=torch.float16, trust_remote_code=True).cuda()
        self.model = self.model.eval() #可省略？
        print("完成本地模型的加载")

     # 如需要可以重新调用函数
     def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        # 重写调用函数
        system_prompt = """....
        """
        
        messages = [(system_prompt, '')]
        response, history = self.model.chat(self.tokenizer, prompt , history=messages)
        return response

     @property
     def _llm_type(self) -> str:
        return "InternLM"
 ```

用一个字典来保存所有的prompt，构建检索问答链需要构建的 Prompt Template

```python
PROMPT_TEMPLATE = dict(
    InternLM_PROMPT_TEMPALTE="""先对上下文进行内容总结,再使用上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:"""
)
```

tips: 使用langchain的

from langchain.prompts import PromptTemplate

调用 LangChain 的方法来实例化一个 Template 对象，该对象包含了 context 和 question 两个变量，在实际调用时，这两个变量会被检索到的文档片段和用户提问填充
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],template=template)

# langchain构建检索问答链

还需实例化一个基于 InternLM 自定义的 LLM 对象

```python
from LLM import InternLM_LLM
llm = InternLM_LLM(model_path = "/root/data/model/Shanghai_AI_Laboratory/internlm-chat-7b")
llm.predict("你是谁")
```

# LLM RAG Demo

```python
from RAG.VectorBase import VectorStore
from RAG.utils import ReadFiles
from RAG.LLM import OpenAIChat, InternLMChat
from RAG.Embeddings import JinaEmbedding, ZhipuEmbedding

if vector is not None:
   # 保存数据库之后
   vector = VectorStore()
   vector.load_vector('./storage') # 加载本地的数据库
else:
   # 没有保存数据库
   docs = ReadFiles('./data').get_content(max_token_len=600, cover_content=150) # 获得data目录下的所有文件内容并分割
   vector = VectorStore(docs)
   embedding = ZhipuEmbedding() # 创建EmbeddingModel
   vector.get_vector(EmbeddingModel=embedding)
   vector.persist(path='storage') # 将向量和文档内容保存到storage目录下，下次再用就可以直接加载本地的数据库

question = 'git的原理是什么？'

content = vector.query(question, model='zhipu', k=1)[0]
chat = InternLMChat(path='model_path')
print(chat.chat(question, [], content))
```




