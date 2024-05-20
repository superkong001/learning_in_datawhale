内容引用参考： 

- https://github.com/datawhalechina/tiny-universe/tree/main/content/Qwen-blog
- https://github.com/InternLM/Tutorial/blob/main/langchain/readme.md

参考文献

- [When Large Language Models Meet Vector Databases: A Survey](http://arxiv.org/abs/2402.01763)
- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997)
- [Learning to Filter Context for Retrieval-Augmented Generation](http://arxiv.org/abs/2311.08377)
- [In-Context Retrieval-Augmented Language Models](https://arxiv.org/abs/2302.00083)

# RAG 基础

检索增强生成技术（Retrieval-Augmented Generation，RAG）：RAG 通过在语言模型生成答案之前，先从广泛的文档数据库中检索相关信息，然后利用这些信息来引导生成过程，极大地提升了内容的准确性和相关性。RAG 有效地缓解了幻觉问题，提高了知识更新的速度，并增强了内容生成的可追溯性，使得大型语言模型在实际应用中变得更加实用和可信。

```bash

# 获取源代码
git clone https://github.com/datawhalechina/tiny-universe.git
# 安装环境依赖 
pip install -r requirements

# 或者直接用docker
镜像地址：https://www.codewithgpu.com/i/datawhalechina/tiny-universe/tiny-universe-tiny-rag
```

向量数据库软件：llama-index 或者 QAnything (有道的)


RAG的基本结构有哪些呢？

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


