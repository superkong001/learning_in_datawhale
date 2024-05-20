参考： 

- https://github.com/datawhalechina/tiny-universe/tree/main/content/Qwen-blog
- https://github.com/InternLM/Tutorial/blob/main/langchain/readme.md

# RAG 基础

检索增强生成技术（Retrieval-Augmented Generation，RAG）：RAG 通过在语言模型生成答案之前，先从广泛的文档数据库中检索相关信息，然后利用这些信息来引导生成过程，极大地提升了内容的准确性和相关性。RAG 有效地缓解了幻觉问题，提高了知识更新的速度，并增强了内容生成的可追溯性，使得大型语言模型在实际应用中变得更加实用和可信。

-- 获取源代码
git clone https://github.com/datawhalechina/tiny-universe.git
pip install -r requirements

向量数据库软件：llama-index 或者 kill anything
