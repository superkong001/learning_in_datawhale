参考：

- https://github.com/datawhalechina/tiny-universe/tree/main/content/TinyEval
- https://zhuanlan.zhihu.com/p/33088748

# 大模型评测常用指标

常用词重叠评价指标metric：rouge, blue，METEOR 用于评估机器翻译质量的常用指标，通常用于衡量生成式语言模型（如机器翻译模型）的性能。

常用词向量评价指标：Greedy matching、Embedding Average、Vector Extrema、perplexity困惑度。

## 词重叠评价指标

### ROUGE（Recall-Oriented Understudy for Gisting Evaluation）

ROUGE 是一组用于评估文本摘要或自动生成的摘要与参考摘要之间的相似程度的度量标准，词可以不是连续的。它主要关注召回率（Recall），即生成的摘要中覆盖参考摘要多少内容。ROUGE 标准包括 ROUGE-N（比较 n 元组）、ROUGE-L（比较最长公共子序列）、ROUGE-W（比较加权 n 元组）、ROUGE-S（比较重要的非连续序列）等。ROUGE 分数越高，表示生成的摘要与参考摘要越相似。

<img width="212" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/2a1555c8-9b35-43f0-a7de-59185a823f56">

### BLEU（Bilingual Evaluation Understudy）

BLEU 计算两句子的共现词频率，n-gram要求词语必须连续出现。它是一种用于评估机器翻译质量的指标，它通过比较生成的翻译结果与参考翻译之间的 n-gram 重叠来度量翻译的准确性。BLEU 评分考虑了精确匹配的 n-gram 以及模糊匹配的情况，以及短文本的处理。BLEU 分数通常在 0 到 1 之间，越接近 1 表示生成的翻译结果越好。

<img width="244" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/b7334935-15d4-4692-b452-64696f7846f5">

第一个公式Pn用于计算n-gram短语词组在整个数据集中的准确度。h(k,r)表示每个n-gram词组在真实响应中出现的次数。故上式就是每个n-gram词组在真实和生成响应中出现次数的较小值求和除以其在生成响应中出现次数求和，表征了一种精确度度量。n的取值（一般取1-4）。

第二个公式，beta表示各个n-gram的权重（可以取均匀分布），也就是对1-4进行加权求和，而b(r,r^)表示长度惩罚因子，即我们不想让生成的答案长度太短，所以加一个惩罚因子来改善效果。

参考论文[《BLEU: a Method for Automatic Evaluation of Machine Translation》](https://link.zhihu.com/?target=http%3A//www.aclweb.org/anthology/P02-1040.pdf) 

机器翻译评价指标之BLEU详细计算过程：https://blog.csdn.net/guolindonggld/article/details/56966200

机器翻译自动评估-BLEU算法详解：https://blog.csdn.net/qq_31584157/article/details/77709454

### METEOR

METEOR是基于BLEU进行了一些改进，加入了生成响应和真实响应之间的对其关系。使用WordNet计算特定的序列匹配，同义词，词根和词缀，释义之间的匹配关系，改善了BLEU的效果，使其跟人工判别共更强的相关性。

![image](https://github.com/superkong001/learning_in_datawhale/assets/37318654/d067e7c3-75fd-4af2-a901-e440539bfaaa)

论文《METEOR: An automatic metric for mt evaluation with improved correlation with human judgments》

## 词向量评价指标

### Greedy matching

<img width="258" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/ffb639d3-a045-4df0-8990-657b31105b37">

### Embedding Average

<img width="232" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/c03bae65-3e77-44b2-9e6c-7d3fd0bc162d">

### Vector Extrema

与Embedding Average方法类似，也是先通过词向量计算出句向量，在使用句向量之间的余弦相似度表示二者的相似度。不过句向量采用向量极值法进行计算。

### perplexity困惑度

<img width="473" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/8a8b7bb7-4805-4f0c-a7a7-780682d0d200">

# 数据集

## 对话数据集

参考[《A Survey of Available Corpora for Building Data-Driven Dialogue Systems》的作者整理](https://docs.google.com/spreadsheets/d/1SJ4XV6NIEl_ReF1odYBRXs0q6mTkedoygY3kLMPjcP8/pubhtml#)

## 英文数据集

- [Cornell Movie Dialogs：电影对话数据集](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)。
- [Ubuntu Dialogue Corpus：Ubuntu日志对话数据](https://arxiv.org/abs/1506.08909)。
-[ OpenSubtitles：电影字幕](http://opus.lingfil.uu.se/OpenSubtitles.php)。
- [Twitter：twitter数据集](https://github.com/Marsan-Ma/twitter_scraper)。
- [Papaya Conversational Data Set](https://github.com/bshao001/ChatLearner)：基于Cornell、Reddit等数据集重新整理之后，好像挺干净的。

相关数据集的处理代码或者处理好的数据可以参见下面两个github项目：

- [DeepQA](https://github.com/Conchylicultor/DeepQA)
- [chat_corpus](https://github.com/Marsan-Ma/chat_corpus)

## 中文数据集

- [dgk_shooter_min.conv](https://github.com/rustch3n/dgk_lost_conv)：中文电影台词数据集，
- [白鹭时代中文问答语料](https://github.com/Samurais/egret-wenda-corpus)：白鹭时代论坛问答数据，一个问题对应一个最好的答案。
- [微博数据集：华为李航实验室发布](http://61.93.89.94/Noah_NRM_Data/)，也是论文“Neural Responding Machine for Short-Text Conversation”使用数据集。
- [新浪微博数据集](http://lwc.daanvanesch.nl/openaccess.php)，评论回复短句。

# 大模型Eval包含流程

评测任务的基础pipeline

<img width="521" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/f538477f-57c7-4c97-a227-a75b711052db">

- 首先，根据目标数据集的任务类型指定合理的评测metric.
- 根据目标数据的形式总结模型引导prompt.
- 根据模型初步预测结果采纳合理的抽取方式.
- 对相应的pred与anwser进行得分计算.

选用数据集：

|name|type|metric|
|---|---|---|
|multi_news|长文本问答|Rouge|
|multifieldqa_zh|短文本问答|F1|
|trec|生成式选则|accuracy|

## 设计自己prompt(llama factory)

```python
{
    "instruction": "假设你是皇帝身边的女人--甄嬛",
    "input": "你是谁?",
    "output": "臣妾是甄嬛，家父是大理寺少卿。"
}
```

```python
# 配置每个数据集的prompt, 将上面自定义sft数据分别封装到`custom_zh`和`custom_en`,数据形式与sft格式一致
{
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "custom_zh": "提问:{input} \n回答: ",
    "custom_en": "Question:{input} \nAnswer: "
}
```




