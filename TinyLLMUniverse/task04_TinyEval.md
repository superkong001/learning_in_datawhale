参考：

- https://github.com/datawhalechina/tiny-universe/tree/main/content/TinyEval
- https://zhuanlan.zhihu.com/p/33088748

# 大模型评测常用指标

常用词重叠评价指标metric：rouge, blue，METEOR 用于评估机器翻译质量的常用指标，通常用于衡量生成式语言模型（如机器翻译模型）的性能。

常用词向量评价指标：Greedy matching、Embedding Average、Vector Extrema、perplexity困惑度。

## ROUGE（Recall-Oriented Understudy for Gisting Evaluation）

ROUGE 是一组用于评估文本摘要或自动生成的摘要与参考摘要之间的相似程度的度量标准。它主要关注召回率（Recall），即生成的摘要中覆盖参考摘要多少内容。ROUGE 标准包括 ROUGE-N（比较 n 元组）、ROUGE-L（比较最长公共子序列）、ROUGE-W（比较加权 n 元组）、ROUGE-S（比较重要的非连续序列）等。ROUGE 分数越高，表示生成的摘要与参考摘要越相似。

## BLEU（Bilingual Evaluation Understudy）

BLEU 计算两句子的共现词频率。它是一种用于评估机器翻译质量的指标，它通过比较生成的翻译结果与参考翻译之间的 n-gram 重叠来度量翻译的准确性。BLEU 评分考虑了精确匹配的 n-gram 以及模糊匹配的情况，以及短文本的处理。BLEU 分数通常在 0 到 1 之间，越接近 1 表示生成的翻译结果越好。

参考论文《BLEU: a Method for Automatic Evaluation of Machine Translation》(https://link.zhihu.com/?target=http%3A//www.aclweb.org/anthology/P02-1040.pdf) 

机器翻译评价指标之BLEU详细计算过程：https://blog.csdn.net/guolindonggld/article/details/56966200

机器翻译自动评估-BLEU算法详解：https://blog.csdn.net/qq_31584157/article/details/77709454

## METEOR

METEOR是基于BLEU进行了一些改进，加入了生成响应和真实响应之间的对其关系。使用WordNet计算特定的序列匹配，同义词，词根和词缀，释义之间的匹配关系，改善了BLEU的效果，使其跟人工判别共更强的相关性。

![image](https://github.com/superkong001/learning_in_datawhale/assets/37318654/d067e7c3-75fd-4af2-a901-e440539bfaaa)

论文《METEOR: An automatic metric for mt evaluation with improved correlation with human judgments》

# 大模型Eval包含流程

评测任务的基础pipeline

![0d7e6154a76a67343a7daa68e144d1f6_compass](https://github.com/superkong001/learning_in_datawhale/assets/37318654/e610de11-8441-4b48-adca-e8feae39ed54)

- 首先，根据目标数据集的任务类型指定合理的评测metric.
- 根据目标数据的形式总结模型引导prompt.
- 根据模型初步预测结果采纳合理的抽取方式.
- 对相应的pred与anwser进行得分计算.



