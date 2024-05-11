# word2vec

## 基础知识

<img width="949" alt="Image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/3e4e0275-dfd3-43c9-a1c6-42784020341d">

<img width="863" alt="Image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/5bdf8de6-ff69-40f9-8772-e76e39645d0b">

<img width="523" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/c72ee92e-9209-4ea7-99f1-5feb93182a4c">

前向神经网络语言模型 Feedforward Neural Net Language Model(NNLM)

<img width="523" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/88e74b15-60ea-458d-b3d4-cb05b4e8f063">

## skip-gram

<img width="523" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/c46d29da-3e51-432d-a69e-dba762cd039a">

<img width="523" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/2b2f0e97-a0b5-459d-9060-8f04c03eaa3c">

Skip-gram目标函数:

<img width="523" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/fb9e4522-9af6-4492-ba4a-c1b807d89dc8">

## CBow

<img width="523" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/e0811b97-8095-486d-b8c6-f8f1ebe46322">

<img width="523" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/5ee2a76b-6078-48e4-bedb-89793e8eb8c5">

## 层次softmax（Hierarchical Softmax）

构建Huffman树:

<img width="523" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/792880f3-08ef-4b5f-8e40-e20a0480655d">

构建层次softmax:

<img width="523" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/e0a43a8f-3274-4da2-a9d7-285df47385aa">

<img width="523" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/b305b0cc-1866-4d22-a633-0f28e36f69bc">

<img width="523" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/8bfcd8f3-86bc-4093-9039-7c364e383535">

<img width="523" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/23eed1b4-cb41-424d-80f9-f75dc24e5193">

## Negative Sampling

舍弃多分类，提升速度

<img width="523" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/c32d2638-3d76-4604-88f9-734784cf9b4b">

<img width="523" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/2ec1024d-34ba-4467-8d28-a248ca96d9ad">

### CBOW Negative Sampling:

<img width="523" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/48cb24fe-eb3e-4405-8a12-9a042c87fdd2">

### Subsampling of Frequent Words

自然语言处理共识：文档或者数据集中出现频率高的词往往携带信息较少，比如the，is，a，and，而出现频率低的词往往携带信息多。

<img width="523" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/28be545a-4461-4a74-8889-9ce45b8ef2d7">

重采样的原因:

> 想更多地训练重要的词对，比如训练“France”和“Paris”之间的关系比训练“France”和“the”之间的关系要有用。

> 高频词很快就训练好了，而低频次需要更多的轮次。

重采样方法:

<img width="523" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/00c746c8-5f6b-4163-952b-ff8190258758">

重采样的分析:

<img width="523" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/352d03c7-d992-40d2-a58f-4254979ef098">

## 模型复杂度

O=ExTxQ

> O是训练复杂度training complexity

> E是训练迭代次数number of the training epochs

> T是数据集大小number of the words in the training set

> Q是模型计算复杂度model computational complexity

### Bengio A neural probabilistic language model(2003)

<img width="523" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/99a1dae0-7b13-44ef-bd44-f37728265108">

### 循环神经网络语言模型.(RNNLM)

<img width="523" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/a625145a-95c3-4528-ab14-3967e02625b9">

### Hierarchical复杂度

<img width="523" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/45761971-714b-468e-8cde-51d48632c719">

### Negative Sampling复杂度

<img width="523" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/db48c86b-7224-454c-b773-128a3464517f">

### CBOW复杂度

<img width="523" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/5c985c42-af5a-487e-ab38-6f04013915b1">

### 模型复杂度对比

<img width="523" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/b4db4063-eae8-47df-b9ed-0f111d295724">


#  节点表示学习

节点嵌入的目标是对节点进行编码，使得嵌入空间中的相似性（例如点积）近似于原始网络中的相似性，

节点嵌入算法通常由三个基本阶段组成：

<img width="623" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/5edbcc68-b403-463a-858b-88c6c5e4f79e">

节点嵌入的方法：深度游走和Node2Vec

## 深度游走

随机游走：

> 给定一个图和一个起点，我们随机选择它的一个邻居，并移动到这个邻居；

> 然后我们随机选择该点的邻居，并移动到它，以此类推。

<img width="515" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/41b649ba-4e9a-4569-8a43-f952f84206b3">

随机游走满足随机游走的想法，但是深度游走算法特指运行固定长度、无偏的随机游走。即：所有的游走序列具有相同的长度。"无偏"指的是从当前节点到其所有邻居节点的转移概率是均等的，即每个邻居被选择的机会相同，没有特定的方向倾向。这种游走生成的节点序列在训练模型时被视作等价于句子中的词序列，用于通过神经网络模型学习节点的低维表示向量。其将节点看作“单词”，将游走序列看作“句子”，然后用类似于Word2Vec的方法来学习节点嵌入。

<img width="643" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/8789cec6-bedd-4cf4-942e-ed1e84c85fe5">

损失函数𝐿被定义为所有节点对上的负对数概率的总和。对于一个节点𝑢和其邻居𝑣，这个函数计算的是模型预测邻居节点𝑣在节点𝑢的邻域中出现的概率的负对数。
而具体的概率计算是基于节点的特征向量之间的点积，并通过softmax函数进行归一化。

 
𝑃(𝑣∣Z𝑢)表示给定节点𝑢的特征向量Z𝑢时，节点𝑣被选为𝑢的邻居的条件概率。分子代表节点𝑢和𝑣之间相似性度量的指数形式；分母是对图中所有可能邻居𝑛的相似性度量进行同样计算并求和，应用 softmax 函数来计算的。

<img width="635" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/01eb7e54-304f-404b-b65b-878900a7ccde">

原本的softmax计算涉及所有其他节点的特征向量，但是负采样仅仅从一个预定义的分布𝑃𝑣中随机选择𝑘个“负”样本（即不是当前上下文中的节点）。然后，这𝑘个样本被用来近似分母中的求和项。

## 有偏的随机游走：Node2Vec

深度游走，即从每个节点开始运行固定长度、无偏的随机游走。但是，这种游走策略太死板，会限制表征的学习。Node2Vec 提出了一种更高效的、灵活的、有偏的随机游走策略，以得到一个更好的NR(u)。Node2Vec通过图上的广度优先遍历（Breath First Search, BFS）和深度优先遍历（Depth First Search, DFS）在网络的局部视图和全局视图之间进行权衡。

BFS 可以给出邻域的局部微观视图，而 DFS 提供邻域的全局宏观视图。 这里定义返回参数 p 来代表模型返回前一个节点的转移概率和输入输出参数 q 定义 BFS 和 DFS 的“比率”。当 p 的值比较小的时候，Node2Vec 像 BFS；当 q 的值比较小的时候，Node2Vec 像 DFS。

<img width="635" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/17fd18b0-0209-4296-b539-84fc46367252">

Node2Vec 算法：

> 计算随机游走概率。

> 模拟 r 个从节点 u 开始长度为 l 的随机游走。

> 使用随机梯度下降优化 node2vec 目标。

# 图表示学习

在某些应用中嵌入整个图 G（例如，对有毒分子与无毒分子进行分类、识别异常图）。

<img width="297" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/05f31b64-923f-47ee-912a-7e10ef971d34">

完成图嵌入有几种方法：

> 简单的想法是在图 G 上运行图的节点嵌入技术，然后对图 G 中的节点嵌入求和（或平均）。

> 引入“虚拟节点”来表示图并运行标准图节点嵌入技术。

> 使用匿名游走嵌入。 为了学习图嵌入，枚举所有可能的匿名游走，并记录它们的计数，然后将图表示为这些游走的概率分布。


