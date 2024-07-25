图深度学习=图神经网络=图表示学习=几何深度学习：使用深度学习对图进行机器学习，关键是将图表示学习映射成向量（图嵌入、节点表示学习：把节点映射为低维连续稠密向量）

<img width="734" alt="image" src="https://github.com/user-attachments/assets/97360e8c-fc30-427a-97a4-236780f98744">

# word2vec

## 基础知识

<img width="949" alt="Image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/3e4e0275-dfd3-43c9-a1c6-42784020341d">

<img width="863" alt="Image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/5bdf8de6-ff69-40f9-8772-e76e39645d0b">

<img width="757" alt="Image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/05786170-4bd4-4660-99e3-d93ca2d6f828">


前向神经网络语言模型 Feedforward Neural Net Language Model(NNLM)

<img width="1044" alt="Image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/c2b99838-9ed6-4c2f-b23c-4d2500f7d86c">

<img width="654" alt="Image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/2ce0ee0f-fe40-4d8c-ac8f-c2a7f1fb14b9">

## skip-gram

<img width="818" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/1640d6e3-4d96-46ed-b45f-f9bf8abf43ec">

<img width="539" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/99a4c321-d916-4bf5-a8c9-44e89537cc93">

Skip-gram目标函数:

<img width="703" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/6087c591-f1e5-44b8-ae13-ad7815d307ee">

## CBow

<img width="728" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/059fdea4-9246-4bdb-829a-c213f85ecf9f">

<img width="854" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/158ea501-de7c-408f-88b8-3bfff92b376b">


## 层次softmax（Hierarchical Softmax）

构建Huffman树:

<img width="662" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/5c27460f-e10b-4c9a-8bb2-39dc26a22526">

构建层次softmax:

<img width="683" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/ad736c67-eea2-4da0-abd5-7e1380961ca4">

<img width="812" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/1fc04639-37a5-4609-8761-cc5f00dac37f">

<img width="646" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/435bdf35-b686-4950-b098-deedd984c6ce">

<img width="811" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/363c4b95-5302-4d34-b453-6d6a83ef238f">

## Negative Sampling

舍弃多分类，提升速度

<img width="773" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/f23584bb-40c7-4b5c-a717-3e59c332758e">

<img width="803" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/deb00370-59f6-491c-ad25-7782a07b2bdb">

### CBOW Negative Sampling:

<img width="713" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/81b9572a-588e-427d-9862-eeca84a08e37">

### Subsampling of Frequent Words

自然语言处理共识：文档或者数据集中出现频率高的词往往携带信息较少，比如the，is，a，and，而出现频率低的词往往携带信息多。

<img width="488" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/669cfaa5-ebcc-43e3-b492-e60350929016">

重采样的原因:

> 想更多地训练重要的词对，比如训练“France”和“Paris”之间的关系比训练“France”和“the”之间的关系要有用。

> 高频词很快就训练好了，而低频次需要更多的轮次。

重采样方法:

<img width="610" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/6912caf1-2fde-4941-91a5-4ac4ad384fdf">

重采样的分析:

<img width="596" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/9502a703-286d-4c4c-94a1-33779c1f2188">


## 模型复杂度

O=ExTxQ

> O是训练复杂度training complexity

> E是训练迭代次数number of the training epochs

> T是数据集大小number of the words in the training set

> Q是模型计算复杂度model computational complexity

### Bengio A neural probabilistic language model(2003)

<img width="714" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/1245ab88-d01c-49c5-85e2-871676da873d">

### 循环神经网络语言模型.(RNNLM)

<img width="771" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/717c71c1-ccc4-47ec-b4ae-016aafb30964">

### Hierarchical复杂度

<img width="610" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/2cc3237b-827d-4470-94ad-0a9d2a23591d">

### Negative Sampling复杂度

<img width="626" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/76036ed3-667d-4d00-93d2-b7e1603fc142">

### CBOW复杂度

<img width="470" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/d8d74ae7-8a6f-41b5-9add-31e08f51241a">

### 模型复杂度对比

<img width="829" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/126a1234-fcb7-473d-9763-2793577c0a49">

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

- 计算随机游走概率。Alias Sampling （https://keithschwarz.com/darts-dice-coins）

https://www.bilibili.com/video/av798804262

https://www.cnblogs.com/Lee-yl/p/12749070.html

- 模拟 r 个从节点 u 开始长度为 l 的随机游走。
- 使用随机梯度下降优化 node2vec 目标。

# 图表示学习

在某些应用中嵌入整个图 G（例如，对有毒分子与无毒分子进行分类、识别异常图）。

<img width="297" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/05f31b64-923f-47ee-912a-7e10ef971d34">

完成图嵌入有几种方法：

> 简单的想法是在图 G 上运行图的节点嵌入技术，然后对图 G 中的节点嵌入求和（或平均）。

> 引入“虚拟节点”来表示图并运行标准图节点嵌入技术。

> 使用匿名游走嵌入。 为了学习图嵌入，枚举所有可能的匿名游走，并记录它们的计数，然后将图表示为这些游走的概率分布。


