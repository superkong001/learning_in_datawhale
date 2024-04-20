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

 
𝑃(𝑣∣Z𝑢)表示给定节点𝑢的特征向量Z𝑢时，节点𝑣被选为𝑢的邻居的条件概率。分子代表节点𝑢和𝑣之间相似性度量的指数形式；分母是对图中所有可能邻居𝑛的相似性度量进行同样计算并求和。

<img width="635" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/01eb7e54-304f-404b-b65b-878900a7ccde">


## Node2Vec

