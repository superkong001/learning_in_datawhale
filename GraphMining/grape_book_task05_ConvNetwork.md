- https://distill.pub/2021/understanding-gnns
- https://distill.pub/2021/gnn-intro

工具库：
- networkx.org
- www.pyg.org
- www.dgl.ai

summary: 
> 建模图神经网络时，关注重点是如何在网络上构建聚合算子，聚合算子的目的是刻画节点的局部结构。主要方法：谱域方法和空间域方法。

> 由于网络数据上平移不变性的缺失，GCN通过在节点的邻域进行聚合操作来提取特征，而不是简单的平移。

> 谱域方法：傅里叶变换，使用拉普拉斯特征向量矩阵。拉普拉斯谱分解：L：=D-A。但基于全图的傅里叶卷积来实现图的卷积，存在无法保证局部性和计算复杂度比较大，难以扩展到大型图网络结构中的问题，所以采用切比雪夫多项式替代了谱卷积神经网络的卷积核，并对其进行了简化，只取一阶和二阶。

> 空间域方法：空域卷积是从邻居节点信息聚合的角度出发，更加注重节点的局域环境。主要做对节点的信息进行转换和信息聚合。如：GraphSAGE

# GCN

## 问题

<img width="648" alt="image" src="https://github.com/user-attachments/assets/f99467e1-14e0-479e-9c8d-23fa4f4f4641">

- 将邻接矩阵和节点属性直接输入神经网络导致的问题：

<img width="699" alt="image" src="https://github.com/user-attachments/assets/6add36cb-57cb-4834-b57e-e350ffb472a0">

节点的嵌入表示结果与节点编号顺序有关（需要置换不变性）

- 图直接输入卷积网络的问题：

<img width="742" alt="image" src="https://github.com/user-attachments/assets/8faefb03-984d-46e0-9b46-84f1a135142b">

## 思路

使用消息传递图神经网络（MP-GNN，message passing GNN）。

通过局部邻域构建计算图。每个节点的embedding都是基于邻居通过共享的同一个神经网络汇总出来，从而与顺序无关。

<img width="551" alt="image" src="https://github.com/user-attachments/assets/a6aa3eef-2921-4dc6-a1d4-e230d06e9e7a">

<img width="649" alt="image" src="https://github.com/user-attachments/assets/86266be7-1e68-43d7-b80d-cdd36a17c452">

<img width="702" alt="image" src="https://github.com/user-attachments/assets/cfdf9e11-0c7a-4671-9635-aead4ad97a98">

tips: 图神经网络的层数是计算图的层数而不是神经网络的层数。

输入的各个节点的特征属性逐项求平均/求和/最大值（聚合函数与节点顺序无关），然后输入神经网络：

<img width="742" alt="image" src="https://github.com/user-attachments/assets/da426eff-b9b8-4c81-8d37-85f870e0e0f9">

K层KNN感受野：一层图神经网络代表1跳的邻居，K-layer GNN代表K-hop neighborhood，深度过深会导致所有节点embedding都一样

<img width="764" alt="image" src="https://github.com/user-attachments/assets/df7f508f-9270-49ed-8662-19f856d6efcb">

<img width="792" alt="image" src="https://github.com/user-attachments/assets/83c5b85c-7c44-4368-bcef-c7b9ffb4b951">

## 数学形式表达

- 0 层就是输入每个节点给定的属性特征向量，可以One-hot或者全1表示没有特征
- k+1 层为k层所有邻居节点逐元素求和，再取平均；然后输入神经网络；最后经过一个激活函数；
- 最后一层输出的向量作为节点v的Embedding

<img width="575" alt="image" src="https://github.com/user-attachments/assets/3b26e882-fd10-456c-a5af-c03a87bbdad0">

## 矩阵方式计算

<img width="539" alt="image" src="https://github.com/user-attachments/assets/786ec2a2-a9fa-499d-8aa7-655845545ae4">

## 改进

<img width="747" alt="image" src="https://github.com/user-attachments/assets/3c24a99e-9f11-47a6-a91b-4bfd31cfe899">

<img width="755" alt="image" src="https://github.com/user-attachments/assets/f1420240-7f89-46cc-b633-1f5d3d53b91d">

<img width="442" alt="image" src="https://github.com/user-attachments/assets/52fc8582-7ab9-47d7-890b-bea1e1574719">

https://math.stackexchange.com/questions/3035968/interpretation-of-symmetric-normalised-graph-adjacency-matrix

<img width="272" alt="image" src="https://github.com/user-attachments/assets/15189361-ef83-45a3-9d89-2559ce95f2c1">

<img width="571" alt="image" src="https://github.com/user-attachments/assets/3c6093eb-f977-4c5d-975b-ce7f1826e8a1">

最终公式： 

$$
H^{(l+1)} = \sigma\left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}\right)
$$

加上自己的“声音”

<img width="772" alt="image" src="https://github.com/user-attachments/assets/c8d2d068-fbe2-4a5e-9ca7-bc9e24db0c2b">

《图神经网给 基础、刖冶与应用》人氏邮电出版社异步社区

<img width="719" alt="image" src="https://github.com/user-attachments/assets/f21c980f-4a20-441d-b4e8-0e0d901ddcab">

公式4.6=公式4.7，公式4.7为自己邻域节点信息汇聚计算+对自己节点信息汇聚计算

改进版：领域与自身采用不同的神经网络

<img width="791" alt="image" src="https://github.com/user-attachments/assets/4e9ed0a1-97b4-4bfc-a59c-bad06bea815e">

<img width="800" alt="image" src="https://github.com/user-attachments/assets/6e3a461c-3a84-4bea-8b26-27ec4b201951">

## 训练GNN

<img width="701" alt="image" src="https://github.com/user-attachments/assets/3d33bf72-a2c6-4b29-b30d-fe40d6140772">

### 监督训练：

<img width="742" alt="image" src="https://github.com/user-attachments/assets/ff663776-0e50-4fb3-8d86-24294d413de0">

<img width="805" alt="image" src="https://github.com/user-attachments/assets/e86692a0-aaf0-49bf-ac6e-680bbda5190e">

### 无监督训练：
目标：相连节点Embedding向量尽可能相似（点乘，余弦相似度），相似则数量积越大越好

<img width="712" alt="image" src="https://github.com/user-attachments/assets/eb1e6555-b878-4445-bb78-cb6e92e2748b">

直推式学习(Transductive Learning)：

- 用于预测的节点在训练时就见过，无法立刻泛化到新加入的节点，某种程度的过合。如随机游走方法:DeepWalk、Node2Vec，需要重新采样随机游走序列，优化更新所有节点嵌入向量。
- DeepWalk and node2vec do not capture structural similarity. 解决方案:Anonymous Randon Walks、GNN
- 仅利用图本身的连接信息，没利用属性信息或标注信息。

归纳式学习(Inductive Learning)：

- 用于预测的节点在训练时没见过(泛化到新节点)，还能获得功能结构角色信息（结构上相似、地理上远隔）。如图神经网络方法:GCN、GraphSAGE、GAT、GIN
- 利用属性信息或标注信息。节点、连接、子图、全图都可以有特征。

<img width="695" alt="image" src="https://github.com/user-attachments/assets/f3881266-2c3d-40f8-803f-efc11318c0fc">

CNN的卷积核权重需学习得到，而GCN的卷积核权重由 $$<sup>~A$$ 预定义

# 笔记：

现有图神经网络皆基于邻居聚合的框架，即为每个目标节点通过聚合其邻居刻画结构信息，进而学习目标节点的表示。

因此，在建模图神经网络时，关注重点是如何在网络上构建聚合算子，聚合算子的目的是刻画节点的局部结构。

现有的聚合算子构建分为两类：

> 谱域方法：利用图上卷积定理从谱域定义图卷积。图卷积，通过捕获节点和边之间的关系，实现对图数据的有效处理。

> 空间域方法：从节点域出发，通过在节点层面定义聚合函数来聚合每个中心节点和其邻近节点。主要包括空域图卷积神经网络、GraphSAGE和图注意力网络。

# 谱域图卷积神经网络

主要包括谱卷积神经网络、切比雪夫网络和图卷积神经网络。

## 谱图理论和图卷积

网络数据上平移不变性的缺失：在图卷积神经网络（GCN）中，网络数据上的平移不变性缺失指的是传统卷积神经网络（CNN）中对于空间数据（如图像）能够不受具体位置的影响进行特征识别的能力，在GCN中不再适用。这是因为GCN处理的数据是图结构，图结构数据本质上没有像欧式空间中那样固定的、规则的几何排列，每个节点的邻居数量和排列顺序可以大相径庭，因此不能直接应用传统的平移不变性概念。GCN通过在节点的邻域进行聚合操作来提取特征，而这种操作本质上是对节点及其邻居的信息进行整合，而不是简单的平移。

### 卷积的傅里叶变换

傅里叶变换:

数学意义: 傅里叶变换将一个任意的周期函数分解成为无穷个正弦函数的和的形式

物理效果: 傅里叶变换实现了将信号从空间域到频率域的转换

信号分析: 一维傅里叶变换（将杂乱的信号由时域转化到频域中）一维傅里叶变化是将信号分解为正弦波的和的形式

<img width="245" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/71fd31d5-1187-4c07-b146-c202edd15b98">

傅里叶变换的两个作用：

（1）将周期函数（没有周期性可以当成周期为无穷大）拆解为sinx与cosx的组合

（2）将时域信号转为频域信号

<img width="640" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/bdd6be69-c2de-409d-9585-e0c16ce2732d">

<img width="482" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/af0b0885-9882-4799-a90b-7ce4ac738d2d">

频域是时域整体的表达。频域上信号的一个点，对应的是整个时域信号该对应频率的信息。因此，在频域中的乘法，自然就对应了时域整段所有不同频率信号乘法的叠加，这就是卷积。

傅里叶变换只能将周期函数表示为正弦函数与余弦函数的叠加，拉普拉斯变换是加强版的傅里叶变换，追加了对无限增长的函数的表示方法。

### 图傅里叶变换

图傅里叶变换就像传统的傅里叶变换将一个波信号分解成它的组成频率一样，图傅里叶变换在图结构数据进行操作，揭示嵌入其中的信号的频率。

拉普拉斯算子和拉普拉斯矩阵： > https://zhuanlan.zhihu.com/p/67336297/

<img width="531" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/4fe77d73-8f9e-4abf-b472-85bca9465784">

<img width="476" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/f4967f98-cd83-4f0c-be1a-10798aa6d4c0">

1. 拉普拉斯矩阵是半正定矩阵；
  
2. 特征值中0出现的次数就是图连通区域的个数；
   
3. 最小特征值是0，因为拉普拉斯矩阵每一行的和均为0；
   
4. 最小非零特征值是图的代数连通度。

<img width="473" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/04e52407-35e8-438d-a129-7875451132a9">

<img width="530" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/8a4d39db-bac5-4592-8749-088fe0547dcf">

<img width="638" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/fd04353d-fc24-481e-b18b-14e53acd9b31">

傅里叶变换将信号投影到一组正余弦基函数上，而这里的转换则是将信号投影到了图拉普拉斯矩阵的特征向量上。在傅里叶变换中，信号在正余弦基函数上的投影给出了它的频域表示，而在图信号处理中，信号在特征向量上的投影给出了它在特征向量空间中的表示。

U 是正交矩阵，意味着其列向量是标准正交基，即：

<img width="174" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/2cf7de3d-648f-4c03-80ba-ffc3fca2460a">

原始数据 𝑥 在特征向量基底（basis，是一组向量，通过这组向量可以表达空间内所有的点）下的表示，指的是将数据 𝑥 从它原来的坐标系（通常是标准正交基底）转换到由特征向量构成的新的坐标系下的表示。即：原始信号𝑥变换到一个新的坐标系，这个新坐标系是由矩阵 𝐿 的特征向量构成。

特征向量基底（eigenvector basis）是一组特殊的基底，其中每个基底向量都是矩阵的特征向量。特征向量是满足下列等式的非零向量：
<img width="56" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/d120aa35-e993-4dd4-a3f1-7a5a0e178596">

其中 𝐴 是一个矩阵，𝜆 是对应的特征值，𝑣 是特征向量。将数据在特征向量基底下表达时，实际上是在测量数据在每个特征方向上的分量或坐标。特征向量基底的选择通常是基于矩阵的谱分解或奇异值分解，这些特征向量具有捕捉数据方差最大方向的性质，也就是说，数据在这些方向上有最大的扩散或变化。

## 图卷积

先将图进行傅里叶变化，在谱域完成卷积操作，然后再将频域信号转换回原域。

<img width="548" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/fd84c6b8-bbcd-4535-8527-4f7fbce7f36c">

## 谱卷积神经网络

<img width="405" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/a97def9d-81e5-45bc-a6ae-bb7b32f9a9ad">

σ 表示非线性激活函数。

## 切比雪夫网络(ChebyNet)

谱卷积神经网络基于全图的傅里叶卷积来实现图的卷积，其缺点非常明显，难以从卷积形式中保证节点的信息更新由其邻居节点贡献，因此无法保证局部性。另外，谱卷积神经网络的计算复杂度比较大，难以扩展到大型图网络结构中。

切比雪夫网络 ，采用切比雪夫多项式替代了谱卷积神经网络的卷积核，有效的解决了上述的问题。

<img width="630" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/3a2e9b35-b005-4774-8b68-b0e3b8976514">

## 图卷积神经网络

GCN论文：Semi-Supervised Classification with Graph Convolutional Networks, ICLR 2017

图卷积神经网络（Graph Convolutional Network, GCN）对切比雪夫网络进行了简化，只取 0 阶和 1 阶：

<img width="537" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/74230623-4084-49b0-85ee-bb97848ad32d">

<img width="606" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/03290d34-be6d-4b4e-b96f-76cfe2c0654a">

# 空间域图卷积神经网络

空域卷积是从邻居节点信息聚合的角度出发，更加注重节点的局域环境。

## 图卷积神经网络的空域理解

从邻居节点信息聚合的角度出发。图卷积神经网络，应该做的如下两件事情:

> 对节点的信息进行转换（Message Transformation）

> 对节点信息进行聚合 （Message Aggregation）

<img width="635" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/48ad1162-96cc-4baa-8e1c-5b98ac55d7d1">

<img width="435" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/a2264d2c-b96c-4e32-bade-5f7345c7871b">

上面的等式可以矩阵化处理，用于计算整个图的一层表示。考虑到度矩阵 𝐷 和邻接矩阵 𝐴，公式可以改写成：

<img width="639" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/76f86307-cc6a-48a8-a9bc-124778319b0b">

通过这种方式，GCN能够在保持图结构信息的同时学习节点特征的有效表示。

<img width="641" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/556d25ce-671b-4528-bbb7-12befb145ead">

这个时候，如果认为 W 也包含自身节点的操作的话，就可以得到和谱域 GCN 完全一样的公式了。这里，分开表达他们是为了强调，一定要对邻居和自身节点都做信息变换和聚合。

## 空域图卷积的统一范式和 GraphSAGE

### 图卷积的统一范式

从空域图卷积神经网络必须做的两件事情出发，可以得到一个统一范式的图卷积网络。

> 对节点的信息进行转换（Message Transformation）

> 对节点信息进行聚合 （Message Aggregation）

<img width="630" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/b70d9cf6-1167-4601-914e-696924771d8b">

根据图注意力网络（Graph Attention Network, GAT）。可以看出来 GCN 在进行聚合的时候是没有考虑边的权重的而当作 1 进行简单的加和。GAT 的目的就是通过网络来学习边的权重，然后把学到的权重用于聚合。具体地，将边两端得节点送入一个网络，学习输出得到这条边得权重。

### GraphSAGE （Sample aggregate for Graph）

它从两个方面对传统的 GCN 做了改进：

1. 在训练时，采样方式将 GCN 的全图采样优化到部分以节点为中心的邻居抽样，这使得大规模图数据的分布式训练成为可能，并且使得网络可以学习没有见过的节点，这也使得 GraphSAGE 可以做 Inductive Learning。
2. GraphSAGE 研究了若干种邻居聚合的方式，及其 AGG 聚合函数可以使用
> 平均
> Max Pooling
> LSTM

在 GraphSAGE 之前的 GCN 模型中，都是采用的全图的训练方式，也就是说每一轮的迭代都要对全图的节点进行更新，当图的规模很大时，这种训练方式无疑是很耗时甚至无法更新的。mini-batch 的训练时深度学习一个非常重要的特点，那么能否将 mini-batch 的思想用到 GraphSAGE 中呢，GraphSAGE 提出了一个解决方案。它的流程大致分为3步：

<img width="627" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/c5a9f792-f155-43e4-9cad-5d8fbb10f7e7">

拓展参考：
随机游走的艺术-图嵌入表示学习【斯坦福CS224W图机器学习】：

https://www.bilibili.com/video/BV1AP4y1r7Pz/?from_spmid=main.my-history.0.0&plat_id=411&share_from=season&share_medium=android&share_plat=android&share_session_id=238f193d-3639-4a6e-8fde-85b2ea18516b&share_source=WEIXIN&share_tag=s_i&spmid=united.player-video-detail.0.0&timestamp=1713947803&unique_k=a70gqJA

《深入浅出图神经网络：GNN原理解析》配套代码：https://github.com/FighterLYL/GraphNeuralNetwork

## 扩展阅读GNN

李沐老师推荐的GNN博客1：https://distill.pub/2021/gnn-intro

李沐老师推荐的GNN博客2：https://distill.pub/2021/understanding-gnns

Paperswithcode图任务：https://paperswithcode.com/area/graphs

Cora节点分类数据集上的比分：https://paperswithcode.com/sota/node-classification-on-cora

DIFFPOOL论文：https://arxiv.org/pdf/1806.08804.pdf

《图神经网络 基础、前沿与应用》人民邮电出版社 异步图书社区，部分章节电子版

https://graph-neural-networks.github.io

五折购书（正版全新）：公众号 人工智能小技巧 回复 城堡书

## 扩展阅读GCN

论文主页：https://arxiv.org/abs/1609.02907

官方代码：https://github.com/tkipf/gcn

作者本人写的博客：https://tkipf.github.io/graph-convolutional-networks

博客：https://www.inference.vc/how-powerful-are-graph-convolutions-review-of-kipf-welling-2016-2

博客：https://ireneli.eu/2019/01/08/understanding-graph-convolutional-networks

Normalized Adjacency Matrix推导过程：https://math.stackexchange.com/questions/3035968/interpretation-of-symmetric-normalised-graph-adjacency-matrix

在线编辑LaTex公式网站：https://www.latexlive.com

和D矩阵相关的数学公式LaTex脚本：

```latex
```

$$
{\color{Blue} h_{v}^{(k+1)} }  = {\color{Green} \sigma(} {\color{Red} W_{k}} \sum_{u\in N(v)} \frac{{\color{Blue} h_{u}^{k}} }{|N(v)|}{\color{Green} )} 
$$

$$
D = 
\begin{bmatrix} 
  & d(1) & 0 & 0 & \cdots & 0  &\\  
  & 0 & d(2) & 0 & \cdots & 0   \\
  & 0 & 0 & d(3)  & \cdots & 0 \\
  & \vdots & \vdots & \vdots & \ddots & \vdots\\
  & 0 & 0 & 0 & \cdots & d(n)
\end{bmatrix}
$$

$$
D^{-1} = 
\begin{bmatrix} 
  & \frac{1}{d(1)} & 0 & 0 & \cdots & 0  &\\  
  & 0 & \frac{1}{d(2)} & 0 & \cdots & 0   \\
  & 0 & 0 & \frac{1}{d(3)}  & \cdots & 0 \\
  & \vdots & \vdots & \vdots & \ddots & \vdots\\
  & 0 & 0 & 0 & \cdots & \frac{1}{d(n)}
\end{bmatrix}
$$

$$
D^{\frac{1}{2}} = 
\begin{bmatrix} 
  & \sqrt{d(1)} & 0 & 0 & \cdots & 0  &\\  
  & 0 & \sqrt{d(2)} & 0 & \cdots & 0   \\
  & 0 & 0 & \sqrt{d(3)}  & \cdots & 0 \\
  & \vdots & \vdots & \vdots & \ddots & \vdots\\
  & 0 & 0 & 0 & \cdots & \sqrt{d(n)}
\end{bmatrix}
$$

$$
D^{-\frac{1}{2}} = 
\begin{bmatrix} 
  & \frac{1}{\sqrt{d(1)}} & 0 & 0 & \cdots & 0  &\\  
  & 0 & \frac{1}{\sqrt{d(2)}} & 0 & \cdots & 0   \\
  & 0 & 0 & \frac{1}{\sqrt{d(3)}}  & \cdots & 0 \\
  & \vdots & \vdots & \vdots & \ddots & \vdots\\
  & 0 & 0 & 0 & \cdots & \frac{1}{\sqrt{d(n)}}
\end{bmatrix}
$$
