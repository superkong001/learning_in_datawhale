# 同质图中的注意力网络（Graph Attention Networks，GAT）

## 注意力网络

在图卷积神经网络(GCN)聚合节点的 embedding 表征的时候，每个邻居节点的贡献程度都是相同的。但是这种方法太过死板，因为每个邻居节点对目标节点的重要性是有差异的。

GAT 的提出则解决了 GCN 的缺点，GAT 核心就是对于每个顶点都计算其与邻居节点的注意力系数，通过注意力系数来聚合节点的特征。

### 单头注意力机制

<img width="632" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/dd9afad7-19b5-4763-9554-427b32be9508">

### 多头注意力机制

<img width="632" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/d5e0a093-573e-446d-b1ce-e1a42e3e6c76">

# 异质图中的图注意力网络（Heterogeneous Graph Attention Network，HAN）

在将处理同质图的注意力方法扩展到异质图的时候，最重要的一步就是如何确定异质图的节点的邻居节点。

## 异质图注意力网络

异质图注意力网络使用了元路径（Meta Path）来确定异质图中的每个节点的邻居节点。

> 元路径：认为是一种具有一定语意信息的构图方法，因为在异质图中有非常复杂的节点之间的联系，但是这种联系并不全是有效的，所以通过定义元路径来定义一些有意义的连接方式。节点 i 在通过元路径生成的图中的邻居就是依据元路径定义的邻居（Meta-path based Neighbors）。

<img width="482" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/dba2a228-b2a4-412b-afb9-e763fb3014d9">

如图中的（a）演员、电影、导演组成的（b）异质图，它的（c）元路径有 Movie-Actor-Movie (MAM)，Movie-Director-Movie (MDM)，而根据这些元路径，可以得到其（d）邻居。

> 异质图注意力网络整体架构

将注意力机制从同质图扩展到节点和边有不同类型的异质图。异质图注意力网络包含：

（a）节点级注意力：目的是学习节点与基于元路径的邻居节点之间的重要性。
（b）语义级注意力：目的是学习不同元路径的重要性。
（c）通过接 MLP 输出节点 i 的预测 <img width="13" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/68f77096-2513-4cf2-a5b5-c9d52f4a40d7">。


