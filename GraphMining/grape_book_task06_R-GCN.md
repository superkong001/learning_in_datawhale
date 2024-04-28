图卷积神经网络（GCN）并不区分各节点之间的关系，所以仅适用于同质图的场景。而现实场景中的图往往是异质的，如知识图谱。关系图卷积神经网络就是为处理异质图所提出的。

# 异质图和知识图谱

## 同质图与异质图

> 同质图：是指图中的所有节点都具备相同的类型，且边也具有相同的性质。如: 朋友圈，网络中的节点都代表个人，且边表示朋友关系。同质图中的神经网络设计只需要聚合单一类型的邻居来更新节点的表示。

> 异质图：是指图中的节点类型或关系类型多于一种。

从数学定义上，异质图可以为定义为 G=(V,E,R,T)，其中：

<img width="326" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/f0be2d29-c7a8-46f9-9c31-1ec1b7073c6d">

有不同类型的节点，如 Drug、Disease 等，也有不同类型的边，如因果关系、从属关系等等。

<img width="445" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/bcb9129b-27f6-46cb-9f03-c6c24d55f3de">

## 知识图谱

大多数知识图谱是异质图。知识图谱包含实体和实体之间的关系，并以三元组的形式存储（<头实体, 关系, 尾实体>，即异质图定义的边）。

知识图谱往往不完整，需要对知识图谱缺失的信息做补全。知识图谱补全有两种任务：链路预测和实体分类。

<img width="389" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/62f9084a-de09-4937-8f5e-64283d2c832c">

如：知道吐鲁番葡萄种植于新疆，根据这个信息可以推断出吐鲁番葡萄有“水果”或者“植物”的标签或属性，同时知识图谱有一个三元组 <新疆, 属于, 中国>，通过这些信息可以推断出吐鲁番葡萄和中国之间的关系是产品和来源国（<吐鲁番葡萄, 产自, 中国>）。即：通过处理知识图谱中各实体之间不同关系的交互预测出节点的标签或实体与实体之间的链路信息。

Schlichtkrull 等人于 2017 年底提出了关系图卷积神经网络。

# 关系图卷积神经网络 (Relational Graph Convolutional Networks, R-GCN) 

将一个复杂的异质图解耦为多个单一关系下的同构图，然后只需要解决不同关系下的同构图之间的交互，就可以套用同构图的方法去解决异质图的问题。

## 图卷积神经网络

<img width="391" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/29eb33bd-96de-451c-94f9-00c25d2755a3">

<img width="635" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/bd67258c-85d7-40e5-af7a-9aa48b461e00">

## 关系图卷积神经网络

<img width="630" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/2c8bfc3d-3544-4df3-9d3d-050d19022525">

自环关系被认为是同一种关系类型，也共享一个权重。 R-GCN 的层间递推关系为：

<img width="641" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/1b7afa84-8533-452a-872f-d121491b24b1">

如果边是有向的，边的方向也可以作为一种关系类型。如：

<img width="308" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/335062ce-81d3-4b82-aee2-e4db600ad0cf">

1. 对相同关系的边聚合（蓝色）邻居节点得到 d 维向量。
2. 针对每种类型的边（入边、出边和自环）单独进行权重的学习，生成的（绿色）嵌入向量以归一化求和的形式累积，并通过激活函数（如 ReLU ）向前传播。

这里每个节点的更新可以与整个图中的共享参数并行计算。

<img width="322" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/902c224d-e74a-4287-84b5-1a89278242ae">

<img width="517" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/9290c55d-ae0d-4902-b9d4-454ca9dbd04e">

模型容易边过拟合。

## 可学习参数正则化

R-GCN 提出了两种方案减少模型的参数：基底分解（Basis Decomposition）和 块对角矩阵分解（Block Diagonal Decomposition）

### 基底分解（Basis Decomposition）

<img width="637" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/25772de5-7d19-4961-b5e8-9c7b317045b1">

### 块对角矩阵分解（Block Diagonal Decomposition）

<img width="637" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/f19e155c-0410-47d9-87b7-91cd045f17e0">

参考代码：

原论文中 R-GCN 被应用于知识图谱补全里的链路预测和实体分类任务，关系图卷积网络 - DGL文档

> https://docs.dgl.ai/en/latest/tutorials/models/1_gnn/4_rgcn.html

> https://github.com/thiviyanT/torch-rgcn
