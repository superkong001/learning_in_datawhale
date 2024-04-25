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


