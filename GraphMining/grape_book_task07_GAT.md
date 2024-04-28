# 同质图中的注意力网络（Graph Attention Networks，GAT）

## 注意力网络

在图卷积神经网络(GCN)聚合节点的 embedding 表征的时候，每个邻居节点的贡献程度都是相同的。但是这种方法太过死板，因为每个邻居节点对目标节点的重要性是有差异的。

GAT 的提出则解决了 GCN 的缺点，GAT 核心就是对于每个顶点都计算其与邻居节点的注意力系数，通过注意力系数来聚合节点的特征。

### 单头注意力机制

<img width="632" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/dd9afad7-19b5-4763-9554-427b32be9508">

### 多头注意力机制

<img width="632" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/d5e0a093-573e-446d-b1ce-e1a42e3e6c76">




