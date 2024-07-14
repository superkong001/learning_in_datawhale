<img width="683" alt="image" src="https://github.com/user-attachments/assets/9e031c4b-9fbc-498f-9e3b-9229e43458bc"># 柯尼斯堡七桥问题

![image](https://github.com/superkong001/learning_in_datawhale/assets/37318654/6e7e26ea-2d0e-45c7-864c-a63308c852cb)

若从某点出发后最后再回到这点，则这⼀点的线数必须是偶数，这样的点称为偶顶点。相对的，连有奇数条线的点称为奇顶点。

由于柯尼斯堡七桥问题中存在4个奇顶点，它⽆法实现符合题意的遍历。

欧拉把问题的实质归于⼀笔画问题，即判断⼀个图是否能够遍历完所有的边⽽没有重复，⽽柯尼斯堡七桥问题则是⼀笔画问题的⼀个具体情境。

对于⼀个给定的连通图，如果存在超过两个的奇顶点，那么满⾜要求的路线便不存在了，且有n个奇顶点的图⾄少需要⌈n/2⌉笔画出。如果只有两个奇顶点，则可从其中任何⼀地出发完成⼀笔画。若所有点均为偶顶点，则从任何⼀点出发，所求的路线都能实现。

# networkx
```bash
# 导入 networkx 包
# https://networkx.github.io/
import networkx as nx
import matplotlib.pyplot as plt

默认情况下，networkX 创建的是⽆向图
G = nx.Graph()
print(G.is_directed())

# 创建⼀个空⼿道俱乐部⽹络
G = nx.karate_club_graph()
```

<img width="830" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/b24f19cc-631a-447c-af39-8d63d6143f9c">

```bash
# ⽹络平均度的计算
def average_degree(num_edges, num_nodes):
     # this function takes number of edges and number of nodes
     # returns the average node degree of the graph. 
     # Round the result to nearest integer (for example 3.3 will be rounded to 3 and 3.7 will be rounded to 4)
     avg_degree = 0
     #########################################
     avg_degree = 2*num_edges/num_nodes
     avg_degree = int(round(avg_degree))
     #########################################
     return avg_degree

num_edges = G.number_of_edges()
num_nodes = G.number_of_nodes()
avg_degree = average_degree(num_edges, num_nodes)
print("Average degree of karate club network is {}".format(avg_degree))
```

<img width="797" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/6620791e-2d45-4feb-bacf-0e93b0c62b27">

<img width="662" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/013ae480-8cf1-4f72-9a97-7c7114dda4e2">

```bash
# 聚类系数（Clustering Coefficient）
def average_clustering_coefficient(G):
     # this function that takes a nx.Graph
     # and returns the average clustering coefficient. 
     # Round the result to 2 decimal places (for example 3.333 will be rounded to 3.33 and 3.7571 will be rounded to 3.76)
     avg_cluster_coef = 0
     #########################################
     ## Note: 
     ## 1: Please use the appropriate NetworkX clustering function
     avg_cluster_coef = nx.average_clustering(G)
     avg_cluster_coef = round(avg_cluster_coef, 2)
     #########################################
     return avg_cluster_coef

avg_cluster_coef = average_clustering_coefficient(G)
print("Average clustering coefficient of karate club network is {}".format(avg_cluster_coef))
```

<img width="653" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/b79f9714-033a-4757-979c-1fb409fd3113">

```bash
# 接近中⼼度 (closeness centrality)
def closeness_centrality(G, node=5):
     # the function that calculates closeness centrality 
     # for a node in karate club network. G is the input karate club 
     # network and node is the node id in the graph. Please round the 
     # closeness centrality result to 2 decimal places.
     closeness = 0
     #########################################
     # Raw version following above equation
     # source: https://stackoverflow.com/questions/31764515/find-all-nodesconnected-to-n
     path_length_total = 0
     for path in list(nx.single_source_shortest_path(G,node).values())[1:]:
         path_length_total += len(path)-1
            
     closeness = 1 / path_length_total
     closeness = round(closeness, 2)
     return closeness

node = 5
closeness = closeness_centrality(G, node=node)
print("The karate club network has closeness centrality (raw) {:.2f}".format(closeness))
```

<img width="662" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/5564ce03-f94c-4db1-b327-5360d661586b">

<img width="666" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/1edff2dc-0b02-4759-93fd-2976b34a7034">
<img width="617" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/94fcb519-5dbc-49b9-b0a7-9b3790ec2353">

<img width="621" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/cf9e79e9-eae8-463f-aad0-ddb7345b8c3f">

<img width="659" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/bb736c1e-135e-4514-bdf7-b23874438cd2">

> 同质图（Homogeneous Graph）：只有⼀种类型的节点和⼀种类型的边的图。
> 异质图（Heterogeneous Graph）：存在多种类型的节点和多种类型的边的图。

<img width="368" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/68ce2a27-fc9d-482c-acf9-88cf0bf88977">

```bash
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite

# 创建一个二分图
B = nx.Graph()

# 添加具有节点属性 "bipartite" 的节点
top_nodes = [1, 2, 3, 4]
bottom_nodes = ["a", "b", "c"]
B.add_nodes_from(top_nodes, bipartite=0)  # Label top set as 0
B.add_nodes_from(bottom_nodes, bipartite=1)  # Label bottom set as 1

# 添加边，仅在不同集合的节点间
B.add_edges_from([(1, "a"), (1, "b"), (2, "b"), (2, "c"), (3, "c"), (4, "a")])

# 使用二分图布局
pos = nx.bipartite_layout(B, top_nodes)

# 绘制图形
plt.figure(figsize=(8, 4))
# 绘制节点
nx.draw_networkx_nodes(B, pos, nodelist=top_nodes, node_color='lightblue', node_size=500, label='Group 1')
nx.draw_networkx_nodes(B, pos, nodelist=bottom_nodes, node_color='lightgreen', node_size=500, label='Group 2')
# 绘制边
nx.draw_networkx_edges(B, pos)
# 绘制节点标签
nx.draw_networkx_labels(B, pos)
# 添加图例
plt.legend(loc='upper left')
# 显示图
plt.title("Bipartite Graph")
plt.show()
```

图问题类型：

1. 节点层面**
2. 社群层面
3. 连接层面
4. 全图层面

<img width="598" alt="image" src="https://github.com/user-attachments/assets/b95ae8f0-0cf5-436e-8fde-fc9af733520c">

# PageRank

改变世界的十大IT论文: 图灵机、香农信息论、维纳控制论、比特币白皮书、PageRank、AlexNet、ResNet、Alpha Go、Transformer、AlphaFold2

现在网页挑战：
- 随时生成的（如：随时生成的分享链接、推荐链接）
- 不可触达，私域（微信朋友圈、聊天记录。。。）

无标度网络（Scale-Free Network）指的是某些网络中节点的度（即与节点相连的边的数量）分布呈幂律分布（Power-Law Distribution）。这种类型的网络在许多自然和人造系统中都可以找到，如互联网、社交网络、蛋白质相互作用网络、引文网络等。

通过链接结构来计算每个节点重要度。

变种算法：

- PageRank
- Personalized PageRank (PPR)
- Random Walk with Restarts

## PageRank

idea: 
- 以指向改网页的链接作为投票 In-coming links as votes
- 投票数越高的网页投票权重越高 Links from important pages count more
- 递归问题

<img width="557" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/49e72520-c8bf-4d22-ae46-22315f480cc4">

PageRank 算法：

### 迭代求解线性方程组

高斯消元法的基本目标是通过行变换将线性方程组的系数矩阵转化为行最简阶梯形矩阵（或上三角形矩阵），从而简化方程组的求解过程。

所有节点重要度相加=1

<img width="562" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/585c75fb-ccb4-4281-a093-db4569a382fe">

初始化每个节点重要度，然后逐步迭代计算

<img width="553" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/65e044a9-216c-45af-9dd9-fc246ad828fb">

### 迭代左乘M矩阵

<img width="665" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/80c4c847-25fb-42ea-abc1-4199744425f2">

列是概率，求和等于1

<img width="590" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/baaf71af-2417-4e2e-8a66-1b15392e9016">

<img width="486" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/039793b2-7bb7-40c7-bf6f-c326f7b1600c">

矩阵的线性变换可以有多种几何解释，例如旋转、缩放、剪切、投影等。这些变换可以改变向量的方向和大小，但不能平移（原点不变），变化总是以一种线性的方式进行，即不会引入非线性扭曲（如曲线或扭曲路径）。

### 特征向量

线性变化后方向不变只是进行长度缩放

<img width="314" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/a730991c-2629-4c65-aa1a-de61a92fcb74">

因此，r是特征值为1的特征向量，实际就是通过幂迭代求主特征向量

<img width="547" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/8a4f0954-0318-48a5-b3a4-9bb3f216d272">

对于Column Stochastic矩阵,由Perreon-Frobenius定理：最大的特征值为1，存在唯一的主特征向量(特征值1对应的特征向量)，向量所有元素求和为1。

### 随机游走(connection to Random Walk)

变成低维、连续、稠密的向量

<img width="558" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/cfab597a-5ef2-4fc3-ba41-1f3dce9a4b0c">

pagerank值就是随机游走走到某个节点的概率

<img width="555" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/15961c91-aefd-4ac0-860c-a860a0460b6e">

### 马尔科夫链

状态与状态之间的转移来表示，节点表示状态，连接表示状态转移。

<img width="406" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/225bcfdd-291f-44c3-8ca2-65a214f694d6">

PageRank <=> Stationary Distributions of Markov Chains

> 迭代求解线性方程组($O(n^3)$,不推荐)

> 迭代左乘M矩阵(推荐,幂迭代)

> 矩阵的特征向量($O(n^3)$,不推荐)

> 随机游走(需模拟很多游走,不推荐)

> 马尔科夫链(和求解矩阵特征向量，不推荐)

### 求解PageRank

<img width="520" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/5844236e-2d3c-4a04-afd3-ba087ddce9c2">

暴力求解：收敛条件L1一阶用绝对值，L2二阶可以用平方和，一般迭代50次就收敛了

<img width="553" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/1988b7bb-fc79-4215-ab05-b5b5c7d52bef">

### PageRank收敛性分析

> 是否能收敛至稳定值?(而不是发散)
> 不同初始值,是否收敛至同一个结果?
> 收敛的结果在数学上是否有意义?
> 收敛的结果是否真的代表节点重要度?

相关定理：

<img width="593" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/1c5c745c-f78e-412b-8089-429b1d875d25">

可归约马尔科夫链（Reducible Markov Chain） ：如果马尔科夫链中的某些状态不能从其他状态到达，或者某些状态无法到达其他状态，那么这个链就是可归约的。这意味着整个状态空间可以被分割成不相连的子集，每个子集内部的状态可以互相到达，但不能到达其他子集的状态。

<img width="424" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/45210248-5676-4ec7-a4a9-c66aa2a041c9">

周期震荡马尔科夫链（Periodic Markov Chain）：是指马尔科夫链中存在某种规律性的循环或周期行为，其中从一个特定状态返回到自身的步数总是周期 𝑑 的倍数，且 
𝑑>1。如果一个状态的周期是1，那么我们称这个状态是非周期的或者说这个链是非周期的。周期性在马尔科夫链的分析中非常重要，因为它影响到链的长期行为和收敛特性。

<img width="419" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/95584472-3909-4067-a0b5-b919bf006d1f">

### PageRank 问题

- 多个彼此独立的连通域
- (1) Some pages are dead ends 即：死胡同 (have no out-links)
  违背收敛基本假设，全是0，解决方案：改写某些列不全为0，百分百被传走
  
  <img width="502" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/ee90abdf-69b3-48e8-bad1-5d0131b6e0b0">

- (2) Spider traps 即：仅指向自己 (all out-links are within the group)
  符合收敛基本假设，但结果无意义，其他全是0，解决方案：以1-ß概率随机跳到其他节点

<img width="557" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/54c4bc65-eb43-42e6-b6b5-e5aab651abc6">

<img width="555" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/02767016-05bf-43c9-9f96-e62d6f528971">

summary:

<img width="557" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/3fde0ac4-d0c4-4b44-a7b4-0ea118acf001">

<img width="557" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/a6637d2d-e638-42b6-b0b4-5ff90d00fb53">

eg:

<img width="580" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/aed79b06-cd64-463d-85ad-a49ab7887684">

### PageRank 扩展应用

1） 推荐系统：寻找与指定节点最相似的节点（基本假设：被同一用户访问过的节点更可能是相似的）

<img width="543" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/45204c12-7b3e-4ffd-b1b0-bc8ceaf229db">

<img width="525" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/75d7ba47-032b-46fd-922a-3bf826b0fd6e">

如计算Q最相似节点，就是以Q为起点模拟多次随机游走，看访问次数最多的节点是哪个？

<img width="587" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/38444160-d28a-4424-867e-30e866aba817">

<img width="541" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/64b2347a-9c2a-4c4a-821c-f0a3e99fb8b4">

# 半监督节点分类：标签传播和消息传递

Transductive 直推式学习 而不是 Inductive 归纳式学习

<img width="530" alt="image" src="https://github.com/user-attachments/assets/c1d8b87e-2ba3-409e-90ad-c740f22a75ac">

<img width="517" alt="image" src="https://github.com/user-attachments/assets/2f018fd8-ef0c-40e2-9852-671c646b7f99">

<img width="521" alt="image" src="https://github.com/user-attachments/assets/4956e07d-702f-4e62-b1eb-f5b7e296712c">

## 半监督节点分类问题-求解方法对比

| 方法               | 图嵌入 | 表示学习 | 使用属性特征 | 使用标注 | 直推式 | 归纳式 |
| ------------------ | ------ | -------- | ------------ | -------- | ------ | ------ |
| 人工特征工程       | 是     | 否       | 否           | 否       | /      | /      |
| 基于随机游走的方法 | 是     | 是       | 否           | 否       | 是     | 否     |
| 基于矩阵分解的方法 | 是     | 是       | 否           | 否       | 是     | 否     |
| 标签传播           | 否     | 否       | 是/否        | 是       | 是     | 否     |
| 图神经网络         | 是     | 是       | 是           | 是       | 是     | 是     |

- 人工特征工程：节点重要度、集群系数、Graphlet等。

- 基于随机游走的方法，构造自监督表示学习任务实现图嵌入。无法泛化到新节点。

  例如：DeepWalk、Node2Vec、LINE、SDNE等。

- 标签传播：假设“物以类聚，人以群分”，利用邻域节点类别猜测当前节点类别。无法泛化到新节点。

  例如：Label Propagation、Iterative Classification、Belief Propagation、Correct & Smooth等。

- 图神经网络：利用深度学习和神经网络，构造邻域节点信息聚合计算图，实现节点嵌入和类别预测。

  可泛化到新节点。

  例如：GCN、GraphSAGE、GAT、GIN等。

标签传播和集体分类：

1. Label Propagation(Relational Classification)
2. terative Classification
3. Correct & Smooth
4. Belief Propagation
5. Masked Lable Prediction

## Relational Classification

<img width="627" alt="image" src="https://github.com/user-attachments/assets/2d174e1a-7a91-4d07-b24c-1b3055970e77">

<img width="632" alt="image" src="https://github.com/user-attachments/assets/47803694-6bc6-4f21-acad-7b3e0238a54a">

<img width="629" alt="image" src="https://github.com/user-attachments/assets/98133985-662a-4430-9e54-cbf9b658db34">

<img width="558" alt="image" src="https://github.com/user-attachments/assets/3c92e89b-b4f5-408c-9d9a-38908b2a2879">

<img width="559" alt="image" src="https://github.com/user-attachments/assets/319fb4c0-4057-444f-a9cd-a947304f60a9">

不收敛原因：线性代数知识，特征值[-1,1]的话不断左乘邻接矩阵会收敛，其他会发散。

<img width="518" alt="image" src="https://github.com/user-attachments/assets/b83cd1ec-aa70-4118-8cdb-ea104950e4ac">

## Iterative Classification

Relational Classification 不使用节点特征。而Iterative Classification：
Main idea of iterative classification: Classify node $v$ based on its attributes $f_v$ as well as labels $Z_v$ of neighbor set $N_v$ . 

<img width="552" alt="image" src="https://github.com/user-attachments/assets/7182264a-47f4-418d-870e-08963312925a">

<img width="578" alt="image" src="https://github.com/user-attachments/assets/2f8cdf38-29e7-47f9-9d94-4ee9975fac7c">

Eg: Web Page Classification

1. Train classifiers
2. Apply classifier to unlab. set

<img width="566" alt="image" src="https://github.com/user-attachments/assets/b6e5197e-4c03-43b8-9b4d-84cbd94b19be">

<img width="577" alt="image" src="https://github.com/user-attachments/assets/84ec3213-884a-4aac-a532-533f13e6d055">

3. Iterate

   4. Update relational features $z_v$

<img width="588" alt="image" src="https://github.com/user-attachments/assets/abfe932a-7f51-4f65-bd4b-d29fa25b7efe">

   5. Update label $Y_v$  

<img width="640" alt="image" src="https://github.com/user-attachments/assets/72ef3924-37d8-41c0-ab50-bc7dcccfac90">

   根据 $Y_2$ 再更新 $z_v$ 
   
<img width="616" alt="image" src="https://github.com/user-attachments/assets/03760096-deaf-4219-8d4f-69d72edab5a6">

   再更新 $Y_3$ 
   
<img width="604" alt="image" src="https://github.com/user-attachments/assets/72dff135-325e-4d22-9cc4-6f88d773e672">

## C&S (Correct & Smooth)

C&S uses graph structure to post-process the soft node labels predicted by any base model. C&S achieves strong performance on semi-supervised node classification.

> https://ogb.stanford.edu

<img width="700" alt="image" src="https://github.com/user-attachments/assets/f87aeaea-8e6f-4799-ac0c-17cd8b994385">

C&S follows the three-step procedure:
1. Train base predictor
2. Use the base predictor to predict soft labels of all nodes
3. Post-process the predictions using graph structure to obtain the final predictions of all nodes.

   (1) Train a base predictor that predict soft labels (class probabilities) over all nodes.

   - Labeled nodes are used for train/validation data.
   - Base predictor can be simple: Linear model/Multi-Layer-Perceptron(MLP) over node features

<img width="768" alt="image" src="https://github.com/user-attachments/assets/8a9862d0-7c22-4030-ac4f-670447ef31f7">

   (2) Given a trained base predictor, we apply it to obtain soft labels for all the nodes.

    - We expect these soft labels to be decently(体面) accurate.
    - Can we use graph structure to post-process(后处理) the predictions to make them more accurate?

<img width="574" alt="image" src="https://github.com/user-attachments/assets/f1427ed6-d379-403a-a70f-2fcf8921c56a">

   (3) C&S uses the 2-step procedure to post-process the soft predictions.

<img width="766" alt="image" src="https://github.com/user-attachments/assets/5aa6962f-b97e-4090-b0b5-b5cd90fe4d5e">

     1) Correct step: Diffuse and correct for the training errors of the base predictor.    

<img width="751" alt="image" src="https://github.com/user-attachments/assets/f01c4869-52b3-46f7-a5ae-e9ff4402206f">

<img width="718" alt="image" src="https://github.com/user-attachments/assets/330fe12d-3b56-4b14-9324-cbc8c2cc16e0">

<img width="767" alt="image" src="https://github.com/user-attachments/assets/96186ae8-0d7b-440a-8568-7ea56ae67c06">

<img width="759" alt="image" src="https://github.com/user-attachments/assets/2ff10b1f-b110-42e5-8b9f-2cdbe9e57ae3">

<img width="681" alt="image" src="https://github.com/user-attachments/assets/f5f124ca-4d89-4026-8732-f906fe55a1ff">

<img width="767" alt="image" src="https://github.com/user-attachments/assets/82294561-32ac-4763-93c8-493af0a8b99e">

<img width="760" alt="image" src="https://github.com/user-attachments/assets/e73bfe7f-606f-41e3-a333-479f0bb70ea2">

<img width="786" alt="image" src="https://github.com/user-attachments/assets/15538999-d010-4243-9538-691e50a51a98">

<img width="805" alt="image" src="https://github.com/user-attachments/assets/661f89cc-d913-4397-b0e6-4278bb51fd45">

     2) Smooth step: Smoothen the prediction of the base predictor.

<img width="725" alt="image" src="https://github.com/user-attachments/assets/b2cc9037-0eea-4fd1-b662-5e41e4e816c7">

<img width="714" alt="image" src="https://github.com/user-attachments/assets/14092c20-4de6-4abe-acbf-39effc5d07bd">

<img width="775" alt="image" src="https://github.com/user-attachments/assets/3cf9addc-5529-4cdc-8016-c974a00d417d">

<img width="778" alt="image" src="https://github.com/user-attachments/assets/5ec37a1d-3db7-4bf5-82ed-0249fb31013b">

参考资料：
- 斯坦福CS224W图机器学习 https://web.stanford.edu/class/cs224w
- PageRank:A Trillion Dollar Algorithm(作者:Reducible) https://www.youtube.com/watch?v=JGQe4kiPnrU
- Google 's PageRank Algorithm(作者:Global Software Support) https://www.youtube.com/playlist?list=PLH7W8KdUX6P2n4XwDiKsEU6sBhQj5cAqa
- 曼彻斯特大学:https://personalpages.manchester.ac.uk/staff/yanghong.huang/teaching/MATH36032/pagerank.pdf
- 斯坦福CS345 Data Mining: https://wenku.baidu.com/view/5be822bfbb4cf7ec4bfed052.html?_wkts_= 1669773903123&bdQuery=web+in+1839

参考视频
- https://www.youtube.com/watch?v=meonLcN7LD4
- https://www.youtube.com/watch?v=P8Kt6Abq_rM&list=PLH7W8KdUX6P2n4XwDiKsEU6sBhQj5cAqa&index=4
- PageRank与马尔科夫链:https://www.youtube.com/watch?v=JGQe4kiPnrU

其它参考资料
- 《数学之美》
- 得到APP:张潇雨·商业经典案例课-谷歌
- 得到APP:吴军·谷歌方法论
- PageRank原始论文1:http://infolab.stanford.edu/~backrub/google.html
- PageRank原始论文2:http://ilpubs.stanford.edu:8090/422

