# 柯尼斯堡七桥问题

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





