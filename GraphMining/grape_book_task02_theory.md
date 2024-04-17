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
