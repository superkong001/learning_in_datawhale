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

傅里叶变换只能将周期函数表示为正弦函数与余弦函数的叠加，拉普拉斯变换是加强版的傅里叶变换，追加了对无限增长的函数的表示方法。

### 图傅里叶变换

图傅里叶变换就像传统的傅里叶变换将一个波信号分解成它的组成频率一样，图傅里叶变换在图结构数据进行操作，揭示嵌入其中的信号的频率。

<img width="531" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/4fe77d73-8f9e-4abf-b472-85bca9465784">

<img width="476" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/f4967f98-cd83-4f0c-be1a-10798aa6d4c0">

1. 拉普拉斯矩阵是半正定矩阵；
  
2. 特征值中0出现的次数就是图连通区域的个数；
   
3. 最小特征值是0，因为拉普拉斯矩阵每一行的和均为0；
   
4. 最小非零特征值是图的代数连通度。

<img width="473" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/04e52407-35e8-438d-a129-7875451132a9">

<img width="530" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/8a4d39db-bac5-4592-8749-088fe0547dcf">

<img width="638" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/fd04353d-fc24-481e-b18b-14e53acd9b31">

U 是正交矩阵，意味着其列向量是标准正交基，即：

<img width="174" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/2cf7de3d-648f-4c03-80ba-ffc3fca2460a">

原始数据 𝑥 在特征向量基底（basis，是一组向量，通过这组向量可以表达空间内所有的点）下的表示，指的是将数据 𝑥 从它原来的坐标系（通常是标准正交基底）转换到由特征向量构成的新的坐标系下的表示。即：原始信号𝑥变换到一个新的坐标系，这个新坐标系是由矩阵 𝐿 的特征向量构成。

特征向量基底（eigenvector basis）是一组特殊的基底，其中每个基底向量都是矩阵的特征向量。特征向量是满足下列等式的非零向量：
<img width="56" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/d120aa35-e993-4dd4-a3f1-7a5a0e178596">

其中 𝐴 是一个矩阵，𝜆 是对应的特征值，𝑣 是特征向量。将数据在特征向量基底下表达时，实际上是在测量数据在每个特征方向上的分量或坐标。特征向量基底的选择通常是基于矩阵的谱分解或奇异值分解，这些特征向量具有捕捉数据方差最大方向的性质，也就是说，数据在这些方向上有最大的扩散或变化。

## 图卷积

先将图进行傅里叶变化，在谱域完成卷积操作，然后再将频域信号转换回原域。




