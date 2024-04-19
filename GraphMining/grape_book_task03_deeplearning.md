# 神经网络及其基本组成

## 多层感知机

<img width="434" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/1bfc8ef8-d461-4010-9e04-3890a79aeb6f">

<img width="646" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/f8ac639a-4cc4-48af-8bd6-7b8e7739e407">

## 输出层与损失函数

<img width="629" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/a8be190a-7f42-4c4b-9b53-c8b747c49f89">

<img width="633" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/966a83c6-990a-4d98-b38a-ab388b37cacf">

## 模型优化

> 随机梯度下降算法
> Adam 算法

Adam（Adaptive Moment Estimation）是一种自适应学习率的优化算法，结合了动量法和 RMSProp 算法的优点。

<img width="648" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/3cddf35f-ae0e-44cf-a2b3-7d001be386ca">

## 模型的过拟合与欠拟合

> 欠拟合（underfitting）：训练误差和验证误差都很严重，而且它们之间仅有一点差距。 即，如果模型不能降低训练误差，这可能意味着模型过于简单（即表达能力不足）， 无法捕获试图学习的模式。

  当小于最佳的模型复杂度的时候：增加模型复杂度，可以同时降低训练损失和泛化损失。在到达最佳点之前，就是欠拟合的情况。
  
> 过拟合（overfitting）：当我们的训练误差明显低于验证误差时。注意，过拟合并不总是一件坏事。 特别是在深度学习领域，众所周知，最好的预测模型在训练数据上的表现往往比在保留（验证）数据上好得多。 最终，我们通常更关心验证误差，而不是训练误差和验证误差之间的差距。

  当大于最佳的模型复杂度的时候：增加模型复杂度，只能降低训练损失，并且会造成泛化损失的增加。这就是过拟合。

<img width="310" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/feff6717-ad9c-47b0-a5d5-969599abcefd">

常用的缓解过拟合的方法

1. 权重衰退： 它通常也被称为L2正则化。L2正则化线性模型构成经典的岭回归（ridge regression）算法，L1正则化线性回归是统计学中类似的基本模型，通常被称为套索回归（lasso regression）。使用L2范数的原因是对权重向量的大分量施加了巨大的惩罚，使得学习算法偏向于在大量特征上均匀分布权重的模型。在实践中，这可能使它们对单个变量中的观测误差更为稳定。相比之下，L1惩罚会导致模型将权重集中在一小部分特征上，而将其他权重清除为零。
   
   <img width="125" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/9aa90cf2-7fdb-45f1-bcb2-c89223b4bfe9">

<img width="475" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/57f2e94e-a26a-4386-8aa3-29eafcba93fd">

  通常，网络输出层的偏置项不会被正则化。
  
2. 暂退法 Dropout：是一种集成大量深层神经网络的实用的 Bagging 方法。Dropout 可以看作是在每个训练批次中训练一个新的神经网络，每个网络只使用一部分神经元（即 Dropout），然后所有网络的结果集成起来作为最后的预测结果。因此，Dropout 方法可以减少模型复杂度，提高泛化能力。将 Dropout 应用到隐藏层，以p的概率将隐藏单元置为零时，结果可以看作是一个只包含原始神经元子集的网络。

<img width="445" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/60a4ef78-5c91-4329-9ddd-b280e21b1150">

## 前向传播和反向传播

> 前向传播：

<img width="625" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/6b9b9d3e-e9e2-4807-b871-006666654510">

> 反向传播：应用链式法则，依次计算每个中间变量和参数的梯度，计算的顺序与前向传播中执行的顺序相反。

<img width="361" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/0d107c47-ec92-43d0-98f7-890f200499df">



