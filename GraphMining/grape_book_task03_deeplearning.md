# 神经网络及其基本组成

## 多层感知机

多层感知机包含输入层，隐藏层和输出层，它们由多层神经元组成， 每一层与它的上一层相连。

<img width="434" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/1bfc8ef8-d461-4010-9e04-3890a79aeb6f">

<img width="646" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/f8ac639a-4cc4-48af-8bd6-7b8e7739e407">

ReLU: 
优点：解决了梯度消失的问题；计算成本低，函数比较简单。

缺点：会产生Dead Neurons，因此当x<0的时候梯度就变为0，这些神经元就变得无效，这些神经元在后面的训练过程就不会更新。

> LeakyReLU

<img width="398" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/7167996d-58b3-49d9-bbc5-8f1a483db3a0">

优点：是ReLU函数的一个变体，解决了ReLU函数存在的问题，α的默认往往是非常小的，比如0.01，这样就保证了Dead Neurons的问题

缺点：由于它具有线性特性，不能用于复杂的分类问题。

## 输出层与损失函数

<img width="629" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/a8be190a-7f42-4c4b-9b53-c8b747c49f89">

<img width="633" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/966a83c6-990a-4d98-b38a-ab388b37cacf">

分子是元素𝑧j的指数，分母是所有指数值的和，𝐾是类别总数。通过这种方式，softmax确保了输出向量中的每个元素都是一个介于0和1之间的数，并且它们的总和为1，使其可以被解释为概率分布。每个元素的概率是该元素的指数与所有元素指数之和的比值，因此得分高的类别会被赋予更高的概率。

## 模型优化

![image](https://github.com/user-attachments/assets/4dcb78d2-e01d-4418-b6c3-484292b97f34)

## 交叉熵损失函数

<img width="616" alt="image" src="https://github.com/user-attachments/assets/b6381cdc-d982-41b9-bcba-de07b3f29a10">

交叉熵损失函数（Cross-Entropy Loss）的定义如下：

$$
L = -\sum_{i} y_i \log p_i
$$

其中：
- $( y_i \)$ 是实际标签（通常是one-hot编码的形式）。
- $( p_i \)$ 是模型预测的概率。

交叉熵损失函数中包含 $𝑦_𝑖$ 项，是为了确保只计算实际类别的预测概率（即：-log正确类别预测概率），表达预测类别与真实类别之间的差异，从而更有效地指导模型学习并进行参数更新。这样做可以使得模型的训练更有效，最终提高分类的准确性。

<img width="548" alt="image" src="https://github.com/user-attachments/assets/f4be58f8-38a0-4f91-bc66-f789818102b0">

<img width="529" alt="image" src="https://github.com/user-attachments/assets/e683ea58-4ffc-4f16-95a4-8f914f2a321f">

- iteration=step=输入一个mini batch=一次迭代=一步
- epoch=一轮=完整遍历训练集的所有样本一遍

Minibatch SGD:

<img width="520" alt="image" src="https://github.com/user-attachments/assets/6791c092-3bf9-4bcf-a67c-f440cffcdcba">

<img width="773" alt="image" src="https://github.com/user-attachments/assets/a7b524f1-2fba-47be-83f8-4e66c29f2101">

神经网络通过反向传播来求梯度：

<img width="556" alt="image" src="https://github.com/user-attachments/assets/0315ed43-33cf-4222-a812-e2f1ee88bf10">

<img width="563" alt="image" src="https://github.com/user-attachments/assets/023431f3-d867-40c3-86bf-0330d87e2088">

<img width="554" alt="image" src="https://github.com/user-attachments/assets/8deddfc1-3404-4696-bd29-a8aa31e07a53">

> 随机梯度下降算法

<img width="446" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/cdede783-2ebc-4944-ac3b-2f9c6547b6ab">

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

# 卷积神经网络

卷积神经网络是包含卷积层的一类特殊的神经网络。在深度学习中，图像处理的区域检测对象被称为卷积核（convolution kernel）或者滤波器（filter）。

卷积神经网络具有的特性：
> 平移不变性（translation invariance）：不管检测对象出现在图像中的哪个位置，神经网络的前面几层应该对相同的图像区域具有相似的反应，即为“平移不变性”。图像的平移不变性使我们以相同的方式处理局部图像，而不在乎它的位置。

> 局部性（locality）：神经网络的前面几层应该只探索输入图像中的局部区域，而不过度在意图像中相隔较远区域的关系，这就是“局部性”原则。最终，可以聚合这些局部特征，以在整个图像级别进行预测。局部性意味着计算相应的隐藏表示只需一小部分局部图像像素。

<img width="443" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/b8636f56-6571-4989-bb58-309fe05e60dc">

卷积：卷积是当把一个函数“翻转”并移位x时，测量f和g之间的重叠。

## 图像卷积:

<img width="426" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/60cca0d6-c050-4517-beec-eb4a5e534c1a">

输出大小计算：

<img width="516" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/4e56adcb-070e-433e-b9f6-0c2a8fe53f3b">

## 填充（padding）：

<img width="430" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/f79967e1-cf22-4873-ab8a-e3e10f11831e">

输出大小计算：

<img width="548" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/c87bc72e-8f5f-4486-ad4b-5f1f87c58a52">

卷积神经网络中卷积核的高度和宽度通常为奇数，例如1、3、5或7。选择奇数的好处是，保持空间维度的同时，我们可以在顶部和底部填充相同数量的行，在左侧和右侧填充相同数量的列。

## 步幅（stride）：每次滑动元素的数量。

<img width="421" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/b4598fb8-7bec-4d57-8dc5-3c13d492c159">

输出大小计算：

<img width="445" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/7c916e76-6cd1-42f3-8591-3f9bd0862dad">

## 感受野（receptive field）：指在前向传播期间可能影响计算的所有元素（来自所有先前层），即：神经网络中神经元“看到的”输入区域。

<img width="310" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/e9d52f08-1088-4f1f-ac7f-f44f738e291b">

## 池化（pooling）：固定形状窗口遍历的每个位置计算一个输出。目的是降低卷积层对位置的敏感性、降低对空间降采样表示的敏感性。

最后一层的神经元应该对整个输入的全局敏感。通过逐渐聚合信息，生成越来越粗糙的映射，最终实现学习全局表示的目标，同时将卷积图层的所有优势保留在中间层。

此外，当检测较底层的特征时，通常希望这些特征保持某种程度上的平移不变性。而在现实中，随着拍摄角度的移动，任何物体几乎不可能发生在同一像素上。即使用三脚架拍摄一个静止的物体，由于快门的移动而引起的相机振动，可能会使所有物体左右移动一个像素（除了高端相机配备了特殊功能来解决这个问题）。

> 最大池化层（maximum pooling）

> 平均池化层（average pooling）

<img width="243" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/d773661e-f60a-45db-826f-0bcd0aaeba06">

# 循环神经网络

循环神经网络（Recurrent Neural Network，RNN）是一种在序列数据上进行建模的神经网络模型。与传统的前馈神经网络不同，循环神经网络具有循环连接，可以将前面的信息传递到后面的步骤中，从而捕捉到序列数据中的时序关系。

循环神经网络架构：

<img width="645" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/693e3fce-996e-4da2-b486-b951794db364">

由于在当前时间步中，隐状态使用的定义与前一个时间步中使用的定义相同，因此这种计算是循环的（recurrent）。于是基于循环计算的隐状态神经网络被命名为循环神经网络（recurrent neural network）。 在循环神经网络中执行计算的层称为循环层（recurrent layer）。即使在不同的时间步，循环神经网络也总是使用这些模型参数。 因此，循环神经网络的参数开销不会随着时间步的增加而增加。

循环神经网络的经典变体包括长短期记忆网络（Long Short-Term Memory，LSTM）和门控循环单元（Gated Recurrent Unit，GRU）。这些变体通过引入门控机制来解决传统循环神经网络中的梯度消失和梯度爆炸问题，从而改善了模型的长期依赖建模能力。在进行反向传播的时候，使用通过时间反向传播（Backpropagation Through Time, BPTT）。它通过将时间展开的 RNN 视为深度前馈神经网络，并在每个时间步骤上应用标准的反向传播算法来更新模型的权重。

## 扩展阅读

线性分类、损失函数、梯度下降：https://www.bilibili.com/video/BV1K7411W7So?p=3

神经网络、计算图、反向传播：https://www.bilibili.com/video/BV1K7411W7So?p=4

训练神经网络（权重初始化、激活函数、BN）：https://www.bilibili.com/video/BV1K7411W7So?p=7

训练神经网络（优化器、正则化、学习率策略）：https://www.bilibili.com/video/BV1K7411W7So?p=8

超有趣的神经网络可视化工具Tensorflow-Playground：https://www.bilibili.com/video/BV15J411u7Ly

卷积神经网络：公众号 人工智能小技巧 回复 卷积神经网络

循环神经网络：https://www.bilibili.com/video/BV1K7411W7So?p=17

生成对抗神经网络GAN：https://www.bilibili.com/video/BV1oi4y1m7np



