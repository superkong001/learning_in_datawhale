# DQN算法
DQN算法，英文全称 Deep Q-Network, 其主要是在Q-learning算法的基础上引入了深度神经网络来近似动作价值函数 $Q(\boldsymbol{s},\boldsymbol{a})$ 来替代Q表，从而能够处理高维的状态空间。【论文： Human-level control through deep reinforcement learning[J].Nature, 2015. 】

在深度学习中，一个神经网络能够将输入向量x映射到输出向量y，这个映射过程可以用以下公式表示，它的输入输出都是向量，并且拥有可以学习的参数 $\theta$，这些参数可以通过梯度下降的方式来优化，从而使得神经网络能够逼近任意函数。

$$
\boldsymbol{y} = f_{\theta}(\boldsymbol{x})
$$

动作价值函数 $Q(\boldsymbol{s},\boldsymbol{a})$，即将状态向量 $\boldsymbol{s}$ 作为输入，并输出所有动作 $\boldsymbol{a} = (a_1,a_2,...,a_n)$ 对应的价值，如以下所示：

$$
\boldsymbol{y} = Q_{\theta}(\boldsymbol{s},\boldsymbol{a})
$$

## Q表 vs 神经网络
> Q表只能处理离散的状态和动作空间，而DQN神经网络则可以处理连续的状态和动作空间（把每个维度的坐标看作一个输入）。
> Q表中描述状态空间的时候一般用的是状态个数，而在DQN神经网络中用的是状态维度。

输出的都是每个动作对应的Q值，即预测，而不是直接输出动作。要输出动作，就需要额外做一些处理，例如结合贪心算法选择Q值最大对应的动作等，这就是控制过程。
