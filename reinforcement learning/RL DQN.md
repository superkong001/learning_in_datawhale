DQN算法，英文全称 Deep Q-Network, 其主要是在Q-learning算法的基础上引入了深度神经网络来近似动作价值函数 $Q(\boldsymbol{s},\boldsymbol{a})$，从而能够处理高维的状态空间。【论文： Human-level control through deep reinforcement learning[J].Nature, 2015. 】

在深度学习中，一个神经网络能够将输入向量x映射到输出向量y，这个映射过程可以用以下公式表示，它的输入输出都是向量，并且拥有可以学习的参数 $\theta$，这些参数可以通过梯度下降的方式来优化，从而使得神经网络能够逼近任意函数。

$$
\boldsymbol{y} = f_{\theta}(\boldsymbol{x})
$$


