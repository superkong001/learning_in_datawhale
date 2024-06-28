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

<img width="416" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/584958fc-116d-4751-8e89-10b892fa460e">

 $\text{Q-learning}$ 算法的更新公式：

$$
Q(s_t,a_t) \leftarrow Q(s_t,a_t)+\alpha[r_t+\gamma\max_{a}Q^{\prime}(s_{t+1},a)-Q(s_t,a_t)]
$$

在 $\text{DQN}$ 中，我们用神经网络来近似 $Q$ 函数，引入了额外的网络参数 $\theta$ ：

$$
Q\left(s_{i}, a_{i} ; \theta\right) \leftarrow Q\left(s_{i}, a_{i} ; \theta\right)+\alpha[y_i-Q\left(s_{i}, a_{i} ; \theta\right)]
$$

$\qquad$ 其中 $Q\left(s_{i}, a_{i} ; \theta\right)$ 根据习惯不同也可以写成 $Q_{\theta}(s_{i}, a_{i})$ ，注意到，在理想的收敛情况下，实际的 $Q$ 值应该接近于期望的 $Q$ 值，即希望最小化 $r_t+\gamma\max_{a}Q^{\prime}(s_{t+1},a)$ 和 $Q(s_t,a_t)$ 之间的绝对差值。这个差值也就是 $TD$ 误差，也可以写成损失函数的形式并用梯度下降的方式来求解参数 $\theta$ ：

$$
\begin{aligned}
L(\theta)=\left(y_{i}-Q\left(s_{i}, a_{i} ; \theta\right)\right)^{2} 
\\
\theta_i \leftarrow \theta_i - \alpha \nabla_{\theta_{i}} L_{i}\left(\theta_{i}\right)
\end{aligned}
$$

由于 $\text{DQN}$ 算法也是基于 $TD$ 更新的，因此依然需要判断终止状态，在 $\text{Q-learning}$ 算法中也有同样的操作：

$$
y_i = \begin{cases} r_i & \text {对于终止状态} s_{i} 
\\ r_{i}+\gamma \max_{a^{\prime}} Q\left(s_{i+1}, a^{\prime} ; \theta\right) & \text {对于非终止状态} s_{i} \end{cases}
$$







