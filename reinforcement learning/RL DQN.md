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

## DQN

 $\text{Q-learning}$ 算法的更新公式：

$$
Q(s_t,a_t) \leftarrow Q(s_t,a_t)+\alpha[r_t+\gamma\max_{a}Q^{\prime}(s_{t+1},a)-Q(s_t,a_t)]
$$

在 $\text{DQN}$ 中，用神经网络来近似 $Q$ 函数，引入了额外的网络参数 $\theta$ ：

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
\\ 
r_{i}+\gamma \max_{a^{\prime}} Q\left(s_{i+1}, a^{\prime} ; \theta\right) & \text {对于非终止状态} s_{i} \end{cases}
$$

## 强化学习 vs 深度学习

> 强化学习用于训练的样本（包括状态、动作和奖励等等）是与环境实时交互得到的，深度学习则是事先准备好的。
> 在Q-learning 算法中，每次交互得到一个样本之后，就立即更新模型，导致训练的不稳定，从而影响模型的收敛（深度学习一般是小批量梯度下降）。
> 每次迭代的样本都是从环境中实时交互得到的，样本是有关联的，而深度学习训练集是事先准备好的，每次迭代的样本都是从训练集中随机抽取的。（梯度下降法是基于训练集中的样本是独立同分布的假设）

## DQN经验回放

<img width="425" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/51cf15a2-fced-483a-9b46-66b944a01f66">

DQN中会把每次与环境交互得到的样本都存储在一个经验回放中，然后每次从经验池中随机抽取一批样本来训练网络。经验回放的容量不能太小，太小了会导致收集到的样本具有一定的局限性，也不能太大，太大了会失去经验本身的意义。

## DQN目标网络参数更新

<img width="377" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/e17c71f5-2bf6-4dc2-a55c-fd9c6486c330">

每隔若干步更新目标网络：目标网络和当前网络结构都是相同的，都用于近似 $Q$ 值，在实践中每隔若干步才把每步更新的当前网络参数复制给目标网络，这样做的好处是保证训练的稳定，避免 $Q$ 值的估计发散。

同时在计算损失函数的时候，使用的是目标网络来计算 $Q$ 的期望值，如式 $\text(7.3)$ 所示。

$$
Q_{期望} = [r_t+\gamma\max _{a^{\prime}}Q_{\bar{\theta}}(s^{\prime},a^{\prime})]
$$

目标网络的作用，比如当前有个小批量样本导致模型对 $Q$ 值进行了较差的过估计，如果接下来从经验回放中提取到的样本正好连续几个都这样的，很有可能导致 $Q$ 值的发散。




