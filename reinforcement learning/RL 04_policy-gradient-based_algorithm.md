# 深度确定性策略梯度算法（deep deterministic policy gradient，DDPG）

## OU(Ornstein-Uhlenbeck)噪声

DDPG算法中使用是 $\text{OU}$ 噪声。它是一种具有回归特性的随机过程，其与高斯噪声相比的优点在于：
* **探索性**： $\text{OU}$ 噪声具有持续的、自相关的特性。
* **控制幅度**： $\text{OU}$ 噪声可以通过调整其参数来控制噪声的幅度。
* **稳定性**： $\text{OU}$ 噪声的回归特性使得噪声在训练过程中具有一定的稳定性。
* **可控性**：由于 $\text{OU}$ 噪声具有回归特性，它在训练过程中逐渐回归到均值，

OU噪声主要由两个部分组成：随机高斯噪声和回归项：

$$
d x_t=\theta\left(\mu-x_t\right) d t+\sigma d W_t
$$

$\qquad$ 其中 $x_t$ 是OU过程在时间 $t$ 的值，即当前的噪声值，这个 $t$ 也是强化学习中的时步（ $\text{time step}$ ）。 $\mu$ 是回归到的均值，表示噪声在长时间尺度上的平均值。 $\theta$ 是 $\text{OU}$ 过程的回归速率，表示噪声向均值回归的速率。 $\sigma$ 是 $\text{OU}$ 过程的扰动项，表示随机高斯噪声的标准差。 $dW_t$ 是布朗运动（ $\text{Brownian motion}$ ）或者维纳过程（ $\text{Wiener process}$ ），是一个随机项，表示随机高斯噪声的微小变化。 $\qquad$ 在实际应用中，只需要调整 $\mu$ 和 $\sigma$ 就可以了， $\theta$ 通常是固定的，而 $dW_t$ 是随机项，不需要关注。

## DDPG

DDPG是一种确定性的策略梯度算法。由于DQN算法的一个主要缺点就是不能用于连续动作空间，因此要适配连续动作空间，就将选择动作的过程变成一个直接从状态映射到具体动作的函数 $\mu_\theta (s)$ ，其中 $\theta$ 表示模型的参数，这样就把求解 $Q$ 函数、贪心选择动作这两个过程合并成了一个函数Actor。 $Q(s,a)$ 函数有两个变量的，相当于一个曲线平面，当输入某个状态到 $\text{Actor}$ 时，即固定 $s=s_t$ 时，则相当于把曲线平面截断成一条曲线。而Actor的任务就是寻找这条曲线的最高点，并返回对应的横坐标，即最大 $Q$ 值对应的动作，所以DDPG是在寻找最大值。

<img width="591" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/c3a56242-545e-475a-b35b-8e76a7acf83f">

在强化学习基础算法的研究改进当中主要基础核心问题：
1) 如何提高对值函数的估计，保证其准确性，即尽量无偏且低方差；
2) 如何提高探索以及平衡探索-利用的问题，尤其在探索性比较差的确定性策略中。

DDPG特点：
适用但只适用于连续动作空间、高效的梯度优化、经验回放和目标网络、高度依赖超参数、高度敏感的初始条件、容易陷入局部最优。

# TD3(twin delayed DDPG, 双延迟确定性策略梯度算法)算法

TD3算法的改进主要做了三点重要的改进：
1) 双 $Q$ 网络（名字中 twin）。在DDPG算法中的 $\text{Critic}$ 网络上再加一层，形成两个 $\text{Critic}$ 网络，分别记为 $Q_{\omega_1}$ 和 $Q_{\omega_2}$ ，在计算 $\text{TD}$ 误差的时候，就可以取两个 $Q$ 值中较小的那个。
2) 延迟更新（名字中 delayed）。在训练中让Actor的更新频率低于Critic的更新频率，如：Critic每更新10次，Actor只更新1次。
3) 躁声正则（noise regularisation），即：目标策略平滑正则化（ $\text{Target Policy Smoothing Regularization}$ ）。主要思想是通过提高Critic的更新频率来减少值函数的估计误差，也就是“降低领导决策的失误率”。







