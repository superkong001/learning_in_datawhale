策略梯度算法是直接对策略本身进行近似优化，优化目标也基本是每回合的累积奖励期望，即通常讲的回报 $G$（return）。 它将策略描述成一个带有参数 $\theta$ 的连续函数，该函数将某个状态作为输入，输出的不再是某个确定性（ $\text{deterministic}$ ）的离散动作，而是对应的动作概率分布，通常用 $\pi_{\theta}(a|s)$ 表示，称作随机性（ $\text{stochastic}$ ）策略。

基于价值算法的缺点：
1. 无法表示连续动作，如：机器人的运动控制问题。因为DQN等算法是通过学习状态和动作的价值函数来间接指导策略，因此它们只能处理离散动作空间的问题。
2. 高方差，影响算法的收敛性。
3. 不能很好地平衡探索与利用的关系。DQN等算法在实现时通常选择贪心的确定性策略，而很多问题的最优策略是随机策略，即需要以不同的概率选择不同的动作。

# 策略梯度算法

<img width="594" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/db90d957-b21f-44b7-a00d-710c41e13a85">

从开始状态到终止状态为一个**回合**（ $\text{episode}$ ），然后把所有的状态和动作按顺序组合起来，记作 $\tau$ ，称为**轨迹**（ $\text{trajectory}$ ）：

$$
\tau=\left(s_0, a_0, s_1, a_1, \cdots, s_T, a_T\right)
$$

其中 $T$ 表示回合的终止时刻。对于任意轨迹 $\tau$ ，其产生的概率公式：

$$
\begin{aligned}
P_{\theta}(\tau)
&=p(s_{0}) \pi_{\theta}(a_{0} | s_{0}) p(s_{1} | s_{0}, a_{0}) \pi_{\theta}(a_{1} | s_{1}) p(s_{2} | s_{1}, a_{1}) \cdots \\
&=p(s_{0}) \prod_{t=0}^{T} \pi_{\theta}\left(a_{t} | s_{t}\right) p\left(s_{t+1} | s_{t}, a_{t}\right)
\end{aligned}
$$

因为优化目标也是每回合的累积奖励期望，即通常讲的回报 $G$（return），则累积奖励就可以计算为 $R(\tau)=\sum_{t=0}^T r\left(s_t, a_t\right)$ 。

<img width="610" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/528280cb-e8d8-4805-b143-b6c41aee999d">

策略的价值期望公式：

$$
\begin{aligned}
J(\pi_{\theta}) = \underset{\tau \sim \pi_\theta}{E}[R(\tau)] 
& = P_{\theta}(\tau_{1})R(\tau_{1})+P_{\theta}(\tau_{2})R(\tau_{2})+\cdots \\
&=\int_\tau P_{\theta}(\tau) R(\tau) \\ 
&=E_{\tau \sim P_\theta(\tau)}[\sum_t r(s_t, a_t)] 
\end{aligned}
$$

推导：

$$
\nabla_\theta P_{\theta}(\tau)= P_{\theta}(\tau) \frac{\nabla_\theta P_{\theta}(\tau)}{P_{\theta}(\tau) }= P_{\theta}(\tau) \nabla_\theta \log P_{\theta}(\tau)
$$

$$
\log P_{\theta}(\tau)= \log p(s_{0})  +  \sum_{t=0}^T(\log \pi_{\theta}(a_t \mid s_t)+\log p(s_{t+1} \mid s_t,a_t))
$$

$$
\begin{aligned}
\nabla_\theta \log P_{\theta}(\tau) &=\nabla_\theta \log \rho_0\left(s_0\right)+\sum_{t=0}^T\left(\nabla_\theta \log \pi_\theta\left(a_t \mid s_t\right)+\nabla_\theta \log p\left(s_{t+1} \mid s_t, a_t\right)\right) \\
&=0+\sum_{t=0}^T\left(\nabla_\theta \log \pi_\theta\left(a_t \mid s_t\right)+0\right) \\
&=\sum_{t=0}^T \nabla_\theta \log \pi_\theta\left(a_t \mid s_t\right)
\end{aligned}
$$

期望可以被视为连续随机变量的加权平均，权重由随机变量的概率分布给出，最终推导：

$$
\begin{aligned}
\nabla_\theta J\left(\pi_\theta\right) &=\nabla_\theta \underset{\tau \sim \pi_\theta}{\mathrm{E}}[R(\tau)] \\
&=\nabla_\theta \int_\tau P_{\theta}(\tau) R(\tau) \\
&=\int_\tau \nabla_\theta P_{\theta}(\tau) R(\tau) \\
&=\int_\tau P_{\theta}(\tau) \nabla_\theta \log P_{\theta}(\tau) R(\tau) \\
&=\underset{\tau \sim \pi_\theta}{\mathrm{E}}\left[\nabla_\theta \log P_{\theta}(\tau) R(\tau)\right]\\
&= \underset{\tau \sim \pi_\theta}{\mathrm{E}}\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta\left(a_t \mid s_t\right) R(\tau)\right]
\end{aligned}
$$

<div align=center>
<img width="439" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/85c5974d-3160-42c2-86a1-4f5b9f3bb805">
</div>

### 蒙特卡罗方法(REINFORCE算法)

蒙特卡罗方法可以粗略地分成两类：一类是所求解的问题本身具有内在的随机性，借助计算机的运算能力可以直接模拟这种随机的过程。
另一种类型是所求解问题可以转化为某种随机分布的特征数，比如随机事件出现的概率，或者随机变量的期望值。通过随机抽样的方法，以随机事件出现的频率估计其概率，或者以抽样的数字特征估算随机变量的数字特征，并将其作为问题的解。这种方法多用于求解复杂的多维积分问题。

常见应用：
1) 近似计算积分：非权重蒙特卡罗积分，也称确定性抽样，是对被积函数变量区间进行随机均匀抽样，然后对抽样点的函数值求平均，从而可以得到函数积分的近似值。此种方法的正确性是基于概率论的中心极限定理。
2) 近似计算圆周率：让计算机每次随机生成两个0到1之间的数，看以这两个实数为横纵坐标的点是否在单位圆内。生成一系列随机点，统计单位圆内的点数与总点数，（圆面积和正方形面积之比为PI:4，PI为圆周率），当随机点获取越多时，其结果越接近于圆周率（然而准确度仍有争议：即使取10的9次方个随机点时，其结果也仅在前4位与圆周率吻合）。

### 平稳分布

马尔可夫链的平稳分布（stationary distribution）。平稳分布，是熵增原理的一种体现，顾名思义就是指在无外界干扰的情况下，系统长期运行之后其状态分布会趋于一个固定的分布，不再随时间变化。稳定的分布并乘以对应的价值，就可以计算所谓的长期收益。

马尔科夫链中，满足连通性，那么它一定满足细致平稳性，反之亦然。
1) 连通性（connectedness）：在处于平稳分布下，任意两个状态之间都是互相连通的，即任意两个状态之间都可以通过一定的步骤到达。
2) 细致平稳（detailed balance）：任意状态在平稳分布下的概率都是一样的，即任意状态的概率都是相等的。

对于任意马尔可夫链，如果满足以下两个条件：

* 非周期性：由于马尔可夫链需要收敛，那么就一定不能是周期性的，实际上处理的问题基本上都是非周期性的。
* 状态连通性：即存在概率转移矩阵 $P$ ，能够使得任意状态 $s_0$ 经过有限次转移到达状态 $s$ ，反之亦然。

$\qquad$ 这样就可以得出结论，即该马氏链一定存在一个平稳分布，用 $d^{\pi}(s)$ 表示，可得到：

$$
d^\pi(s)=\lim_{t \rightarrow \infty} P\left(s_t=s \mid s_0, \pi_\theta\right)
$$

$\qquad$ 换句话说，对于一个特定的环境， $d^{\pi}(s)$ 相当于一个环境本身常量，类似于状态转移概率矩阵，只是在求解马尔可夫过程的时候无法获得，只能通过其他的方法近似。

对于连续动作空间，通常策略对应的动作可以从高斯分布 ${\mathbb{N}}\left(\phi(s)^{\mathbb{T}} \theta , \sigma^2\right)$ ，对应的梯度也可求得：

$$
    \nabla_\theta \log \pi_\theta(s, a)=\frac{\left(a-\phi(s)^T \theta\right) \phi(s)}{\sigma^2}
$$

$\qquad$ 这个公式实现起来只需要在模型最后一层输出两个值，一个是均值，一个是方差，然后再用这两个值来构建一个高斯分布，然后采样即可。

## 策略梯度算法的特点

特指蒙特卡洛策略梯度算法，即 $\text{REINFORCE}$ 算法。 相比于 $\text{DQN}$ 之类的基于价值的算法优点：
1) 适配连续动作空间。
2) 适配随机策略。由于策略梯度算法是基于策略函数的，因此可以适配随机策略，而基于价值的算法则需要一个确定的策略。此外其计算出来的策略梯度是无偏的，而基于价值的算法则是有偏的。

策略梯度算法也有其缺点：
1) 采样效率低。由于使用的是蒙特卡洛估计，与基于价值算法的时序差分估计相比其采样速度必然是要慢很多的。
2) 高方差。虽然跟基于价值的算法一样都会导致高方差，但是策略梯度算法通常是在估计梯度时蒙特卡洛采样引起的高方差，这样的方差甚至比基于价值的算法还要高。
3) 收敛性差。容易陷入局部最优，策略梯度方法并不保证全局最优解，因为它们可能会陷入局部最优点。策略空间可能非常复杂，存在多个局部最优点，因此算法可能会在局部最优点附近停滞。
4) 难以处理高维离散动作空间：对于离散动作空间，采样的效率可能会受到限制，因为对每个动作的采样都需要计算一次策略。当动作空间非常大时，这可能会导致计算成本的急剧增加。

# Actor-Critic 算法


