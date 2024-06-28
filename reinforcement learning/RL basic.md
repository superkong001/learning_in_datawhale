参考： 

> https://datawhalechina.github.io/easy-rl/#/

> 蘑菇书EasyRL

# 强化学习基础

## 强化学习概述

<img width="698" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/d430f8af-dfbe-449e-9a02-2d5e177773f7">

vs监督学习，监督学习两个假设：

1) 输入的数据（标注的数据）都应是没有关联的。因为如果输入的数据有关联，学习器（learner）是不好学习的。通常假设样本空间中全体样本服从一个未知分布，我们获得的每个样本都是独立地从这个分布上采样获得的，即独立同分布(independent and identically distributed，简称 i.i.d.)。
2) 需要告诉学习器正确的标签是什么，这样它可以通过正确的标签来修正自己的预测。

（1）强化学习输入的样本是序列数据，而不像监督学习里面样本都是独立的。

（2）学习器并没有告诉我们每一步正确的动作应该是什么，学习器需要自己去发现哪些动作可以带来 最多的奖励，只能通过不停地尝试来发现最有利的动作。

（3）智能体获得自己能力的过程，其实是不断地试错探索（trial-and-error exploration）的过程。探索 （exploration）和利用（exploitation）是强化学习里面非常核心的问题。其中，探索指尝试一些新的动作， 这些新的动作有可能会使我们得到更多的奖励，也有可能使我们“一无所有”；利用指采取已知的可以获得最多奖励的动作，重复执行这个动作，因为我们知道这样做可以获得一定的奖励。因此，我们需要在探索和利用之间进行权衡，这也是在监督学习里面没有的情况。

（4）在强化学习过程中，没有非常强的监督者（supervisor），只有奖励信号（reward signal），并且奖励信号是延迟的，即环境会在很久以后告诉我们之前我们采取的动作到底是不是有效的。因为我们没有得 到即时反馈，所以智能体使用强化学习来学习就非常困难。当我们采取一个动作后，如果我们使用监督学习，我们就可以立刻获得一个指导，比如，我们现在采取了一个错误的动作，正确的动作应该是什么。而在强化学习里面，环境可能会告诉我们这个动作是错误的，但是它并没有告诉我们正确的动作是什么。而且更困难的是，它可能是在一两分钟过后告诉我们这个动作是错误的。所以这也是强化学习和监督学习不同的地方。

强化学习的一些特征:

（1）强化学习会试错探索，它通过探索环境来获取对环境的理解。

（2）强化学习智能体会从环境里面获得延迟的奖励。

（3）在强化学习的训练过程中，时间非常重要。因为我们得到的是有时间关联的数据（sequential data）， 而不是独立同分布的数据。在机器学习中，如果观测数据有非常强的关联，会使得训练非常不稳定。这也是为什么在监督学习中，我们希望数据尽量满足独立同分布，这样就可以消除数据之间的相关性。

（4）智能体的动作会影响它随后得到的数据，这一点是非常重要的。在训练智能体的过程中，很多时 候我们也是通过正在学习的智能体与环境交互来得到数据的。所以如果在训练过程中，智能体不能保持稳定，就会使我们采集到的数据非常糟糕。我们通过数据来训练智能体，如果数据有问题，整个训练过程就会失败。所以在强化学习里面一个非常重要的问题就是，怎么让智能体的动作一直稳定地提升。

<img width="607" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/f1f4fc55-dd34-471c-959c-7d03df7130dd">

<img width="594" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/49c8ba3b-2e83-48a3-a440-5161ccf9e724">

## 强化学习智能体的组成成分

对于一个强化学习智能体，它可能有一个或多个如下的组成成分。

1) 策略（policy）。智能体会用策略来选取下一步的动作。

  **随机性策略（stochastic policy）**就是 $\pi$ 函数，即 $\pi(a | s)=p\left(a_{t}=a | s_{t}=s\right)$ 。输入一个状态 $s$ ，输出一个概率。 
这个概率是智能体所有动作的概率，然后对这个概率分布进行采样，可得到智能体将采取的动作。比如可能是有 0.7 的概率往左，0.3 的概率往右，那么通过采样就可以得到智能体将采取的动作。

  **确定性策略（deterministic policy）**就是智能体直接采取最有可能的动作，即 $a^{*}=\underset{a}{\arg \max} \pi(a \mid s)$ 。 

通常情况下，强化学习一般使用随机性策略。

2) 价值函数（value function）。用价值函数来对当前状态进行评估。价值函数用于评估智能体进 入某个状态后，可以对后面的奖励带来多大的影响。价值函数值越大，说明智能体进入这个状态越有利。

价值函数的值是对未来奖励的预测，用它来评估状态的好坏。 价值函数里面有一个折扣因子（discount factor），希望在尽可能短的时间里面得到尽可能多的奖励。

<img width="431" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/8bdbcfb1-052b-44bf-89ef-bfc33351cc03">

期望 $\mathbb{E}_{\pi}$ 的下标是 $\pi$ 函数， $\pi$ 函数的值可反映在我们使用策略 $\pi$ 的时候，到底可以得到多少奖励。

还有一种价值函数：Q 函数。Q 函数里面包含两个变量：状态和动作。其定义为:

<img width="425" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/de7462cc-6ff3-4094-a017-3b73cd52da5b">

3) 模型（model）。模型表示智能体对环境的状态进行理解，它决定了环境中世界的运行方式。 下面我们深入了解这 3 个组成部分的细节。

模型决定了下一步的状态。下一步的状态取决于当前的状态以及当前采取的动作。它由状态转移概率和奖励函数两个部分组成。状态转移概率即

$$
p_{s s^{\prime}}^{a}=p\left(s_{t+1}=s^{\prime} \mid s_{t}=s, a_{t}=a\right)
$$

奖励函数是指我们在当前状态采取了某个动作，可以得到多大的奖励，即

$$
R(s,a)=\mathbb{E}\left[r_{t+1} \mid s_{t}=s, a_{t}=a\right]
$$

有了策略、价值函数和模型3个组成部分后，就形成了一个马尔可夫决策过程（Markov decision process）。

<img width="316" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/f71eab2e-470c-43bf-9548-69436d97aedc">

## 强化学习智能体的类型

基于策略的智能体（policy-based agent）直接学习策略，给它一个状态，它就会输出对应动作的概率。基于策略的智能体并没有学习价值函数。

<img width="417" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/624a2a1e-dbe7-4dd7-b79f-03019b665ba5">

基于价值的智能体（value-based agent）显式地学习价值函数，隐式地学习它的策略。策略是其从学到的价值函数里面推算出来的。

<img width="414" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/ad81c33d-0e3a-478d-812d-514e7d8ee3e1">

把基于价值的智能体和基于策略的智能体结合起来就有了**演员-评论员智能体（actor-critic agent）**。这一类智能体把策略和价值函数都学习了，然后通过两者的交互得到最佳的动作。

基于策略的强化学习方法VS基于价值的强化学习方法：

对于一个状态转移概率已知的马尔可夫决策过程，可以使用动态规划算法来求解。决策方式是智能体在给定状态下从动作集合中选择一个动作的依据，它是静态的，不随状态变化而变化。 在基于策略的强化学习方法中，智能体会制定一套动作策略（确定在给定状态下需要采取何种动作），并根据这个策略进行操作。强化学习算法直接对策略进行优化，使制定的策略能够获得最大的奖励。 而在基于价值的强化学习方法中，智能体不需要制定显式的策略，它维护一个价值表格或价值函数，并通过这个价值表格或价值函数来选取价值最大的动作。基于价值迭代的方法只能应用在不连续的、离散的环境下（如围棋或某些游戏领域），对于动作集合规模庞大、动作连续的场景（如机器人控制领域），其很难学习到较好的结果（此时基于策略迭代的方法能够根据设定的策略来选择连续的动作）。 基于价值的强化学习算法有Q学习（Q-learning）、 Sarsa 等，而基于策略的强化学习算法有策略梯度（Policy Gradient，PG）算法等。此外，演员-评论员算法同时使用策略和价值评估来做出决策。其中，智能体会根据策略做出动作，而价值函数会对做出的动作给出价值，这样可以在原有的策略梯度算法的基础上加速学习过程，取得更好的效果。

有模型强化学习智能体VS免模型强化学习智能体：

有模型（model-based）强化学习智能体通过学习状态的转移来采取动作，即增加对真实环境进行建模。 免模型（model-free）强化学习智能体没有去直接估计状态的转移，也没有得到环境的具体转移变量，它通过学习价值函数和策略函数进行决策。免模型强化学习智能体的模型里面没有环境转移的模型。

用马尔可夫决策过程来定义强化学习任务，并将其表示为四元组 <S,A,P,R>，即状态集合、动作集合、状态转移函数和奖励函数。如果这个四元组中所有元素均已知，且状态集合和动作集合在有限步数内是有限集，则智能体可以对真实环境进行建模，构建一个虚拟世界来模拟真实环境中的状态和交互反应。
具体来说，当智能体知道状态转移函数 $P(s_{t+1}|s_t,a_t)$ 和奖励函数 $R(s_t,a_t)$ 后，它就能知道在某一状态下执行某一动作后能带来的奖励和环境的下一状态，这样智能体就不需要在真实环境中采取动作，直接在虚拟世界中学习和规划策略即可。这种学习方法称为**有模型强化学习**。

<img width="413" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/e6352c84-3713-4066-8ab6-748a9ffdda22">

## Gym库

https://www.gymlibrary.dev/environments/classic_control/

<img width="411" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/fd6be7d2-2c43-4612-9b21-509beca9c407">

OpenAI 的 Gym库是一个环境仿真库，里面包含很多现有的环境。针对不同的场景，我们可以选择不同的环境。离散控制场景（输出的动作是可数的，比如Pong游戏中输出的向上或向下动作）一般使用雅达利环境评估；连续控制场景（输出的动作是不可数的，比如机器人走路时不仅有方向，还有角度，角度就是不可数的，是一个连续的量 ）一般使用 MuJoCo 环境评估。Gym Retro是对 Gym 环境的进一步扩展，包含更多的游戏。

```bash
pip install gym==0.25.2 #  Gym 库 0.26.0 及其之后的版本对之前的代码不兼容
pip install pygame # 用于显示图形界面
```

<img width="427" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/5af0cf04-acbc-46f7-8bec-8197467d0a62">

```python
import gym  # 导入 Gym 的 Python 接口环境包
env = gym.make('CartPole-v0')  # 构建实验环境
env.reset()  # 重置一个回合
for _ in range(1000):
    env.render()  # 显示图形界面
    action = env.action_space.sample() # 在该游戏的所有动作空间里随机选择一个作为输出
    observation, reward, done, info = env.step(action)  # 用于提交动作，括号内是具体的动作
    print(observation)
env.close() # 关闭环境
```

env.step()完成了一个完整的 $S \to A \to R \to S'$ 过程。
马尔可夫决策过程来定义强化学习任务，并将其表示为四元组 <S,A,P,R>，即状态集合、动作集合、状态转移函数和奖励函数。

### 小车上山（MountainCar-v0）例子

```python
# 看任务的观测空间和动作空间
# 环境的观测空间用 env.observation_space 表示，动作空间用 env.action_space 表示。
# 离散空间 gym.spaces.Discrete 类表示，连续空间用 gym.spaces.Box 类表示。
# 对于离散空间，Discrete (n) 表示可能取值的数量为 n；对于连续空间，Box类实例成员中的 low 和 high 表示每个浮点数的取值范围。
import gym
env = gym.make('MountainCar-v0')
print('观测空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
print('观测范围 = {} ~ {}'.format(env.observation_space.low,
        env.observation_space.high))
print('动作数 = {}'.format(env.action_space.n))
```

```python
# 智能体来控制小车移动
class SimpleAgent:
    def __init__(self, env):
        pass
    
    def decide(self, observation): # 决策
        position, velocity = observation
        lb = min(-0.09 * (position + 0.25) ** 2 + 0.03,
                0.3 * (position + 0.9) ** 4 - 0.008)
        ub = -0.07 * (position + 0.38) ** 2 + 0.07
        if lb < velocity < ub:
            action = 2
        else:
            action = 0
        return action # 返回动作

    def learn(self, *args): # 学习
        pass

# 智能体与环境交互
'''
env 是环境类。agent 是智能体类。render 是 bool 型变量，其用于判断是否需要图形化显示。如果 render 为 True，则在交互过程中会调用 env.render() 以显示图形界面，通过调用 env.close() 可关闭图形界面。train 是 bool 型变量，其用于判断是否训练智能体，在训练过程中设置为 True，让智能体学习；在测试过程中设置为 False，让智能体保持不变。该函数的返回值 episode_reward 是 float 型的数值，其表示智能体与环境交互一个回合的回合总奖励。
'''
def play(env, agent, render=False, train=False):
    episode_reward = 0. # 记录回合总奖励，初始值为0
    observation = env.reset() # 重置游戏环境，开始新回合
    while True: # 不断循环，直到回合结束
        if render: # 判断是否显示
            env.render() # 显示图形界面
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action) # 执行动作
        episode_reward += reward # 收集回合奖励
        if train: # 判断是否训练智能体
            agent.learn(observation, action, reward, done) # 学习
        if done: # 回合结束，跳出循环
            break
        observation = next_observation
    return episode_reward # 返回回合总奖励
```

```python
# 智能体和环境交互，并显示图形界面
env.seed(3) # 设置随机种子，让结果可复现
agent = SimpleAgent(env)
episode_reward = play(env, agent, render=True)
print('回合奖励 = {}'.format(episode_reward))

# 计算出连续交互 100 回合的平均回合奖励
episode_rewards = [play(env, agent) for _ in range(100)]
print('平均回合奖励 = {}'.format(np.mean(episode_rewards)))

env.close() # 关闭图形界面
```

# 马尔可夫决策过程(MDP)

## 马尔可夫性质（Markov property）

在随机过程中，**马尔可夫性质（Markov property）** 是指一个随机过程在给定现在状态及所有过去状态情况下，其未来状态的条件概率分布仅依赖于当前状态。

$$
p\left(X_{t+1}=x_{t+1} \mid X_{0:t}=x_{0: t}\right)=p\left(X_{t+1}=x_{t+1} \mid X_{t}=x_{t}\right)
$$

其中， $X_{0:t}$ 表示变量集合 $X_{0}, X_{1}, \cdots, X_{t}$,  $x_{0:t}$ 为在状态空间中的状态序列 $x_{0}, x_{1}, \cdots, x_{t}$。
马尔可夫性质也可以描述为给定当前状态时，将来的状态与过去状态是条件独立的。如果某一个过程满足马尔可夫性质，那么未来的转移与过去的是独立的，它只取决于现在。马尔可夫性质是所有马尔可夫过程的基础。

## 马尔可夫链

马尔可夫过程是一组具有马尔可夫性质的随机变量序列 $s_1,\cdots,s_t$ ，其中下一个时刻的状态 $s_{t+1}$ 只取决于当前状态 $s_t$ 。我们设状态的历史为 
 $h_{t}=\left(s_{1}, s_{2}, s_{3}, \ldots, s_{t}\right)$ ( $h_t$ 包含了之前的所有状态)，则马尔可夫过程满足条件：

$$
p\left(s_{t+1} \mid s_{t}\right) =p\left(s_{t+1} \mid h_{t}\right) \tag{2.1}
$$

从当前 $s_t$ 转移到 $s_{t+1}$ ，它是直接就等于它之前所有的状态转移到 $s_{t+1}$ 。

用**状态转移矩阵（state transition matrix）**$\boldsymbol{P}$ 来描述状态转移 $p\left(s_{t+1}=s^{\prime} \mid s_{t}=s\right)$：

$$
  \boldsymbol{P}=\left(\begin{array}{cccc}
    p\left(s_{1} \mid s_{1}\right) & p\left(s_{2} \mid s_{1}\right) & \ldots & p\left(s_{N} \mid s_{1}\right) \\
    p\left(s_{1} \mid s_{2}\right) & p\left(s_{2} \mid s_{2}\right) & \ldots & p\left(s_{N} \mid s_{2}\right) \\
    \vdots & \vdots & \ddots & \vdots \\
    p\left(s_{1} \mid s_{N}\right) & p\left(s_{2} \mid s_{N}\right) & \ldots & p\left(s_{N} \mid s_{N}\right)
    \end{array}\right)
$$

状态转移矩阵类似于条件概率（conditional probability），它表示当我们知道当前我们在状态 $s_t$ 时，到达下面所有状态的概率。

## 马尔可夫奖励过程

马尔可夫奖励过程（Markov reward process, MRP）是马尔可夫链加上奖励函数（reward function）
奖励函数 $R$ 是一个期望，表示当我们到达某一个状态的时候，可以获得多大的奖励。
另外定义了折扣因子 $\gamma$ ,越往后得到的奖励，折扣越多。这说明我们更希望得到现有的奖励，对未来的奖励要打折扣。如果状态数是有限的，那么 $R$ 可以是一个向量。

### 回报与价值函数

回报（return）定义为奖励的逐步叠加，假设时刻$t$后的奖励序列为 $r_{t+1},r_{t+2},r_{t+3},\cdots$ ，则回报为

$$
  G_{t}=r_{t+1}+\gamma r_{t+2}+\gamma^{2} r_{t+3}+\gamma^{3} r_{t+4}+\ldots+\gamma^{T-t-1} r_{T}
$$

其中， $T$ 是最终时刻， $\gamma$ 是折扣因子，越往后得到的奖励，折扣越多。这说明更希望得到现有的奖励，对未来的奖励要打折扣。有了回报之后，就可以定义状态的价值了，就是**状态价值函数（state-value function）**。对于马尔可夫奖励过程，状态价值函数被定义成回报的期望，即

$$
\begin{aligned}
    V^{t}(s) &=\mathbb{E}\left[G_{t} \mid s_{t}=s\right] \\
    &=\mathbb{E}\left[r_{t+1}+\gamma r_{t+2}+\gamma^{2} r_{t+3}+\ldots+\gamma^{T-t-1} r_{T} \mid s_{t}=s\right]
\end{aligned}  
$$

其中， $G_t$ 是之前定义的**折扣回报（discounted return）**。对 $G_t$ 取了一个期望，期望就是从这个状态开始，可能获得多大的价值。所以期望也可以看成未来可能获得奖励的当前价值的表现，就是当进入某一个状态后，现在有多大的价值。

使用折扣因子的原因如下：
1) 有些马尔可夫过程是带环的，它并不会终结，避免无穷的奖励。
2) 并不能建立完美的模拟环境的模型，对未来的评估不一定是准确的，因为这种不确定性，所以对未来的评估增加一个折扣, 希望尽可能快地得到奖励。
3) 如果奖励是有实际价值的，更希望立刻就得到奖励，而不是后面再得到奖励。
4) 更想得到即时奖励。

把折扣因子设为 0( $\gamma=0$ )，就只关注当前的奖励。折扣因子设为 1( $\gamma=1$ ),对未来的奖励并没有打折扣，未来获得的奖励与当前获得的奖励是一样的。

蒙特卡洛（Monte Carlo，MC）采样的方法：从某个状态 $s_4$ 开始，采样生成很多轨迹，把这些轨迹的回报都计算出来，然后将其取平均值作为进入 $s_4$ 的价值。（撒豆子来估计图形面积）


### 贝尔曼方程

$$
V(s)=\underbrace{R(s)}_ {\text {即时奖励}}+\underbrace{\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s\right) V\left(s^{\prime}\right)}_{\text {未来奖励的折扣总和}}
$$

其中，
* $s'$ 可以看成未来的所有状态，
* $p(s'|s)$  是指从当前状态转移到未来状态的概率。
* $V(s')$ 代表的是未来某一个状态的价值。从当前状态开始，有一定的概率去到未来的所有状态，所以要把 $p\left(s^{\prime} \mid s\right)$ 写上去。得到了未来状态后，乘一个 $\gamma$ ，这样就可以把未来的奖励打折扣。
* $\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s\right) V\left(s^{\prime}\right)$ 可以看成未来奖励的折扣总和（discounted sum of future reward）。

贝尔曼方程定义了当前状态与未来状态之间的关系。

$$
  \mathbb{E}[V(s_{t+1})|s_t]=\mathbb{E}[\mathbb{E}[G_{t+1}|s_{t+1}]|s_t]=\mathbb{E}[G_{t+1}|s_t]
$$

全期望公式也被称为叠期望公式（law of iterated expectations，LIE）。
如果 $A_i$ 是样本空间的有限或可数的划分（partition），则全期望公式可定义为

$$
  \mathbb{E}[X]=\sum_{i} \mathbb{E}\left[X \mid A_{i}\right] p\left(A_{i}\right)
$$

如果 $X$ 和 $Y$ 都是离散型随机变量，则条件期望（conditional expectation） $\mathbb{E}[X|Y=y]$ 定义为

$$
  \mathbb{E}[X \mid Y=y]=\sum_{x} x p(X=x \mid Y=y)
$$

贝尔曼方程的推导过程如下：

$$
  \begin{aligned}
    V(s)&=\mathbb{E}\left[G_{t} \mid s_{t}=s\right]\\
    &=\mathbb{E}\left[r_{t+1}+\gamma r_{t+2}+\gamma^{2} r_{t+3}+\ldots \mid s_{t}=s\right]  \\
    &=\mathbb{E}\left[r_{t+1}|s_t=s\right] +\gamma \mathbb{E}\left[r_{t+2}+\gamma r_{t+3}+\gamma^{2} r_{t+4}+\ldots \mid s_{t}=s\right]\\
    &=R(s)+\gamma \mathbb{E}[G_{t+1}|s_t=s] \\
    &=R(s)+\gamma \mathbb{E}[V(s_{t+1})|s_t=s]\\
    &=R(s)+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s\right) V\left(s^{\prime}\right)
    \end{aligned}  
$$

贝尔曼方程，也叫作“动态规划方程”，定义的就是当前状态与未来状态的迭代关系，即：把状态转移概率乘它未来的状态的价值，再加上它的即时奖励（immediate reward），就会得到它当前状态的价值。

<img width="587" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/f86a4ac6-f90f-45dd-988c-ba6e9e6d158e">

直接求解：

$$
  \begin{aligned}
    \boldsymbol{V} &= \boldsymbol{\boldsymbol{R}}+ \gamma \boldsymbol{P}\boldsymbol{V} \\
    \boldsymbol{I}\boldsymbol{V} &= \boldsymbol{R}+ \gamma \boldsymbol{P}\boldsymbol{V} \\
    (\boldsymbol{I}-\gamma \boldsymbol{P})\boldsymbol{V}&=\boldsymbol{R} \\
    \boldsymbol{V}&=(\boldsymbol{I}-\gamma \boldsymbol{P})^{-1}\boldsymbol{R}
    \end{aligned}
$$

可以直接得到**解析解（analytic solution）**：

$$
  \boldsymbol{V}=(\boldsymbol{I}-\gamma \boldsymbol{P})^{-1} \boldsymbol{R}
$$

可以通过矩阵求逆把 $\boldsymbol{V}$ 的价值直接求出来。但是一个问题是这个矩阵求逆的过程的复杂度是 $O(N^3)$ 。所以当状态非常多的时候，比如从10个状态到1000个状态，或者到100万个状态，当有100万个状态的时候，状态转移矩阵就会是一个100万乘100万的矩阵，对这样一个大矩阵求逆是非常困难的。所以这种通过解析解去求解的方法只适用于很小量的马尔可夫奖励过程。

## 迭代算法计算马尔可夫奖励过程价值

迭代计算状态非常多的马尔可夫奖励过程（large MRP），比如：动态规划的方法，蒙特卡洛的方法（通过采样的办法计算它），时序差分学习（temporal-difference learning，TD learning）的方法（时序差分学习是动态规划和蒙特卡洛方法的一个结合）。

状态转移概率是已知的，这种情况下使用的算法称为有模型算法。但大部分情况下对于智能体来说，环境是未知的，这种情况下使用的算法称为免模型算法。

### 蒙特卡洛估计

蒙特卡洛的方法：从某个状态开始随机产生多条“轨迹”, 然后直接取每条轨迹回报的平均值，就等价于开展节点的价值（撒豆子来估计图形面积）。其分成首次访问蒙特卡洛（FVMC）和每次访问蒙特卡洛（EVMC）

<img width="410" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/288681a6-c619-475d-972a-d2198a175a23">

动态规划的方法：通过自举（bootstrapping）的方法一直迭代贝尔曼方程，直到价值函数收敛（当最后更新的状态与上一个状态的区别并不大）。

<img width="518" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/f5122292-5bae-44ed-96c2-ccc5d336288e">

<img width="516" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/a6d3d0ef-f7ce-4722-903d-67acb426a0bc">

新的估计值 -> 旧的估计值 + 步长 *（目标值-旧的估计值）

$$
V(s_t) \leftarrow V(s_t) + \alpha[G_t- V(s_{t})]
$$

其中 $\alpha$ 表示学习率， $G_t- V(S_{t+1})$  为目标值与估计值之间的误差（ $\text{error}$ ）。此外， $\text{FVMC}$ 是一种基于回合的增量式方法，具有无偏性和收敛快的优点，但是在状态空间较大的情况下，依然需要训练很多个回合才能达到稳定的结果。而 $\text{EVMC}$ 则是更为精确的预测方法，但是计算的成本相对也更高。

## 马尔可夫决策过程 

马尔可夫决策过程多了决策（决策是指动作），其他的定义与马尔可夫奖励过程的是类似的。此外，状态转移也多了一个条件，变成了 $p\left(s_{t+1}=s^{\prime} \mid s_{t}=s,a_{t}=a\right)$ 。未来的状态不仅依赖于当前的状态，也依赖于在当前状态智能体采取的动作。马尔可夫决策过程满足条件：

$$
  p\left(s_{t+1} \mid s_{t}, a_{t}\right) =p\left(s_{t+1} \mid h_{t}, a_{t}\right)   
$$

奖励函数，它也多了一个当前的动作，变成了 $R\left(s_{t}=s, a_{t}=a\right)=\mathbb{E}\left[r_{t} \mid s_{t}=s, a_{t}=a\right]$ 。

### 马尔可夫决策过程中策略

把当前状态代入策略函数来得到一个概率，即 

$$
  \pi(a \mid s)=p\left(a_{t}=a \mid s_{t}=s\right)
$$

概率代表在所有可能的动作里面怎样采取行动。另外策略也可能是确定的，它有可能直接输出一个值，或者直接告诉我们当前应该采取什么样的动作，而不是一个动作的概率。假设概率函数是平稳的（stationary），不同时间点，采取的动作其实都是在对策略函数进行采样。

已知马尔可夫决策过程和策略 $\pi$ ，可以把马尔可夫决策过程转换成马尔可夫奖励过程。在马尔可夫决策过程里面，状态转移函数 $P(s'|s,a)$ 基于它当前的状态以及它当前的动作。因为现在已知策略函数，也就是已知在每一个状态下，可能采取的动作的概率，所以就可以直接把动作进行加和，去掉 $a$ ，这样就可以得到对于马尔可夫奖励过程的转移，这里就没有动作，即

$$
  P_{\pi}\left(s^{\prime} \mid s\right)=\sum_{a \in A} \pi(a \mid s) p\left(s^{\prime} \mid s, a\right)
$$

对于奖励函数，也可以把动作去掉，这样就会得到类似于马尔可夫奖励过程的奖励函数，即

$$
  r_{\pi}(s)=\sum_{a \in A} \pi(a \mid s) R(s, a)
$$

<img width="512" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/2c37ed30-d6e2-47e1-84d9-65e059681a34">

### 马尔可夫决策过程中的价值函数

引入了一个 **Q 函数（Q-function）**。Q 函数也被称为**动作价值函数（action-value function）**。Q 函数定义的是在某一个状态采取某一个动作，它有可能得到的回报的一个期望，即

$$
Q_{\pi}(s, a)=\mathbb{E}_{\pi}\left[ \right] 
$$

$$ G_{t} \mid s_{t}=s, a_{t}=a $$

这里的期望其实也是基于策略函数的。所以需要对策略函数进行一个加和，然后得到它的价值。
对 Q 函数中的动作进行加和，就得到价值函数：

$$
V_{\pi}(s)=\sum_{a \in A} \pi(a \mid s) Q_{\pi}(s, a)
$$

## 免模型预测

### 时序差分估计

**单步时序差分**（ $\text{one-step TD}$ ,  $TD(0)$ ）: 

$$
V(s_t) \leftarrow V(s_t) + \alpha[r_{t+1}+\gamma V(s_{t+1})- V(s_{t})]
$$

类似于蒙特卡罗方法，更新过程中使用了当前奖励和后继状态的估计；但同时也利用了贝尔曼方程的思想，将下一状态的值函数作为现有状态值函数的一部分估计来更新现有状态的值函数。此外，时序差分还结合了自举（ $\text{bootstrap}$ ）的思想，即未来状态的价值是通过现有的估计 $r_{t+1}+\gamma V(s_{t+1})$ （也叫做**时序差分目标**）进行计算的，即使用一个状态的估计值来更新该状态的估计值，没有再利用后续状态信息的计算方法。这种方法的好处在于可以将问题分解成只涉及一步的预测，从而简化计算。此外， $\delta=r_{t+1}+\gamma V(s_{t+1})- V(s_{t})$ 被定义为 **时序差分误差**（ $\text{TD error}$ ）。

n 步时序差分：如调整为两步，利用两步得到的回报来更新状态的价值，调整 $n$ 步就是 $n$ 步时序差分（ $\text{n-step TD}$ ）。

$$
\begin{aligned}
& n=1(\mathrm{TD}) \quad G_t^{(1)}=r_{t+1}+\gamma V\left(s_{t+1}\right) \\
& n=2 \quad G_t^{(2)}=r_{t+1}+\gamma r_{t+2}+\gamma^2 V\left(s_{t+2}\right) \\
& n=\infty(\mathrm{MC}) \quad G_t^{\infty}=r_{t+1}+\gamma r_{t+2}+\cdots+\gamma^{T-t-1} r_T \\
&
\end{aligned}
$$

当 $n$ 趋近于无穷大时，就变成了蒙特卡洛方法，因此可以通过调整自举的步数，来实现蒙特卡洛方法和时序差分方法之间的权衡。这个 $n$ 通常会用 
$\lambda$ 来表示，即 $\text{TD}(\lambda)$ 方法。

常见的以下方法来选择合适的 $\lambda$ ：网格搜索（ $\text{Grid Search}$ ）、随机搜索（ $\text{Random Search}$ ）、自适应选择、交叉验证（ $\text{Cross-validation}$ ）、经验取值。

### 时序差分和蒙特卡洛的比较

<img width="428" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/67445bc9-a719-4cbd-ac59-babbda2c056a">


## 免模型控制

给定一个马尔可夫决策过程，输出最优策略以及对应的最优价值函数。而免模型则是指不需要知道环境的状态转移概率的一类算法。基础的免模型算法， $\text{Q-learning}$ 和 $\text{Sarsa}$ ，也都是基于时序差分的方法。

### Q-learning 算法

$\text{Q-learning}$ 算法更新公式下式所示：

$$
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha [r_t+\gamma\max_{a}Q(s_{t+1},a)-Q(s_t,a_t)]
$$

其中 $y_t = r_t+\gamma\max_{a}Q^{\prime}(s_{t+1},a)$ 表示期望的 $Q$ 值， $Q(s_t,a_t)$ 表示实际的 $Q$ 值， $\alpha$ 是学习率。在 $\text{Q-learning}$ 算法中，由于没有额外的参数，因此只需要直接一步步迭代更新 $Q$ 值即可。

时序差分方法中状态价值函数的更新公式：

$$
V(s_t) \leftarrow V(s_t) + \alpha[r_{t+1}+\gamma V(s_{t+1})- V(s_{t})]
$$

会发现两者的更新方式是一样的，都是基于时序差分的更新方法。不同的是，动作价值函数更新时是直接拿最大的未来动作价值的 $\gamma\max_{a}Q(s_{t+1},a)$ 来估计的，而在状态价值函数更新中相当于是拿对应的平均值来估计的。这就会导致这个估计相当于状态价值函数中的估计更不准确，一般称为 **Q 值的过估计**，当然这个过估计仅仅限于以 $\text{Q-learning}$ 为基础的算法，不同的算法为了优化这个问题使用了不同的估计方式。

Q 表格： $3 \times 3$ 网格，以左上角为起点，右下角为终点，机器人可以向上、向下、向左和向右随意走动，每次移动后机器人会收到一个 $-1$ 的奖励，即奖励函数 $R(s,a)=1$ ，然后求出机器人从起点走到终点的最短路径。

<img width="230" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/a42572c2-3e58-4111-b7f6-bc358ccbfd27">

把机器人的位置看作状态，这样一来总共有 $\text{9}$ 个状态，$\text{4}$ 个动作 $a_1,a_2,a_3,a_4$（分别对应上、下、左、右）。知道 Q 函数，也就是状态价值函数的输入就是状态和动作，输出就是一个值，由于这里的状态和动作都是离散的，这样就用一个表格来表示：


<div style="text-align: center;">
  <div style="display: table; margin: 0 auto;">
    <table>
      <tr>
        <th> $\space$ </th>
        <th>$s_1$</th>
        <th>$s_2$</th>
        <th>$s_3$</th>
        <th>$s_4$</th>
        <th>$s_5$</th>
        <th>$s_6$</th>
        <th>$s_7$</th>
        <th>$s_8$</th>
        <th>$s_9$</th>
      </tr>
      <tr>
        <td>$a_1$</td>
        <td>$\text{0}$</td>
        <td>$\text{0}$</td>
        <td>$\text{0}$</td>
        <td>$\text{0}$</td>
        <td>$\text{0}$</td>
        <td>$\text{0}$</td>
        <td>$\text{0}$</td>
        <td>$\text{0}$</td>
        <td>$\text{0}$</td>
      </tr>
      <tr>
        <td>$a_2$</td>
        <td>$\text{0}$</td>
        <td>$\text{0}$</td>
        <td>$\text{0}$</td>
        <td>$\text{0}$</td>
        <td>$\text{0}$</td>
        <td>$\text{0}$</td>
        <td>$\text{0}$</td>
        <td>$\text{0}$</td>
        <td>$\text{0}$</td>
      </tr>
      <tr>
        <td>$a_3$</td>
        <td>$\text{0}$</td>
        <td>$\text{0}$</td>
        <td>$\text{0}$</td>
        <td>$\text{0}$</td>
        <td>$\text{0}$</td>
        <td>$\text{0}$</td>
        <td>$\text{0}$</td>
        <td>$\text{0}$</td>
        <td>$\text{0}$</td>
      </tr>
      <tr>
        <td>$a_4$</td>
        <td>$\text{0}$</td>
        <td>$\text{0}$</td>
        <td>$\text{0}$</td>
        <td>$\text{0}$</td>
        <td>$\text{0}$</td>
        <td>$\text{0}$</td>
        <td>$\text{0}$</td>
        <td>$\text{0}$</td>
        <td>$\text{0}$</td>
      </tr>
    </table>
  </div>
  <div>  Q表格 </div>
</div>

表格的横和列对应状态和动作，数值表示对应的 $Q$ 值。在实践中可以给所有的 $Q$ 预先设一个值，这就是 $Q$ 值的初始化。这些值是可以随机的。但终止状态 $s_9$ 对应的所有 $Q$ 值，包括 $Q(s_9,a_1),Q(s_9,a_2),Q(s_9,a_3),Q(s_9,a_4)$ 等都必须为 $\text{0}$ ，并且也不参与 $Q$ 值的更新。

更新过程，是时序差分方法。具体的做法是，让机器人自行在网格中走动，走到一个状态，就把对应的 $Q$ 值 更新一次，这个过程就叫做 **探索** 。这个探索的过程也是时序差分方法结合了蒙特卡洛方法的体现。

### 探索策略

### Sarsa 算法


## 强化学习 vs 深度学习

强化学习的问题分为预测问题和控制问题。预测主要是告诉我们当前状态下采取什么动作比较好（谋士），而控制则是按照某种方式决策（主公）。
深度强化学习中深度学习就是用来提高强化学习中预测的效果的。比 Q-learning 的 Q 表就完全可以用神经网络来拟合。
当然，在控制问题中，也可以利用深度学习或者其他的方法来提高性能，例如结合进化算法来提高强化学习的探索能力。

PS: 基于大量的样本来对相应算法进行迭代更新并且达到最优的，这个过程我们称之为训练。
但强化学习是在交互中产生样本的，是一个产生样本、算法更新、再次产生样本、再次算法更新的动态循环训练过程，而不是一个准备样本、算法更新的静态训练过程。

强化学习解决的是序列决策问题，而深度学习解决的是“打标签”问题。

<img width="481" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/974b76b4-04af-4683-b77f-8da3efa95a0c">

