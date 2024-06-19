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

把基于价值的智能体和基于策略的智能体结合起来就有了演员-评论员智能体（actor-critic agent）。这一类智能体把策略和价值函数都学习了，然后通过两者的交互得到最佳的动作。

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

在随机过程中，**马尔可夫性质（Markov property）**是指一个随机过程在给定现在状态及所有过去状态情况下，其未来状态的条件概率分布仅依赖于当前状态。

$$
p\left(X_{t+1}=x_{t+1} \mid X_{0:t}=x_{0: t}\right)=p\left(X_{t+1}=x_{t+1} \mid X_{t}=x_{t}\right)
$$

其中， $X_{0:t}$ 表示变量集合 $X_{0}, X_{1}, \cdots, X_{t}$,  $x_{0:t}$ 为在状态空间中的状态序列 $x_{0}, x_{1}, \cdots, x_{t}$。
马尔可夫性质也可以描述为给定当前状态时，将来的状态与过去状态是条件独立的。如果某一个过程满足马尔可夫性质，那么未来的转移与过去的是独立的，它只取决于现在。马尔可夫性质是所有马尔可夫过程的基础。

## 马尔可夫链

马尔可夫过程是一组具有马尔可夫性质的随机变量序列 $s_1,\cdots,s_t$ ，其中下一个时刻的状态 $s_{t+1}$ 只取决于当前状态 $s_t$ 。我们设状态的历史为 
 $h_{t}=\{s_{1}, s_{2}, s_{3}, \ldots, s_{t}\}$ ( $h_t$ 包含了之前的所有状态)，则马尔可夫过程满足条件：

$$
p\left(s_{t+1} \mid s_{t}\right) =p\left(s_{t+1} \mid h_{t}\right) \tag{2.1}
$$

从当前 $s_t$ 转移到 $s_{t+1}$ ，它是直接就等于它之前所有的状态转移到 $s_{t+1}$ 。
