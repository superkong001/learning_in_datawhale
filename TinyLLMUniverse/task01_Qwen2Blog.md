引用参考：
- https://github.com/datawhalechina/tiny-universe
- https://github.com/huggingface/transformers/tree/v4.39.3/src/transformers/models/qwen2
- [https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch0](https://github.com/datawhalechina/so-large-lm/tree/main)

# 基础概念
## 语言模型

<img width="686" alt="image" src="https://github.com/user-attachments/assets/6271b41f-4a96-4d1f-9235-53daa515b94f" />

语言模型两类问题：

1. 输入序列问题：输入是文本信号，而计算机能进入神经网络处理和计算的是数值，所以需要讲字符通过一定方式转化为数值。 如：独热编码。
2. 输出序列问题：输出要求是文本，而神经网络的输出是数值类型的（分类问题：二分类问题对应01输出，多分类对应多个01输出；回归问题：对应数值类型输出），所以需要建立神经网络的数值类型输出和最终字符输出的映射关系。如：构建神经网络的输出独热编码后每个字符的概率，选取最高的那个。

语言模型（LM）的经典定义是一种对词元序列(token)的概率分布。假设我们有一个词元集的词汇表 $V$ 。语言模型p为每个词元序列 $x_{1},...,x_{L}$ ∈ $V$ 分配一个概率（介于0和1之间的数字）：

$$
p(x_1, \dots, x_L)
$$

概率直观地告诉我们一个标记序列有多“好（good）”。例如，如果词汇表为{ate, ball, cheese, mouse, the}，语言模型可能会分配以下概率（演示）：

$$
\begin{aligned}
p(\text{the, mouse, ate, the, cheese}) = 0.02 \\
p(\text{the, cheese ate, the, mouse}) = 0.01 \\
p(\text{mouse, the, the, cheese, ate}) = 0.0001 \\
\end{aligned}
$$

- 语言模型是序列  $x_{1:L}$ 的概率分布 p。
- 一个好的语言模型应具有语言能力和世界知识。
- 自回归语言模型允许有效地生成给定提示 $x_{1:i}$ 的补全 $x_{i+1:L}$。
- 温度可以用来控制生成中的变异量。

如：语言模型应该隐含地赋予"𝗆𝗈𝗎𝗌𝖾 𝗍𝗁𝖾 𝗍𝗁𝖾 𝖼𝗁𝖾𝖾𝗌𝖾 𝖺𝗍𝖾"一个非常低的概率，因为它在语法上是不正确的（句法知识）。由于世界知识的存在，语言模型应该隐含地赋予"𝗍𝗁𝖾 𝗆𝗈𝗎𝗌𝖾 𝖺𝗍𝖾 𝗍𝗁𝖾 𝖼𝗁𝖾𝖾𝗌𝖾"比"𝗍𝗁𝖾 𝖼𝗁𝖾𝖾𝗌𝖾 𝖺𝗍𝖾 𝗍𝗁𝖾 𝗆𝗈𝗎𝗌𝖾"更高的概率。这是因为两个句子在句法上是相同的，但在语义上却存在差异，而语言模型需要具备卓越的语言能力和世界知识，才能准确评估序列的概率。

语言模型也可以做生成任务。如定义所示，语言模型p接受一个序列并返回一个概率来评估其好坏。也可以根据语言模型生成一个序列。最纯粹的方法是从语言模型 $p$ 中以概率 $p(x_{1:L})$ 进行采样，表示为：

$$
x_{1:L}∼p.
$$

### 自回归语言模型(Autoregressive language models)

将序列  $x_{1:L}$  的联合分布  $p(x_{1:L})$  的常见写法是使用概率的链式法则：

$$
p(x_{1:L}) = p(x_1) p(x_2 \mid x_1) p(x_3 \mid x_1, x_2) \cdots p(x_L \mid x_{1:L-1}) = \prod_{i=1}^L p(x_i \mid x_{1:i-1}).
$$

例子：

$$
\begin{aligned}
p({the}, {mouse}, {ate}, {the}, {cheese}) = p({the})p({mouse} \mid {the})p({ate} \mid {the}, {mouse})p({the} \mid {the}, {mouse}, {ate})p({cheese} \mid {the}, {mouse}, {ate}, {the})
\end{aligned}
$$

$$
\begin{aligned}
\text { for } i & =1, \ldots, L: \\
    x_i & \sim p\left(x_i \mid x_{1: i-1}\right)^{1 / T},
\end{aligned}
$$

其中  $T≥0$  是一个控制我们希望从语言模型中得到多少随机性的温度参数：
- T=0：确定性地在每个位置 i 选择最可能的词元 $x_{i}$
- T=1：从纯语言模型“正常（normally）”采样
- T=∞：从整个词汇表上的均匀分布中采样

如果仅将概率提高到  $1/T$  的次方，概率分布可能不会加和到 1。可以通过重新标准化分布来解决这个问题。将标准化版本  $p_{T}(x_{i}∣x_{1:i−1})∝p(x_{i}∣x_{1:i−1})^{1/T}$ 称为退火条件概率分布。"退火"类比的是对概率分布进行调整的过程。"退火"分布是通过将原始概率分布的每个元素都取幂  $1/T$ ，然后重新标准化得到的新分布。当 $T ≠ 1$ 时，这个过程会改变原始概率分布，因此从"退火"分布中采样得到的结果可能与对每一步的条件分布应用 T 并进行迭代采样的结果不同。

另外当  $T$  值较高时，会获得更平均的概率分布，生成的结果更具随机性；反之，当 $T$ 值较低时，模型会更倾向于生成概率较高的词元。

Tips:

模拟退火算法（simulated annealing algorithm）：参考金属加工工艺中的退火原理：退火的冷却速度（缓慢的速度冷却）比淬火（快速冷却）慢，（由于温度与随机性对应，温度越高随机性越大，退火随机性由高到底衰减）可以使金属释放内部应力、增加材料延展性和韧性，从而达到平衡状态，获得良好的工艺性能和使用性能。开始以较大概率接受不完美，随时间推移概率变小。

- 淬火：显著提高金属的硬度、强度和耐磨性，但会使材料变脆，增加内应力，甚至可能导致工件变形或开裂。例如，刀具通过淬火提升切削能力。
- 退火：降低材料的硬度，提高塑性和韧性，消除内部残余应力，稳定尺寸，减少变形与裂纹倾向，还能细化晶粒、调整组织，为后续加工或最终热处理做组织准备。如薄壁零件通过退火改善加工性能。

退火采样的原理：
在退火过程中，我们的目标是通过一个渐进的“温度”变化来寻找全局最优解。温度 $𝑇$ 从较高的值开始，逐渐降低，目的是避免模型在初期过早陷入局部最优，而是探索更广的空间。温度的逐步降低过程，使得模型在初期采样时能够接受一些较差的选择（即增加随机性），而在后期（当温度降低时）逐渐倾向于选择那些更有可能的结果，从而收敛到最优解。

退火分布 vs. 条件分布：
- 退火分布：退火过程中的分布是根据温度逐渐降低的过程进行调整的。随着温度的逐渐降低，退火分布趋近于一个“确定性”的选择（类似于 $𝑇=0$ 时的选择，接近最大概率），这使得模型最终可能选择最可能的结果。退火的过程通过在较高的温度下允许较大范围的随机性，最终在低温时收敛。它是为了全局优化而设计的，允许在高温下进行较大的随机探索，因此可能接受一些在低温时看似不太可能的选择。随着温度逐步降低，模型会逐渐收敛到一个更优的解（更有可能的词）。
- 条件分布与迭代采样：在常规的条件概率分布采样中，我们每次都使用当前的 条件分布 来生成下一个词。例如，给定前一个词，会从条件概率分布中直接采样，选择下一个词。这个过程与退火不同，退火在每一步的选择不仅依赖于当前的条件分布，还考虑了过去温度的影响，允许更大的探索性。迭代采样的条件分布则更加直接、局部化，它每一步都依赖于当前的条件概率进行采样，通常不考虑之前步骤的温度变化，也不会在后期的采样中做“退火式”的修正。

例如：

$$
\begin{array}{cl}
p(\text { cheese })=0.4, & p(\text { mouse })=0.6 \\
p_{T=0.5}(\text { cheese })=0.31, & \left.p_{T=0.5} \text { (mouse }\right)=0.69 \\
\left.p_{T=0.2} \text { (cheese }\right)=0.12, & p_{T=0.2} \text { (mouse) }=0.88 \\
\left.p_{T=0} \text { (cheese }\right)=0, & \left.p_{T=0} \text { (mouse }\right)=1
\end{array}
$$

## 大模型理论

香农在信息理论中用于度量概率分布的熵（Entropy），熵是一个概率分布的度量，它衡量的是随机变量的不确定性。对于离散随机变量 $𝑋$ 的概率分布 $𝑝(𝑥_1),𝑝(𝑥_2),…,p(x_n)$ ，香农熵 $𝐻(𝑋)$ 被定义为：

$$
H(𝑋) = \sum_x p(x) \log \frac{1}{p(x)}.
$$

其中：

- $𝑝(𝑥)$ 是事件 $𝑥_𝑖$ 发生的概率。
- $log_2$ 是以2为底的对数，这样计算的熵单位是比特（bit），也就是信息量的标准单位。
- 熵越大，表示不确定性越高，也就是信息量越大。

熵实际上是一个衡量将样本 $x∼p$ 编码（即压缩）成比特串所需要的预期比特数的度量，它告诉我们一个系统或一个信息源的不确定性或者信息量。如："the mouse ate the cheese" 可能会被编码成 "0001110101"。

熵 $𝐻(𝑋) 衡量了随机变量 $𝑋$ 的不确定性。如果一个随机变量的取值具有很高的不确定性（比如每个事件发生的概率相等），那么熵值较大。如果随机变量的某个取值的概率接近 1，而其他取值的概率接近 0，那么不确定性较低，熵值较小。也就是说，熵值低代表信息更具确定性，熵值高代表信息更加不确定。

熵的值越小，表明序列的结构性越强，编码的长度就越短。直观地理解， $\log \frac{1}{p(x)}$  可以视为用于表示出现概率为 $p(x)$ 的元素 $x$ 的编码的长度。

例如：

- $p(x)=1/8$ ，就需要分配  $log_{2}(8)=3$ 个比特（或等价地， $log(8)=2.08$ 个自然单位）。
- 公平的掷硬币（正面和反面概率各为 0.5），那么熵是：

$$
H(𝑋) = − \[ 0.5 \log_{2}(0.5)+0.5 \log_{2}(0.5) \] = 1 bit
$$ 
  
- 不公平的掷硬币（比如正面 0.9，反面 0.1），那么熵是：
  
$$
H(X)= − \[ 0.9 \log_{2}(0.9)+0.1 \log_{2}(0.1) \] ≈ 0.468 bits
$$ 

交叉熵：

$$
H(p,q) = \sum_x p(x) \log_2 \frac{1}{q(x)} = -\sum_x p(x) \log_2 q(x) 
$$

- $p(x)$ ：表示真实的分布（真实的概率）。
- $𝑞(𝑥)$ ：表示我们用来估计 $𝑝(𝑥)$ 的假设分布。

这测量了需要多少比特（nats）来编码样本x∼p，使用由模型q给出的压缩方案（用长度为1/q(x)的代码表示x）。

通过语言模型估计熵。一个关键的属性是，交叉熵H(p,q)上界是熵H(p): 如果能够完美地估计 $𝑝(𝑥)$（即 $𝑝(𝑥)=𝑞(𝑥)$ ），那么交叉熵就会等于熵，表示没有任何错误或不确定性。然而，如果估计 $𝑞(𝑥)$ 不准确，那么交叉熵将大于熵，表示估计误差。

它是一种用来衡量两个概率分布之间差异的度量。这意味着可以通过构建一个只有来自真实数据分布 $p$ 的样本的（语言）模型 $q$ 来估计 $H(p,q)$ ，而 $H(p)$ 通常无法访问，所以可以通过构建更好的模型q来得到熵H(p)的更好的估计，由H(p,q)衡量。就是，$q(x)$ 是用来估计 $𝑝(𝑥)$ (真实的概率分布)的概率分布（通常是通过某种模型或猜测得出的）。

交叉熵实际上可以分解为两个部分：

$$
H(p,q) = H(p) + D_\text{KL}(p∥q) = H(p) + \sum_{x} p(x) \log_2 \frac{p(x)}{q(x)}
= -\sum_{x} p(x) \log_2 p(x) + ( \sum_{x} p(x) ( \log_2 p(x) - \log_2 q(x))
= -\sum_x p(x) \log_2 q(x)
$$

其中：

- $𝐻(𝑝)$ 是真实分布 $𝑝(𝑥)$ 的熵，表示使用真实分布时的平均信息量。
- $𝐷_\text{KL}(𝑝∥𝑞)$ 是 KL散度（Kullback-Leibler divergence），它度量了 $𝑝(𝑥)$ 和 $𝑞(𝑥)$ 之间的差异。

KL散度是一个非对称的度量，它表示的是如果用 $𝑞(𝑥)$ 代替 $𝑝(𝑥)$ 时，增加的额外信息量。 $p(x)$ 和 $𝑞(𝑥)$ 不一致时，KL散度大于零。这意味着 $𝑞(𝑥)$ 在某些地方与 $𝑝(𝑥)$ 的概率分布不匹配，从而导致交叉熵大于熵。
因此交叉熵包含了真实熵和KL散度，它不仅衡量了真实分布的平均信息量，还反映了两个分布之间的差异。

- 统计语言模型（SLM）
    - 主要建立在统计学习理论框架，通常使用链式法则建模句子序列
      - 例如： $p(I, am, fine) = p(I \mid START) * p(am\mid I) * p(fine\mid I, am)$ 
    -  $n$ -gram 语言模型：基于马尔科夫假设，当前词概率仅与前 $n - 1$ 个词有关

$$
\begin{align*}
p(s) &= p(w_1)p(w_2\mid w_1) \cdots p(w_m\mid w_{m - n + 1}, \cdots, w_{m - 1})\\
&=\prod_{i = 1}^{m} p(w_i\mid w_{i - n + 1}, \cdots, w_{i - 1})
\end{align*}
$$

### N-gram模型
以前解决语音识别、机器翻译任务的主要模型是噪声信道模型。以语音识别为例：
- 假设有一些从某个分布p中抽取的文本
- 这些文本被转换为语音（声音信号）
- 然后给定语音，我们希望恢复（最有可能的）文本。这可以通过贝叶斯定理实现：

$$
p(\text{text} \mid \text{speech}) \propto \underbrace{p(\text{text})}_\text{language model} \underbrace{p(\text{speech} \mid \text{text})} _ \text{acoustic model}  
$$

在n-gram模型中，关于 $x_{i}$ 的预测只依赖于最后的 $n-1$ 个字符 $x_{i−(n−1):i−1}$ ，而不是整个历史：

$$
p(x_i \mid x_{1:i-1}) = p(x_i \mid x_{i-(n-1):i-1}).
$$


如: 

- trigram（n=2）二元语言模型示例：

$p(I,am,fine) = p(I|START) * P(am|I) * p(fine|am)$

- trigram（n=3）模型示例：

$$
p(𝖼𝗁𝖾𝖾𝗌𝖾∣𝗍𝗁𝖾,𝗆𝗈𝗎𝗌𝖾,𝖺𝗍𝖾,𝗍𝗁𝖾)=p(𝖼𝗁𝖾𝖾𝗌𝖾∣𝖺𝗍𝖾,𝗍𝗁𝖾)
$$

概率是基于各种 $\text{n-gram}$（例如，𝖺𝗍𝖾 𝗍𝗁𝖾 𝗆𝗈𝗎𝗌𝖾和𝖺𝗍𝖾 𝗍𝗁𝖾 𝖼𝗁𝖾𝖾𝗌𝖾）在大量文本中出现的次数计算的，并且适当地平滑以避免过拟合（例如，Kneser-Ney平滑）。

- 四元语言模型估计示例（最大似然估计）：

$$
P(\boldsymbol{w}|students\ opened\ their)=\frac{count(students\ opened\ their\ \boldsymbol{w})}{count(students\ opened\ their)}
$$  

“students opened their” 出现了1000次、 
“students opened their books”出现了400次、 
$P(books|students\ opened\ their)= 0.4$ 、 
“students opened their exams” 出现了100次、 
$P(exams|students\ opened\ their)= 0.1$ 

存在问题1，例如以下的前缀：

```
𝖲𝗍𝖺𝗇𝖿𝗈𝗋𝖽 𝗁𝖺𝗌 𝖺 𝗇𝖾𝗐 𝖼𝗈𝗎𝗋𝗌𝖾 𝗈𝗇 𝗅𝖺𝗋𝗀𝖾 𝗅𝖺𝗇𝗀𝗎𝖺𝗀𝖾 𝗆𝗈𝖽𝖾𝗅𝗌. 𝖨𝗍 𝗐𝗂𝗅𝗅 𝖻𝖾 𝗍𝖺𝗎𝗀𝗁𝗍 𝖻𝗒 ___
```

如果 $n$ 太小，那么模型将无法捕获长距离的依赖关系，下一个词将无法依赖于𝖲𝗍𝖺𝗇𝖿𝗈𝗋𝖽。然而，如果 $n$ 太大，统计上将无法得到概率的好估计（所有合理的长序列都出现0次）：

$$
count(𝖲𝗍𝖺𝗇𝖿𝗈𝗋𝖽,𝗁𝖺𝗌,𝖺,𝗇𝖾𝗐,𝖼𝗈𝗎𝗋𝗌𝖾,𝗈𝗇,𝗅𝖺𝗋𝗀𝖾,𝗅𝖺𝗇𝗀𝗎𝖺𝗀𝖾,𝗆𝗈𝖽𝖾𝗅𝗌)=0
$$

因此，语言模型被限制在如语音识别和机器翻译等任务中，其中声音信号或源文本提供了足够的信息，只捕获局部依赖关系（而无法捕获长距离依赖关系）。

存在问题2：如果词没有出现过, $count = 0$ 

<img width="581" alt="image" src="https://github.com/user-attachments/assets/b36cb30e-464c-46eb-a0b0-7f2388aa8f0c" />

采用方法：
- 加一平滑 (又称为 Laplace smoothing )
  - 每个词都加上一次出现

    原始估计 $P_{MLE}(w_i|w_{i - 1})=\frac{count(w_{i - 1}, w_i)}{count(w_{i - 1})}$
          
    加一平滑 $P_{Add - 1}(w_i|w_{i - 1})=\frac{count(w_{i - 1}, w_i)+1}{count(w_{i - 1})+|V|}$  <span style="text-decoration: underline;">词典大小</span>
          
  - 仍然保持概率分布，不破坏概率分布基本性质
      -  $P(w_i)>0, \forall w_i\in V$ 
      -  $\sum_{i}P(w_i) = 1$ 
- 回退 (back - off)
    - 当 $count(w_{i - n + 1}, \ldots, w_i)=0$ ， $n$ 元语言模型退化成更低阶数元语言模型，
        $$P(w_i|w_{i - 1}, \ldots, w_{i - n + 1}) = P(w_i|w_{i - 1}, \ldots, w_{i - n + k + 1})$$
        - 例如：当$count(w_{i - 2}, w_{i - 1}, w_i)=0$时，三元语言模型可以退化成二元语言模型进行估计
        $$P(w_i|w_{i - 1}, w_{i - 2}) = P(w_i|w_{i - 1})$$
- 插值 (interpolation)
    - 例如：混合多个不同阶数的语言模型
      $$P'(w_i|w_{i - 1}, w_{i - 2})=\alpha P(w_i|w_{i - 1}, w_{i - 2})+\beta P(w_i|w_{i - 1})+\gamma P(w_i)$$
    - 可以证明，仍然能够保证语言模型的概率性质
    - 通常这种方式可以结合不同阶数估计方法的优势
    - 但仍然不能从根本解决数据稀疏性问题

### 神经语言模型
神经语言模型，其中 $p(x_{i}∣x_{i−(n−1):i−1})$ 由神经网络给出：

$$
p(cheese∣ate,the)=\text{some-neural-network(ate,the,cheese)}
$$

训练神经网络在统计上是高效的，但在计算上是低效的，因为它在计算上要昂贵得多，而在相同数据量上优于n-gram模型。但由于n-gram模型的扩展性更好，且数据并非瓶颈，所以n-gram模型在至少接下来的十年中仍然占主导地位。

- **Recurrent Neural Networks**（RNNs），包括长短期记忆（LSTMs），使得一个词元 $x_{i}$ 的条件分布可以依赖于整个上下文 $x_{1:i−1}$ （有效地使 $n=∞$ ），但这些模型难以训练。
- **Transformers**是2017年为机器翻译开发，再次返回固定上下文长度n，但更易于训练（并利用了GPU的并行性）。

#### Word2Vec
- 基本功能：给定文本数据,对于每个单词学习一个低维表示
- 基于分布式语义的思想进行设计：词义=背景单词的语义
- 不考虑窗口内单词的顺序：应用了简单的average pooling的策略
- 充分考虑实践和效果：有很多的优化trick,速度快、效果稳定

<img width="167" alt="image" src="https://github.com/user-attachments/assets/b4e632a0-cd2f-4ec1-854f-4034c9d57d43" />

### 大模型
将语言模型转化为任务模型的过程：
- 训练（标准的有监督学习）：训练一个新模型，使其能将输入映射到输出。这可以通过创建一个新模型并利用语言模型作为特征（探针法），或者从现有的语言模型出发，根据训练实例进行更新（微调），或者在这两者之间找到平衡（轻量级的微调）。
- 提示（上下文）学习：根据对任务的描述建一个或一组提示/上下文信息，将其输入到语言模型中以获取基于该任务的生成结果。根据提示/上下文信息的数量，还可以进一步细分：
    - 零样本学习(Zero-shot)：提示/上下文信息的数量为0，模型直接基于对任务的理解输出结果。
    - 单样本学习(One-shot)：提示/上下文信息的数量为1，一般来说模型基于1个例子可以更好的理解任务从而较好的生成结果。
    - 少样本学习(Few-shot)：提示/上下文信息的数量大于1，大模型可以看到更丰富的例子，一般来说获得比单样本学习更好的效果。

模型的大小和训练样本的数量都很重要。(可以进行消融实验，以查看模型的大小和上下文训练实例的数量是否真的重要。)
实验的任务选择如下：Language modeling、Question answering、Translation、Arithmetic、News article generation、Novel tasks

Tips: 消融实验（Ablation Study）是一种常用于机器学习和深度学习研究中的实验方法，基本思想是构建一个完整的系统或模型，并通过逐步删除或修改系统的某些部分，观察性能变化，从而确定哪些部分对模型的性能至关重要。主要目的是通过逐步移除或“消融”系统的某些部分，来评估各个组件对整体性能的贡献。这种实验帮助研究人员理解不同模块、特性或算法设计在模型性能中的重要性。

<img width="908" alt="image" src="https://github.com/user-attachments/assets/d075380a-443f-4190-9037-f137ca7bba31" />

<img width="793" alt="image" src="https://github.com/user-attachments/assets/68456354-1085-436d-9fa3-4e9fa9a72d7b" />

- 预训练输出(Pre-training)：base model
- 后训练输出(Post-training)：instruct model
    - 指令微调(Instruction Tuning)：SFT
        - 使用输入与输出配对的指令数据对于模型进行微调
        - 提升模型通过问答形式进行任务求解的能力

          <img width="644" alt="image" src="https://github.com/user-attachments/assets/b30509ad-0e25-407d-94a9-f934c03b0c1c" />

    - 人类对齐(Human Alignment)
        - 将大语言模型与人类的期望、需求以及价值观对齐
        - 基于人类反馈的强化学习对齐方法(RLHF)

        <img width="654" alt="image" src="https://github.com/user-attachments/assets/03ffa5bf-718b-470c-8f78-7c2d58c19aa7" />

- 扩展定律：通过扩展参数规模、数据规模和计算算力，大语言模型的能力会出现显著提升。
    - KM 扩展定律：OpenAI团队建立的神经语言模型性能与参数规模(N)、数据规模(D)和计算算力(C)之间的幂律关系：

      <img width="802" alt="image" src="https://github.com/user-attachments/assets/e48af5aa-fafb-4c9a-a79f-5dda3cb20440" />

    - Chinchilla扩展定律：DeepMind团队于提出的旨在指导大语言模型充分利用给定的算力资源优化训练：
 
      <img width="702" alt="image" src="https://github.com/user-attachments/assets/9b25bcb6-de4f-4f2b-a42c-c47209e621a8" />

基于扩展定律可以使用小模型性能去预估大模型的性能，或帮助超参数选择；训练过程中使用模型早期性能来预估后续性能。

问题：
- 随着模型参数、数据数量的扩展,模型性能增益将逐渐减小
- 开放数据已经接近枯竭,难以支持扩展定律的持续推进

- 模型的语言建模损失分解为：
    - 可约损失:真实分布和模型分布之间KL散度,可通过优化减少
    - 不可约损失:真实数据分布的熵,无法通过优化减少
      
$$
L(x)=\underbrace{L_{\infty}}_ \text{不可约损失} + \underbrace{\left(\frac{x_0}{x}\right)^{\alpha_x}}_\text{可约损失}
$$

- 涌现能力
    - 指令遵循(Instruction Following)：大语言模型能够按照自然语言指令来执行对应的任务。
 
      <img width="494" alt="image" src="https://github.com/user-attachments/assets/871aa02e-28f1-4880-9ee5-3ad4e7494b07" />

    - 上下文学习(In-context Learning)：在提示中为语言模型提供自然语言指令和任务示例，无需显式梯度更新就能为测试样本生成预期输出。
 
      <img width="754" alt="image" src="https://github.com/user-attachments/assets/911e685e-a676-47f7-88e7-73662292a9bc" />

    - 逐步推理(Step-by-step Reasoning)：在提示中引入任务相关的中间推理步骤来加强复杂任务的求解，从而获得更可靠的答案。
 
      <img width="584" alt="image" src="https://github.com/user-attachments/assets/7aef1498-4246-4aca-8c44-51726008ca10" />

#### 语言模型
语言模型的目标是给定一个序列 $𝑥_1,𝑥_2,…,𝑥_𝐿$ 来预测每个词的条件概率。公式表示为：

$$
p(x_{1:L}) = \prod_{i=1}^L p(x_i \mid x_{1:i-1}) = \prod_{i=1}^L p(x_i \mid {x_1,x_2,...,x_{i-1}})
$$

即给定前面出现的所有词，模型可以预测下一个词 $𝑥_𝑖$ 的概率。这表示模型对整个序列 $𝑋$ 的预测概率，而不是单个词的概率。

困惑度（Perplexity）是用来衡量语言模型预测准确性的一个指标。可以解释为模型在预测下一个词时的平均不确定性。简单来说，如果一个模型的困惑度较低，那么它在预测下一个词的时候就会更加准确。对于给定的语言模型和一个测试数据集，困惑度被定义为

$$
Perplexity(X) = P(X)^{(-1/N)} = P(x_1,x_2,...,x_N)^{(-1/N)} 
$$

其中， $X=x_{1},x_{2},...,x_{N}$ 是测试集中的词序列， $N$ 是测试集中的总词数。困惑度与语言模型的质量紧密相关。
使用 $−1/𝑁$ 次方的目的是将整个序列的概率 $𝑃(𝑋)$ 转化为每个单独词的平均概率，并使得困惑度反映出模型在每个词上的不确定性。

- 较低的困惑度：表示模型对下一个词的预测更准确，也就是模型的预测更具确定性。如果模型能完美预测所有词，困惑度会非常低，接近1。
- 较高的困惑度：表示模型对下一个词的预测更不确定，说明模型的性能较差。一个完全随机的模型，所有词出现的概率都相同，那么这个模型的困惑度会很高。

存在问题：一个序列的联合概率取决于其长度，并且随着长度的增长，其值趋近于零，这使得困惑度变得难以追踪。直观上，希望对每个词标记（token）的概率 $p(x_{i}∣x_{1:i−1})$ 进行平均。这里的 $p(x_i∣x_{1:i−1})$ 表示给定之前的词序列 $x_{1:i−1}$ 后，下一个词 $x_{i}$ 出现的概率。这样做的目的是评估模型在处理各种词标记时的平均性能。

事实上不希望采取算术平均，因为如果给一个词标记分配了0的概率（即模型认为这个词在特定的上下文中绝对不可能出现），那么在算术平均中这会造成极大的问题。因为算术平均并不会为此惩罚你，它只是简单地将所有词标记的概率加在一起，然后除以总数，因此一个非常低的概率（如0）可能会被其他较高的概率抵消。

相反，采用几何平均，这就是困惑度（perplexity）所做的。在几何平均中，每个词标记的概率都被同等看待，并且一个极低的概率（如0）将会导致整个几何平均大幅度下降。因此，通过计算几何平均，可以更好地衡量模型在处理所有可能的词标记时的性能，特别是在处理那些模型可能会出错的情况。

Tips 几何平均的定义：
对于一组数据 $𝑥_1,𝑥_2,…,𝑥_𝑛$ ，几何平均可以通过以下公式计算：

$$
(\prod_{i=1}^n x_i)^\text{1/n}
$$

即：将所有数据值相乘： $𝑥_1\times 𝑥_2\times … \times 𝑥_𝑛$ ，再对结果取 n 次方根。

$$
perplexity_p \left( x_{1:L} \right) = \exp \left( \frac{1}{L} \sum_{i = 1}^{L} \log \frac{1}{p \left( x_{i} \mid x_{1:i - 1} \right)} \right) \text {. }
$$

公式中： 
- $p \left( x_{i} \mid x_{1:i - 1} \right)$ 表示模型在给定前面 $i-1$ 个词 $x_{1:i-1}$ 后，预测第 $i$ 个词 $x_i$ 的条件概率。
- $\log \frac{1}{p \left( x_{i} \mid x_{1:i - 1} \right)}$ 对每个词的条件概率 $p \left( x_{i} \mid x_{1:i - 1} \right)$ 取倒数，并取对数。它度量了在模型预测该词时的“不确定性”或“困惑度”。如果模型的预测准确，条件概率会较高，取对数后的值会较低。同时表达式也代表了编码长度，因为在计算的是平均编码长度，这个长度反映了给定当前词或标记后，下一个词或标记可能的选择数量。因此，通过对平均编码长度取指数，可以得到可能的选择数量，这也就是"分支因子"。
- 困惑度反映了模型在预测下一个词时的平均不确定性。当困惑度为 10 时，模型在预测下一个词时，平均上会考虑大约 10 个词作为可能的选择，且它们的概率相对接近。这表示模型的预测不完全确定，在给定的上下文下有多个可能的候选词。
- $\sum_{i = 1}^{L} \log \frac{1}{p \left( x_{i} \mid x_{1:i - 1} \right)}$ 对整个序列进行求和，得到所有词的累积不确定性。
- $\frac{1}{L}$ 计算每个词预测的不确定性的平均值。这个结果描述了模型在给定整个序列时的平均困惑度。
- $\exp \left( \cdot \right)$ 取指数，得到困惑度值。因为对数的单位是信息量（比如比特），取指数后的结果则是实际的困惑度度量，反映了模型对整个序列的预测不确定性。
与整个序列的联合概率 $Perplexity_p \left( x_{1:L} \right) = P(X)^{(-1/L)} = \left(\prod_{i = 1}^{L}p(x_{i} \mid x_{1:i - 1}) \right)^{(-1/L)}$ 是等价的。

Tips 因为：

$$
\log Perplexity_p \left( x_{1:L} \right) = \log P(X)^{\frac{1}{L}} = \log \left(\prod_{i = 1}^{L}p(x_{i} \mid x_{1:i - 1}) \right)^{\frac{1}{L}} = - \frac{1}{L} \sum_{i = 1}^{L} \log {p \left( x_{i} \mid x_{1:i - 1} \right)} 
$$

$$
\log Perplexity_p \left( x_{1:L} \right) = \log \left( \exp \left( \frac{1}{L} \sum_{i = 1}^{L} \log \frac{1}{p \left( x_{i} \mid x_{1:i - 1} \right)} \right) \right) = \frac{1}{L} \sum_{i = 1}^{L} \log \frac{1}{p \left( x_{i} \mid x_{1:i - 1} \right)} 
$$

- 用对数和指数化来简化计算和表达模型的 不确定性。在自然语言处理中，困惑度本质上衡量了语言模型对文本的预测不确定性，而对数函数可以将乘法变成加法，使得计算更加方便。此外，对数函数还有一个重要的性质：它能够压缩极端值，避免某些非常小的概率导致数值计算上的溢出或不稳定。

**两类错误**：语言模型可能会犯两种类型的错误，而困惑度对这两种错误的处理方式并不对称：

- 召回错误：语言模型未能正确地为某个词符分配概率值。这种情况下，困惑度是毫不留情的。例如，如果模型为词组 '𝖺𝗍𝖾' 在 '𝗍𝗁𝖾,𝗆𝗈𝗎𝗌𝖾' 后出现的概率预测为接近0，那么对应的困惑度值将趋近于无穷大。

$$
p({ate} \mid {the}, {mouse}) \to 0 \quad\Rightarrow\quad \text{perplexity}_p({the}, {mouse}, {ate}, {the}, {cheese}) \to \infty.
$$

- 精确度错误：语言模型为某些错误的词序列过度分配了概率值。在这种情况下，困惑度会进行适度的惩罚。给定一个语言模型 p，假设我们将一些垃圾分布 $r$ 按照概率 $ϵ$ 混入：

$$
q(x_i \mid x_{1:i-1}) = (1-\epsilon) p(x_i \mid x_{1:i-1}) + \epsilon r(x_i \mid x_{1:i-1}).
$$

那么，在 $q$ 下的 $x_{1:L}$ 的困惑度：

$$
perplexity_q(x_{1:L}) \le \frac{1}{1 - \epsilon} perplexity_p(x_{1:L}) \approxeq (1 + \epsilon) perplexity_p(x_{1:L}),
$$

如果混入 $ϵ=5%$ 的垃圾信息，那么困惑度只增加5%，但这样生成的语言结果会非常糟糕，因为平均每 20 个词符就会生成一个无意义的词符。

Tips：

- 根据困惑度定义：

$$
perplexity_p \left( x_{1:L} \right) = \exp \left(- \frac{1}{L} \sum_{i = 1}^{L} \log {p \left( x_{i} \mid x_{1:i - 1} \right)} \right) 
$$

- 对数的加法不等式推导：

$$
\begin{aligned}
perplexity_q(x_{1:L}) = \exp \left(- \frac{1}{L} \sum_{i = 1}^{L} \log {q \left( x_{i} \mid x_{1:i - 1} \right)} \right) \\
\le \exp \left(- \frac{1}{L} \sum_{i = 1}^{L} {\left( (1-\epsilon) \log p(x_i \mid x_{1:i-1}) + \epsilon \log r(x_i \mid x_{1:i-1}) \right)} \right) \\
= \exp \left(- \frac{1}{L} \sum_{i = 1}^{L} {\left( (1-\epsilon) \log p(x_i \mid x_{1:i-1}) \right)} \right) \cdot \exp \left(- \frac{1}{L} \sum_{i = 1}^{L} {\left( \epsilon \log r(x_i \mid x_{1:i-1}) \right)} \right) \\
= perplexity_p(x_{1:L})^{1-\epsilon} \cdot \exp \left(- \frac{\epsilon}{L} \sum_{i = 1}^{L} {\left( \log r(x_i \mid x_{1:i-1}) \right)} \right) \\
= \frac{1}{(1-\epsilon)} perplexity_p(x_{1:L}) \cdot \exp \left(- \frac{\epsilon}{L} \sum_{i = 1}^{L} {\left( \log r(x_i \mid x_{1:i-1}) \right)} \right) \\
\end{aligned}
$$

根据log是凹函数，根据Jensen’s(詹森)不等式：

$$
\begin{aligned}
f\left(\sum_{i = 1}^{n}\lambda_ix_i\right)\leq\sum_{i = 1}^{n}\lambda_if(x_i) (在凸函数情况下)\\
\sum_{i}\log\left((1 - \epsilon)p(x_i|x_{1:i - 1})+\epsilon r(x_i|x_{1:i - 1})\right)\geq(1 - \epsilon)\sum_{i}\log p(x_i|x_{1:i - 1})+\epsilon\sum_{i}\log r(x_i|x_{1:i - 1}) \\
-\sum_{i}\log\left(perplexity_q(x_{1:L})\right)\leq -(1 - \epsilon)\sum_{i}\log p(x_i|x_{1:i - 1})-\epsilon\sum_{i}\log r(x_i|x_{1:i - 1}) \\
-\frac{1}{L}\sum_{i}\log\left(perplexity_q(x_{1:L})\right)\leq -(1 - \epsilon)\frac{1}{L}\sum_{i}\log p(x_i|x_{1:i - 1})-\epsilon\frac{1}{L}\sum_{i}\log r(x_i|x_{1:i - 1}) \\
\end{aligned}
$$

- 根据上界估计，当 $x<1$ 时

$$
exp(x) \leq \frac{1}{1-x}
$$

- 幂函数的不等式(在 $0<1−\epsilon<1$ 的情况下)

$$
M^{1-\epsilon} \leq \frac{M}{1-\epsilon}
$$

- 根据泰勒展开公式，在 $x_0$ 的某个邻域内， $f(x)$ 可以展开为：

$$
f(x)=f(x_0)+f^{\prime}(x_0)(x - x_0)+\frac{f^{\prime\prime}(x_0)}{2!}(x - x_0)^2+\cdots+\frac{f^{(n)}(x_0)}{n!}(x - x_0)^n+R_n(x)
$$

其中 $R_n(x)$ 是余项。所以，在 $\epsilon=0$ 处：

$$
\frac{1}{1 - \epsilon} \approxeq 1 + \epsilon + \epsilon^2 + \epsilon ^3 + ...
$$

在 $\epsilon$ 很小的时候， $\epsilon ^2$ 和更高阶的项变得非常小，因此可以忽略它们。

## 模型架构
根据输入需求的语言描述（prompt）生成符合需求的结果（completion），形式表达为：

$$
prompt \overset{model}{\leadsto} completion \ \ or \ \ model(prompt) = completion
$$

构建大模型需要考虑的因素：归一化方法、位置编码、激活函数、注意力计算（层数L、注意力头数N、特征维度H，根据模型规模大小确定)

<img width="916" alt="image" src="https://github.com/user-attachments/assets/5a991683-b0b5-4940-a2cd-6e7270a288d9" />

### 分词
分词（Tokenization）：即如何将一个字符串拆分成多个词元。分词方法：

1. 基于空格的分词。如：`text.split(' ')`
   [可视化的词编码](https://observablehq.com/@simonw/gpt-tokenizer)

2. 字节对编码（BPE, Byte pair encoding）算法。
   先将每个字符作为自己的词元，并组合那些经常共同出现的词元。整个过程可以表示为：

    - Input(输入)：训练语料库（字符序列）。
    算法步骤
    - Step1. 初始化词汇表 $V$ 为字符的集合。
    - while(当我们仍然希望V继续增长时)：
      Step2. 找到$V$中共同出现次数最多的元素对 $x,x'$ 。
    - Step3. 用一个新的符号 $xx'$ 替换所有 $x,x'$ 的出现。
    - Step4. 将 $xx'$ 添加到V中。

    存在训练数据中不可能见到所有的字符的问题。可以对字节而不是Unicode（统一码）字符运行BPE算法
3. Unigram model (SentencePiece工具)
    Unigram（一元模型）模型是一种基于统计的语言模型，用于为给定的文本选择最佳的词划分。其目标是通过给定一个文本序列，使用一种合适的分词策略，尽可能优化整个分词的质量。这个分词模型通过目标函数进行优化，能够找到一种最优的分词方式。

    SentencePiece 工具进行分词：
   
    给定一个序列 $x_{1:L}$ ，一个分词器 $T$ 是 $p\left(x_{1: L}\right)=\prod_{(i, j) \in T} p\left(x_{i: j}\right)$ 的一个集合。如这个实例：

    - 训练数据（字符串）： $𝖺𝖻𝖺𝖻𝖼$
    - 分词过程：
        - 通过 SentencePiece 工具进行分词。它将字符字符串 "ababc" 拆分成三个子序列：
            - $T={(1,2)}$ ，即"ab"
            - $T={(3,4)}$ ，即"ab"
            - $T={(5,5)}$ ，即"c"
        - 训练数据的词汇表为 $V=\{𝖺𝖻,𝖼\}$
        - 概率计算： $p(𝖺𝖻)=\frac{2}{3}$ ， $p(𝖼)=\frac{1}{3}$
    - 分词结果  $T={(1,2),(3,4),(5,5)}$ (其中 $V=\{𝖺𝖻,𝖼\}$ ) 
    - 似然值： $p(x_{1:L})= \frac{2}{3}⋅\frac{2}{3}⋅\frac{1}{3}=\frac{4}{27}$ 是根据 unigram 模型计算得出的概率，表示训练数据的似然度，即将训练数据分词为所给的分词结果 $T$的概率。通过将各个词汇的概率相乘得到整个训练数据的似然值为 $\frac{4}{27}$ 。似然值用于评估分词结果的质量。较高的似然值表示训练数据与分词结果之间的匹配程度较高，这意味着该分词结果较为准确或合理。
    - 算法流程：从一个“相当大”的种子词汇表 $V$ 开始。重复以下步骤：
      - 给定 $V$ ，使用EM算法优化 $p(x)$ 和 $T$ 。
      - 计算每个词汇 $x∈V$ 的 $loss(x)$ ，衡量如果将 $x$ 从 $V$ 中移除，似然值会减少多少。
      - 按照 $loss$ 进行排序，并保留 $V$ 中排名靠前的80%的词汇。

### 向量化

需要将词元序列转换为序列的向量形式。 $EmbedToken$ 函数通过在嵌入矩阵 $E∈ℝ^{|v|×d}$ 中查找每个词元所对应的向量，该向量的具体值这是从数据中学习的参数：

def  $EmbedToken(x_{1:L}:V^{L})→ℝ^{d×L}$ ：

- 将序列 $x_{1:L}$ 中的每个词元 $xi$ 转换为向量。
- 返回[Ex1,…,ExL]。

以上的词嵌入是传统的词嵌入，向量内容与上下文无关。这里定义一个抽象的 $SequenceModel$ 函数，它接受这些上下文无关的嵌入，并将它们映射为上下文相关的嵌入。

 $def SequenceModel(x_{1:L}:ℝ^{d×L})→ℝ^{d×L}$ ：

- 针对序列 $x_{1:L}$ 中的每个元素xi进行处理，考虑其他元素。
- [抽象实现（例如， $FeedForwardSequenceModel$ ， $SequenceRNN$ ， $TransformerBlock$ ）]

最简单类型的序列模型基于前馈网络（Bengio等人，2003），应用于固定长度的上下文，就像n-gram模型一样，函数的实现如下：

def  $FeedForwardSequenceModel(x_{1:L}:ℝ^{d×L})→ℝ^{d×L}$ ：

- 通过查看最后 $n$ 个元素处理序列 $x_{1:L}$ 中的每个元素 $xi$ 。
- 对于每个 $i=1,…,L$ ：
  - 计算 $h_{i}=FeedForward(x_{i−n+1},…,x_{i})$ 。
- 返回[ $h_{1},…,h_{L}$ ]。

## 语言模型架构
上下文向量表征 (Contextual Embedding): 作为模型处理的先决条件，其关键是将词元序列表示为响应的上下文的向量表征。

$$
[the, mouse, ate, the, cheese] \stackrel{\phi}{\Rightarrow}\left[\left(\begin{array}{c}
1 \\
0.1
\end{array}\right),\left(\begin{array}{l}
0 \\
1
\end{array}\right),\left(\begin{array}{l}
1 \\
1
\end{array}\right),\left(\begin{array}{c}
1 \\
-0.1
\end{array}\right),\left(\begin{array}{c}
0 \\
-1
\end{array}\right)\right]. 
$$

语言模型分为三个类型：编码端（Encoder-Only），解码端（Decoder-Only）和编码-解码端（Encoder-Decoder）。

- 编码端（Encoder-Only）
    如：BERT、RoBERTa等。这些语言模型生成上下文向量表征，但不能直接用于生成文本。可以表示为， $x_{1:L}⇒ϕ(x_{1:L})$ 。这些上下文向量表征通常用于分类任务（也被称为自然语言理解任务）。任务形式比较简单，下面以情感分类/自然语言推理任务举例：

$$
情感分析输入与输出形式：[[CLS], 他们, 移动, 而, 强大]\Rightarrow 正面情绪
$$

$$
自然语言处理输入与输出形式：[[CLS], 所有, 动物, 都, 喜欢, 吃, 饼干, 哦]⇒蕴涵
$$

该架构的优势是对于文本的上下文信息有更好的理解，因此该模型架构才会多用于理解任务。该架构的有点是对于每个 $x{i}$ ，上下文向量表征可以双向地依赖于左侧上下文 $(x_{1:i−1})$ 和右侧上下文  $(x_{i+1:L})$ 。但是缺点在于不能自然地生成完成文本，且需要更多的特定训练目标（如掩码语言建模）。

<img width="565" alt="image" src="https://github.com/user-attachments/assets/ccb91327-1cf0-4dea-8080-1d3b85995bd2" />

- 解码端（Decoder-Only）
    如：GPT系列模型。这些是常见的自回归语言模型，给定一个提示  $x_{1:i}$ ，它们可以生成上下文向量表征，并对下一个词元 $x_{i+1}$ （以及递归地，整个完成 
 $x_{i+1:L}$） 生成一个概率分布。 $x_{1:i}⇒ϕ(x_{1:i}),p(x_{i+1}∣x_{1:i})$ 。以自动补全任务来说，输入与输出的形式为， $[[CLS], 他们, 移动, 而]⇒强大$ 。与编码端架构比，其优点为能够自然地生成完成文本，有简单的训练目标（最大似然）。缺点也很明显，对于每个  $xi$ ，上下文向量表征只能单向地依赖于左侧上下文  ($x_{1:i−1}$) 。

<img width="530" alt="image" src="https://github.com/user-attachments/assets/c1f26495-12bb-4bd8-b8dd-df2fd806dbc9" />

- 编码-解码端（Encoder-Decoder）
    如：Transformer、BART、T5等模型。这些模型在某种程度上结合了两者的优点：它们可以使用双向上下文向量表征来处理输入 $x_{1:L}$ ，并且可以生成输出 $y_{1:L}$ 。可以公式化为：
    
$$
x1:L⇒ϕ(x1:L),p(y1:L∣ϕ(x1:L))。
$$

以表格到文本生成任务为例，其输入和输出的可以表示为：
    
$$
[名称:, 植物, |, 类型:, 花卉, 商店]⇒[花卉, 是, 一, 个, 商店]。
$$
    
该模型的具有编码端，解码端两个架构的共同的优点，对于每个 $x_{i}$ ，上下文向量表征可以双向地依赖于左侧上下文  $x_{1:i−1}$ ) 和右侧上下文 ( $x_{i+1:L}$ )，可以自由的生成文本数据。缺点就说需要更多的特定训练目标。

<img width="665" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/fc17f20b-726c-4943-966e-df9dc419f54f">

<img width="608" alt="image" src="https://github.com/user-attachments/assets/6e364403-8703-4ef0-a5fb-b7893fc2970a" />

<img width="405" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/d291e84d-3431-45b8-a786-fe5c95c439d5">

<img width="617" alt="image" src="https://github.com/user-attachments/assets/ddccf4ae-d353-4bc8-8087-a4cc62b92b05" />

### 递归神经网络（RNN）

它是一类模型，包括简单的RNN、LSTM和GRU。基本形式的RNN通过递归地计算一系列隐藏状态来进行计算。

<img width="691" alt="image" src="https://github.com/user-attachments/assets/c33104b8-3c8c-4628-965a-9372559370c9" />

def $SequenceRNN(x:ℝ^{d×L})→ℝ^{d×L}$ ：

- 从左到右处理序列 $x_{1},…,x_{L}$ ，并递归计算向量 $h_{1},…,h_{L}$ 。
- 对于 $i=1,…,L$ ：
  - 计算 $h_{i}=RNN(h_{i−1},x_{i})$ 。
  - 返回 $[h_{1},…,h_{L}]$ 。

工作的模块是RNN，类似于有限状态机，它接收当前状态h、新观测值x，并返回更新后的状态：

def $RNN(h:ℝ^d,x:ℝ^d)→ℝ^d$ ：

- 根据新的观测值x更新隐藏状态h。
- [抽象实现（例如，SimpleRNN，LSTM，GRU）]

- 简单RNN：

def $SimpleRNN(h:ℝd,x:ℝd)→ℝd$ ：

- 通过简单的线性变换和非线性函数根据新的观测值 $x$ 更新隐藏状态 $h$ 。
- 返回 $σ(Uh+Vx+b)$ 。

- 双向RNN：

def $BidirectionalSequenceRNN(x_{1:L}:ℝ^{d×L})→ℝ^{2d×L}$ ：

- 同时从左到右和从右到左处理序列。
- 计算从左到右： $[h→_{1},…,h→_{L}]←SequenceRNN(x_{1},…,x_{L})$ 。
- 计算从右到左： $[h←_{L},…,h←_{1}]←SequenceRNN(x_{L},…,x_{1})$ 。
- 返回 $[h→_{1}h←_{1},…,h→_{L}h←_{L}]$ 。

存在问题：简单RNN由于梯度消失的问题很难训练。为了解决这个问题，发展了长短期记忆（LSTM）和门控循环单元（GRU）（都属于RNN）。

## Transformer
Transformer核心模块：注意力。它完全抛弃传统的CNN和RNN，整个网络结构完全由注意力机制组成。

编码器-解码器结构：
- 编码器将输入序列变换为隐藏层特征。在编码器中Q,K,V相同,均为自身前一层的输出；
    - N个堆叠的编码器层：多头注意力 + 前馈网络 + 残差连接和层归一化
      
$$
\begin{aligned}
\boldsymbol{X}_ l' = \text{LayerNorm} ({MHA} (\boldsymbol{X}_ {l - 1}) + \boldsymbol{X}_{l - 1})\\
\boldsymbol{X}_l = \text{LayerNorm} (\text{FFN} (\boldsymbol{X}_l') + \boldsymbol{X}_l')
\end{aligned}
$$

$$
\begin{aligned}
\small
\boldsymbol{X}_ {l - 1} ：编码器第 l - 1 层的输出
\end{aligned}
$$
      
- 解码器将隐藏层特征变换为输出序列。在解码器中Q来自前一层输出,K,V是编码器输出；
    - N个堆叠的解码器层：(掩码)多头注意力 + 前馈网络 + 残差连接和层归一化
      
$$
\begin{aligned}
\boldsymbol{Y}_ l' = \text{LayerNorm} (\text{MaskedMHA} (\boldsymbol{Y}_ {l - 1})+\boldsymbol{Y}_{l - 1})\\
\boldsymbol{Y}_l'' = \text{LayerNorm} (\text{CrossMHA} (\boldsymbol{Y}_l', \boldsymbol{X}_L)+\boldsymbol{Y}_l')\\
\boldsymbol{Y}_l = \text{LayerNorm} (\text{FFN} (\boldsymbol{Y}_l'')+\boldsymbol{Y}_l'')
\end{aligned}
$$

$$
\begin{aligned}
\small
\boldsymbol{Y}_ {l - 1} ：解码器第 l - 1 层的输出 \\
\small
\boldsymbol{X}_{L} ：编码器第 L 层的输出
\end{aligned}
$$

<img width="376" alt="image" src="https://github.com/user-attachments/assets/53aafcce-25e8-4926-81a4-e4a21986fd86" />


Transformer学习资源：

- [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)和[Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)：对Transformer的视觉描述非常好。
- [Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)：Transformer的Pytorch实现。

- high level

  <img width="684" alt="image" src="https://github.com/user-attachments/assets/dba88382-20fb-40d8-ac9a-f6f6fe97a871" />

  <img width="478" alt="image" src="https://github.com/user-attachments/assets/028676ad-762e-47f2-932d-27424d99ab24" />

每个编解码组件是一堆编码器：

  <img width="647" alt="image" src="https://github.com/user-attachments/assets/33bab4b7-4cac-4d89-98be-758134be4e36" />

每个编码器又包含注意力层（self-attention）和前馈神经网络（Feed Forward Neural Network ）两个子层。而每个解码器中前面的两个子层之间加一个注意力层，可帮助解码器专注于输入句子的相关部分（类似于注意力在[seq2seq models](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)).

<img width="658" alt="image" src="https://github.com/user-attachments/assets/e184bf65-16f4-41de-b4ef-ce6dc5a1607e" />

编码器接收向量列表作为输入。它通过将这些向量传递到“自注意力”层，然后传递到前馈神经网络，然后将输出向上发送到下一个编码器来处理此列表。

<img width="687" alt="image" src="https://github.com/user-attachments/assets/605b12cf-2ea9-4aa8-813d-7b650ac6d039" />

### 注意力机制
假设以下句子是要翻译的输入句子：

”The animal didn't cross the street because it was too tired”  “ The animal didn't cross the street because it was too tired ”

这句话中的“它”指的是什么？它指的是街道还是动物？这对人类来说是一个简单的问题，但对算法来说却没有那么简单。

<img width="308" alt="image" src="https://github.com/user-attachments/assets/c8f83709-a4c9-4bc0-a3d7-ed2662302617" />

代码参考：[Tensor2Tensor notebook](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb) ，在其中加载 Transformer 模型，并使用此交互式可视化对其进行检查。

<img width="713" alt="image" src="https://github.com/user-attachments/assets/655df108-72b8-4673-b8c3-c956a7b25990" />

#### 单头注意力
$$
\text{Attention}(Q, K, V)=\text{softmax}\left(\frac{QK^{\mathrm{T}}}{\sqrt{D}}\right)V
$$

步骤：

1. 从编码器的每个输入向量（如本例为每个单词的嵌入）创建Query、Key和Value三个向量。（新向量的维度小于嵌入向量。它们的维度为 64，而嵌入向量和编码器输入/输出向量的维度为 512）

<img width="653" alt="image" src="https://github.com/user-attachments/assets/8c4baadf-27ef-4994-bc38-978979a459fa" />

<img width="395" alt="image" src="https://github.com/user-attachments/assets/178381db-630e-4753-ac40-cdd451590c93" />

2. 计算分数。如计算第一个单词 “Thinking” 的自我关注，需要根据这个单词对输入句子的每个单词进行评分，分数决定了在某个位置对单词进行编码时，将多少注意力放在输入句子的其他部分上。分数的计算方法是将查询向量的点积与正在评分的相应单词的关键向量。因此，如果处理位置 #1 中的单词的自我注意，第一个分数将是 q1 和 k1 的点积，第二个分数将是 q1 和 k2 的点积。

<img width="487" alt="image" src="https://github.com/user-attachments/assets/c8707cdd-cd1e-4954-b228-dd358d8429cd" />

3. 将分数除以 8（缩放维度scale，即除以 $\sqrt{D}$ ， $D$ 为隐藏层特征维度，论文中使用的关键向量维度的平方根 64）。（论文中猜测当隐藏层维度很大时，点积结果会指数增加，会导致softmax函数饱和，即梯度极小，所以缩放会导致更稳定的梯度）
   
4. 通过 softmax。 对分数进行标准化，使其全部为正数，总和为 1。

<img width="560" alt="image" src="https://github.com/user-attachments/assets/c3a70121-e616-4cbb-a615-1870fa83a3eb" />

<img width="496" alt="image" src="https://github.com/user-attachments/assets/4dd9ba41-83d5-49b5-8ff0-ed7d276a2c97" />

5. 将每个值向量乘以 softmax 分数（准备对它们求和）。目的保持想要关注的单词的值完整，并淹没不相关的单词（例如，将它们乘以 0.001 等小数字）。
6. 将加权值向量相加。在此位置（对于第一个单词）产生自我注意层的输出。

<img width="531" alt="image" src="https://github.com/user-attachments/assets/2b185565-02b0-4176-8084-4c8eaa916952" />

$$
score_i = x_i^{\top} W_{\text{key}}^{\top} W_{\text{query}} y
$$

进行指数化和归一化，形成关于词元位置 ${1,…,L}$ 的概率分布：

$$
(α_{1},…,α_{L})=softmax((score_{1},…,score_{L}))
$$

def $Attention(x_{1:L}:ℝ^{d×L},y:ℝ^d)→ℝ^d$ ：

- 通过将其与每个 $x_{i}$ 进行比较来处理 $y$ 。
- 返回

$$
W_{\text{value}} x_{1:L} \cdot \mathrm{softmax} \left( \frac{x_{1:L}^{\top} W_{\text{key}}^{\top} W_{\text{query}}y}{\sqrt{d}} \right)
$$

可以将注意力看作是具有多个方面（例如，句法、语义）的匹配，即多个注意力头（拼接多个注意力头）：

def $MultiHeadedAttention(x_{1:L}:ℝ^{d×L},y:ℝ^{d})→ℝ^{d}$ :

- 通过将其与每个 $x_i$ 与 $nheads$ 个方面进行比较，处理y。
- 返回

$$
W_{\text{output}} \left( \underbrace{\mathrm{Attention}(x_{1:L}, y), \ldots, \mathrm{Attention}(x_{1:L}, y)}_ {n_{\text{heads}} \text{ times}} \right)
$$

#### 多头注意力
- 多头注意力：对于多头注意力，有多组 Query/Key/Value 权重矩阵（Transformer 使用 8 个注意力头，因此最终为每个编码器/解码器提供 8 组）。这些集合中的每一个都是随机初始化的。然后，在训练之后，每个集合用于将输入嵌入（或来自较低编码器/解码器的向量）投影到不同的表示子空间中，从而可以让模型去关注不同方面信息，最后再将各个方面的信息综合起来。
- 多次注意力计算综合的结果类比CNN中同时使用多个卷积核的作用。

$$
\begin{aligned}
\text{MHA}&=\text{Concat}(\text{head}_1,\ldots,\text{head}_N)\boldsymbol{W}^O \\
\text{head}_n&=\text{Attention}(\boldsymbol{X}\boldsymbol{W}_n^Q,\boldsymbol{X}\boldsymbol{W}_n^K,\boldsymbol{X}\boldsymbol{W}_n^V)
\end{aligned}
$$

<img width="630" alt="image" src="https://github.com/user-attachments/assets/f1a556fd-465c-4765-a078-374a3e771dc8" />

进行与上面概述的相同的自我注意计算，只需 8 次不同的权重矩阵，最终得到 8 个不同的 Z 矩阵：

<img width="611" alt="image" src="https://github.com/user-attachments/assets/a6b601ef-bc4a-4aac-a316-ec16de9abd41" />

将这 8 个压缩成一个矩阵：连接矩阵，然后将它们乘以一个额外的权重矩阵 WO。

<img width="683" alt="image" src="https://github.com/user-attachments/assets/999e32cb-1cc4-4840-a253-c490b3d1fac2" />

完整来看就是：

<img width="714" alt="image" src="https://github.com/user-attachments/assets/f41e5978-4720-4455-be72-152a87dbd086" />

<img width="307" alt="image" src="https://github.com/user-attachments/assets/4f0dd2da-8402-49ba-ab60-89044157d30c" />

编码 “it” 这个词时，一个注意力头最关注 “the animal”，而另一个 attention 头关注 “tired” —— 从某种意义上说，模型对 “it” 这个词的表示在一些 “animal” 和 “tired” 的表示中融入了。

<img width="842" alt="image" src="https://github.com/user-attachments/assets/e8014da9-4bb0-4a2f-8bdd-26cf63ad1836" />

- 多头隐式注意力(Multi-Head Latent Attention，MLA)：

由DeepSeek-V2提出，主要目的是降低推理时KVCache的存储开销

<img width="790" alt="image" src="https://github.com/user-attachments/assets/5e4b99ae-d41b-4955-9f7b-e7fce185c0a6" />

<img width="827" alt="image" src="https://github.com/user-attachments/assets/bfeea177-bb4f-4c69-8a59-30138283738c" />

- 硬件优化的注意力机制
    - FlashAttention:通过矩阵分块计算以及减少内存读写次数提高了计算效率
    - PagedAttention:针对解码阶段,对KV Cache进行分块存储并优化计算方式

<img width="660" alt="image" src="https://github.com/user-attachments/assets/feed0de7-f0f5-4393-9535-6b6d0b04d075" />

### 位置编码
添加位置编码向量，为了让模型了解单词的顺序。

<img width="713" alt="image" src="https://github.com/user-attachments/assets/3a55b247-d618-4089-975a-539771952de6" />

<img width="728" alt="image" src="https://github.com/user-attachments/assets/0c106ace-061a-4817-a85e-33ac2ebda0bb" />

假设嵌入的维度为 4，则实际的位置编码将如下所示：

<img width="678" alt="image" src="https://github.com/user-attachments/assets/0b7ede7a-070d-4587-a977-49365a9da04d" />

嵌入大小为 512（列）的 20 个单词（行）的位置编码的真实示例（图中对它们进行了颜色编码，以便可以看到图案）：

<img width="707" alt="image" src="https://github.com/user-attachments/assets/dc3afa37-fc05-48a2-8257-4453fc03e956" />

在图中，每行对应于向量的位置编码。因此，第一行将是将添加到输入序列中第一个单词的嵌入向量。每行包含 512 个值 – 每个值的值介于 1 和 -1 之间。可以看到似乎在中心一分为二。这是因为左半部分的值由一个函数（使用正弦）生成，而右半部分由另一个函数（使用余弦）生成，然后将它们连接起来形成每个位置编码向量。

参考用于生成位置编码的代码： [get_timing_signal_1d()](https://github.com/tensorflow/tensor2tensor/blob/23bd23b9830059fbc349381b70d9429b5c40a139/tensor2tensor/layers/common_attention.py)

两个信号交织在一起的话（[代码](https://github.com/jalammar/jalammar.github.io/blob/master/notebookes/transformer/transformer_positional_encoding_graph.ipynb)）：

<img width="623" alt="image" src="https://github.com/user-attachments/assets/96c44f95-63b4-4338-b36f-64e907e7db40" />

[Self-Attention with Relative Position Representations 使用相对位置表示的自我注意](https://arxiv.org/abs/1803.02155)

#### 旋转位置编码（Rotary Position Embedding，RoPE）：

引入旋转矩阵的位置编码，位置编码的含义是对每一个token的每一个dim赋予不同的位置信息。

def $EmbedTokenWithPosition(x_{1:L}:ℝ^{d×L})$ ：
- 添加位置信息。
- 定义位置嵌入：
  - 偶数维度索引，使用正弦函数： $P_{pos,2i}=sin(pos/10000^{2i/dmodel})$ 
  - 奇数维度索引，使用余弦函数： $P_{pos,2i+1}=cos(pos/10000^{2i/dmodel})$ 
- 返回 $[x_1+P_1,…,x_L+P_L]$ 。

上面的函数中， $pos$ 表示句子中词元的位置， $i$ 表示该词元的向量表示维度位置（索引）， $dmodel$ 是位置向量的维度（通常与模型的隐藏层维度相同，例如512）。公式中的 ${10000^{2i/dmodel}}$ 是一个缩放因子，它随 $i$ 的增大而增大，这样对于不同的 $i$ ，正弦和余弦函数的波长会随之增长。这种设计使得模型能够在每个维度捕捉到不同频率的位置信息。

<img width="875" alt="image" src="https://github.com/user-attachments/assets/175108d7-ac1f-42e2-bc03-7b1827c19a7a" />

Tips：傅里叶变换用一组正弦和余弦函数作为框架，适合分析信号中的频率分量；小波变换使用一组小波函数作为框架，能够同时捕捉信号的频率和时间特性；主成分分析则用数据的多个主成分作为框架，适合降维和提取关键特征。

基于RoPE公式定义:

![image](https://github.com/superkong001/learning_in_datawhale/assets/37318654/58f0f9f6-4d7b-4762-b4b5-826af5259975)

概念：通过旋转编码，使得每个token既有相对位置信息，又有绝对位置信息。

- RoPE本质是实现对特征向量的旋转操作，如果以二维特征向量举例，对于相邻两个token来说，其对应同一个。
- 既能以自注意力矩阵偏置的形式作用于，直接反映两个token的相对位置信息，又能拆解到向量和上，通过直接编码token的绝对位置实现。
- 使用了基于绝对位置信息的旋转矩阵来表示注意力中的相对位置信息，为序列中每个绝对位置设置了特定的旋转矩阵 $\boldsymbol{R}_{\theta,t}$ (位置索引为 $t$ )

<img width="850" alt="image" src="https://github.com/user-attachments/assets/1251d0d3-3988-4911-bc91-f5050e97b490" />

- 在处理query和key向量时，将连续出现的两个元素视为一个子空间
- 每一个子空间 $i$ 所对应的两个元素都会根据一个特定的旋转角度 $t\cdot\theta_i$ 进行旋转
- 根据三角函数的特性，位置索引为 $i$ 的旋转矩阵与位置索引为 $j$ 的旋转矩阵的乘积等于位置索引为相对距离 $i - j$ 的旋转矩阵，即

$$
R_{\theta,i}R_{\theta,j}^{\mathrm{T}} = R_{\theta,i - j}
$$

- 通过这种方式将相对位置信息融入注意力分数

$$
\begin{aligned}
\boldsymbol{q}_ i = \boldsymbol{x}_ i \boldsymbol{W}^Q \boldsymbol{R}_ {\theta,i}\\
\boldsymbol{k}_ j = \boldsymbol{x}_ j \boldsymbol{W}^K \boldsymbol{R}_ {\theta,j}\\
A_{ij} = (\boldsymbol{x}_ i \boldsymbol{W}^Q \boldsymbol{R}_ {\theta,i})(\boldsymbol{x}_ j \boldsymbol{W}^K \boldsymbol{R}_ {\theta,j})^{\mathrm{T}} = \boldsymbol{x}_ i \boldsymbol{W}^Q \boldsymbol{R}_ {\theta,i - j} \boldsymbol{W}^{K^{\mathrm{T}}} \boldsymbol{x}_ j^{\mathrm{T}}
\end{aligned}
$$

代码实现：

<img width="592" alt="image" src="https://github.com/user-attachments/assets/a1a720ef-1e8c-4cd3-97bb-91174eaaed87" />

  其定义为:

![bcfcb5136238da2cca5641a70169cc23_ROPE2](https://github.com/superkong001/learning_in_datawhale/assets/37318654/3e698be4-2a31-43cf-af96-6e50a8b859cd)

可得，其本质就是: $q_{t}$, $k_{s}$ 旋转后的结果，就是 $q_{t}$, $k_{s}$乘上cos再加上 $q_{t}$, $k_{s}$翻转维度并取反一维后乘上sin。
- 对于高纬向量，由于奇、偶数维度两两交错实现较为复杂，则现在可简化为将特征维度一切二，如下图所示，在实现过程中对前后各半进行的操作即为rotate_half操作：

![b9732c2d7d6e7e265bfd933fb481cc9b_ROPE3](https://github.com/superkong001/learning_in_datawhale/assets/37318654/2204dd5d-2fae-4455-9fb0-600c17c3aa11)

a) 生成角度: $$\theta = \left(\frac{1}{10000^{2n/d}}\right)$$

其中，n表示维度数，其取值范围为[0, 1, ..., d/2-1]

b) 将上述生成角度与每一个位置乘积，区分一个seq中各个词：其实等价于:
$$\theta = \left(\frac{i}{10000^{2n/d}}\right)$$  
其中: `i`为行数。

c) emb将二者cat起来，得到dim维度，每dim/2一循环。

d) 在取出位置编码信息cos与sin的时候，就是将seq的部分切出来，原先设置的1024是最大pos编码，每次用的时候只取当下seq_len的即可.之前求得外积，是为了保证seq里面得每一个词都能有不同的1024个位置编码。

e) 进行旋转嵌入。

将 𝑞 视为复数，其中实部和虚部分别是 𝑞 向量的两个分量。 $𝑒^𝑖𝜃$ 是由 cos⁡(𝜃)+𝑖sin⁡(𝜃) 表示的单位复数

复数乘法可以表示为两个复数相乘。如果你把一个复数 𝑎+𝑏𝑖 与另一个复数 𝑐+𝑑𝑖 相乘，结果是 𝑎𝑐−𝑏𝑑+(𝑎𝑑+𝑏𝑐)𝑖。

#### ALiBi 位置编码
- 用于增强Transformer模型的长度外推能力
- 引入了与相对距离成比例关系的惩罚因子来调整注意力分数

$$
A_{ij}=\boldsymbol{x}_i\boldsymbol{W}^Q\boldsymbol{W}^{K^{\mathrm{T}}}\boldsymbol{x}_j^{\mathrm{T}}\boldsymbol{{-m(i - j)}}
$$

- $i - j$ ：查询和键之间的位置偏移量
- $m$ ：每个注意力头独有的惩罚系数

<img width="641" alt="image" src="https://github.com/user-attachments/assets/bfd5996b-66c0-419d-bcd7-b0474c54150d" />

虽然能实现超过上下文长度情况下的外推扩展，但仍然无法保证在超出上下文窗口后对文本的理解能力：

<img width="809" alt="image" src="https://github.com/user-attachments/assets/6444f90c-5d86-4d32-8497-1603ac447cf9" />

#### 扩展位置编码
- 模型在一定长度的数据上训练,超过训练长度的位置编码没有得到充分训练
- 目标：原始上下文窗口 $T_{max}$ 扩展为目标上下文窗口 $T'_{max}$ 

<img width="495" alt="image" src="https://github.com/user-attachments/assets/3a58dbce-db5a-4d81-8423-679c2b5975c9" />

- RoPE扩展

在每个子空间 $i$ 上，相对位置 $t$ 的旋转角度为

$$
f(t, i) = t \cdot \theta_i
$$

通过调整旋转角度 $f(t, i)$ 达到扩展上下文长度的目标：

1. 修改相对位置索引 $t:g(t)$ 

    - 方法一位置内插：将位置索引成比例缩放，保证旋转角度不超过最大值。

      所有位置索引乘以一个小于1的系数，即： $$g(t) = \frac{T_{max}} {T_{max}'} \cdot t $$ 
        
      <img width="677" alt="image" src="https://github.com/user-attachments/assets/1315eb1a-7ba8-45e0-8f5e-d4376a3ba5aa" />
    
    - 方法二位置截断：设置最大距离阈值 $w$ ，阈值内保留，阈值外截断或者插值

      代表方法：ReRoPE（将超过阈值的位置索引设为固定值）和LeakyReRoPE（将超过阈值的位置索引线性内插到原始上下文窗口大小）
      
$$
g(t)=
\begin{cases}
t, & t \leq w;\\ 
w, & t > w 且使用ReRoPE;\\ 
w + \frac{(T_{max} - w)(t - w)}{T'_{max} - w}, & t > w且使用LeakyReRoPE.
\end{cases} \\
$$    

        可以直接应用于更长的上下文而无需重新训练，同时保持正常文本的建模能力，但需要对注意力矩阵做二次计算，增加计算开销

2. 修改旋转基 $\theta_i:h(i)$

关键子空间：波长超过上下文窗口长度的子空间 $(\lambda_i > T_{\text{max}})$ ，无法对完整的旋转周期进行训练，需要调整旋转角度

对旋转基 $\theta_i$ 进行缩放:

$$
f(T'_ {max}, i)=T'_ {max} \cdot h(i) \leq T_{max} \cdot \theta_i
$$

- 方法一修改旋转基的底数 $b (\theta_i = b^{-2(i - 1)/H})$ 
    - 增大底数以减小旋转基 $h(i)=(\alpha\cdot b)^{-(i - 1)/H}\ (\alpha\geq 1)$
    - NTK-RoPE $\alpha=(T_{\text{max}}'/T_{\text{max}})^{H/H - 2}$
    - Dynamic-NTK-RoPE $\alpha=\text{max}(1, T/T_{\text{max}})$ 

<img width="439" alt="image" src="https://github.com/user-attachments/assets/d3321731-3411-4487-9d9c-05f58c54b18f" />

- 方法二旋转基截断

设置两个阈值 $a$ 和 $c$ ，将子空间的基分为三部分

$$
h(i)=
\begin{cases}
\theta_i, & \theta_i\geq c; \quad 较小的基保留原先的值\\
\beta, & c\geq\theta_i\geq a; \quad 处于中间的基设置为一个较小的固定值\\
0, & \theta_i\leq a. \quad 较大的基直接置为0
\end{cases}
$$

有效防止位置索引较大时超出预期分布的旋转角度，提升长度外推能力

削弱了子空间对不同位置索引的分区能力

#### 扩展上下文窗口
采用受限注意力机制实现对长文本的建模

1. 并行上下文窗口
   
  <img width="202" alt="image" src="https://github.com/user-attachments/assets/027085db-332c-4282-8f83-b0244fe82251" />
  
  将文本分成若干片段，每个片段单独编码，生成时关注所有前序词元。代表方法：PCW
  
  <img width="697" alt="image" src="https://github.com/user-attachments/assets/bf1dc7f4-6162-4839-bfc4-7c6a94c91a62" />

2.  $\Lambda$ 型上下文窗口

<img width="202" alt="image" src="https://github.com/user-attachments/assets/ac7b5c83-db0f-427a-afee-8a841e86ba00" />

每个词元仅关注序列最开始和邻近的词元。代表方法：StreamingLLM，LM-infinite

<img width="736" alt="image" src="https://github.com/user-attachments/assets/85cad687-4bbd-4049-935f-7c5534ef7c5e" />

3.  词元选择

   <img width="215" alt="image" src="https://github.com/user-attachments/assets/51c4de68-e5c4-42e8-80a6-49c06157312f" />

- 基于查询与词元相似度的选择
    - 将词元按照距离分为近距离词元和远距离词元(存储于外部向量库)
    - 可通过检索获得最相关的远距离词元,补充远程语义信息
    - 代表方法：Focused Transformer
- 基于查询与分块相似度的选择
    - 对文本分块并压缩为向量表示,使用k近邻方法选择最相关的分块并重新排序词元
    - 代表方法：LongHeads，InfLLM

<img width="841" alt="image" src="https://github.com/user-attachments/assets/820b79be-aa1f-425e-82b7-545456c23f6e" />


### 前馈网络层
学习复杂的函数关系和特征

$$
\text{FFN}(\boldsymbol{X}) = \sigma(\boldsymbol{X}\boldsymbol{W}^U + \boldsymbol{b}_1)\boldsymbol{W}^D+\boldsymbol{b}_2
$$

- 线性变换：先升维、后降维
- 非线性激活函数 $\sigma$ ：ReLU 或 GELU 等

### 残差
通过将输入与输出相加，缓解梯度爆炸和消失。

每个 encoder 中的每个子层 （self-attention，ffnn） 周围都有一个残差连接，然后是[层归一化](https://arxiv.org/abs/1607.06450)步骤。

<img width="518" alt="image" src="https://github.com/user-attachments/assets/04ce24e5-82f8-4019-9d14-9db672f0c403" />

同样适用于 decoder 的子层。如果考虑一个由 2 个堆叠编码器和解码器组成的 Transformer：

<img width="691" alt="image" src="https://github.com/user-attachments/assets/1b43856e-b0ea-44e9-ac1b-a4613ce6d038" />

### 层归一化
对数据进行重新放缩，实现以下目标：

- 不同特征在空间中的尺度不同,对损失优化的影响不一致
- 提升训练稳定性,加速模型收敛

<img width="598" alt="image" src="https://github.com/user-attachments/assets/e9e54851-5174-4fbd-8487-919df231fa82" />

<img width="875" alt="image" src="https://github.com/user-attachments/assets/da550520-b1a6-4b44-bf19-2a14d24f609b" />

RMSNorm代码实现：

<img width="605" alt="image" src="https://github.com/user-attachments/assets/9313325f-249c-4e92-9a78-c5ac3ec98de1" />

Qwen2的RMSNormal：

$$
\text{RMSNorm}(x)=\frac{x}{\sqrt{\frac{1}{n}\sum_{i = 1}^{n}w_{i}^{2}+\epsilon}}
$$

例如：

$$
\begin{bmatrix}
1 & 2 & 3 & 4\\
5 & 6 & 7 & 8\\
9 & 10 & 11 & 12
\end{bmatrix}
$$

要计算这个张量的根均方（RMS）。根均方是通过以下步骤计算的：

步骤1: 计算每个元素的平方

$$
\begin{bmatrix}
1^2 & 2^2 & 3^2 & 4^2 \\
5^2 & 6^2 & 7^2 & 8^2 \\
9^2 & 10^2 & 11^2 & 12^2
\end{bmatrix} =
\begin{bmatrix}
1 & 4 & 9 & 16 \\
25 & 36 & 49 & 64 \\
81 & 100 & 121 & 144
\end{bmatrix}
$$

步骤2: 计算所有元素平方的平均值


将所有上述平方值相加，然后除以总元素数（12个）:

$$
平均值 = \frac{1 + 4+9 + 16+25 + 36+49 + 64+81 + 100+121 + 144}{12}=\frac{650}{12}\approx 54.17
$$

步骤3: 取平均值的平方根以得到RMS

$$
\text{RMS} = \sqrt{54.17}\approx 7.36
$$

层一化模块位置：
1. 层后归一化(Post-Layer Normalization,Post-Norm)

   归一化模块放置于残差计算之后 $Post-Norm(x) = Norm(x +Sublayer(x))$ 。

- 优点：加快训练收敛速度，防止梯度爆炸或梯度消失，降低神经网络对于超参数的敏感性。
- 缺点：可能导致训练不稳定，目前较少单独使用

<img width="162" alt="image" src="https://github.com/user-attachments/assets/8fa23d99-7c70-45e7-9d14-d13d98967c65" />

2. 层前归一化(Pre-Layer Normalization,Pre-Norm)
  
   归一化模块应用在每个子层之前 $Pre-Norm(x) =x+ Sublayer(Norm(x))$ 。

- 缺点：性能略有逊色
- 优点：训练更加稳定，主流模型采用较多

<img width="161" alt="image" src="https://github.com/user-attachments/assets/0838b6c2-9f59-480d-8560-294ff70b46f7" />

3. 夹心归一化(Sandwich-Norm)

    Pre-Norm和Post-Norm两种方法的组合， $Sandwich-Norm(x) = x + Norm(Sublayer(Norm(x)))$ 。

- 理论上更优，但仍无法保证稳定训练

<img width="159" alt="image" src="https://github.com/user-attachments/assets/b77e7270-3e74-4a22-89e0-98dcd9fedd90" />

### 解码（Decoder）端
![transformer_decoding_1](https://github.com/user-attachments/assets/159dde71-fee8-4cc8-a134-c84bfb64cf66)

重复该过程，直到到达一个特殊符号，指示 transformer 解码器已完成其输出。每个步骤的输出在下一个时刻步骤中馈送到底部解码器，解码器就像编码器一样冒泡其解码结果。

![transformer_decoding_2](https://github.com/user-attachments/assets/7fb17afe-4ae6-4c01-899b-4dd82ace0768)

解码器中的自我注意层的工作方式与编码器中的自我注意层略有不同：在 decoder 中，self-attention 层只允许关注 output sequence 中的较早位置。这是通过在自我注意力计算中的 softmax 步骤之前屏蔽未来位置（将它们设置为 -inf）来完成的。

### 输出层：Linear & Softmax 
输出层（全连接 + 归一化指数函数Softmax）目的输出词源的概率分布。

$$
O = \text{softmax}(\boldsymbol{W}^L\boldsymbol{Y}_L)
$$

线性层是一个简单的全连接神经网络，它将解码器堆栈生成的向量投影到一个大得多的向量中，称为 logits 向量。如有个10,000 个英语词汇表（训练之前在预处理阶段创建的），这将使 logits 向量宽 10,000 个单元格 - 每个单元格对应于一个唯一单词的分数。

![image](https://github.com/user-attachments/assets/41269628-4452-4754-ac3d-8661b50d77b5)

然后，softmax 层将这些分数转换为概率（均为正数，加起来均为 1.0）。

![image](https://github.com/user-attachments/assets/0130867f-6579-4764-8ba9-0b1238ed4922)

### 损失函数
将模型输出与实际输出进行比较，然后使用反向传播调整模型的所有权重，使输出更接近所需的输出。

<img width="461" alt="image" src="https://github.com/user-attachments/assets/a8113bc7-b384-40c2-ba10-a0adc9fbf501" />

<img width="454" alt="image" src="https://github.com/user-attachments/assets/a890aacb-0164-4d50-84f2-c8da7d195f22" />

### 激活函数

<img width="762" alt="image" src="https://github.com/user-attachments/assets/c66c5bd2-2e94-446d-bdbf-b1303304da5d" />

### GPT
GPT 系列模型成体系推进：
- 2017年,谷歌提出Transformer
- 2018年,OpenAI提出GPT(1亿+参数)：Decode-only Transformer架构；预训练后针对特定任务微调。
- 2019年,GPT-2(15亿参数)：将任务形式统一为单词预测； $\text{Pr (output | input, task)}$ ；预训练与下游任务一致；使用提示进行无监督任务求解
初步尝试了规模扩展。
- 2020年,GPT-3(1750亿参数)：涌现出上下文学习能力。
- 2021年,CodeX(基于GPT-3,代码预训练)：代码数据训练；推理和代码合成能力。
- 2021年,WebGPT(搜索能力)
- 2022年2月,InstructGPT(人类对齐)：大语言模型与人类价值观对齐；提出RLHF算法。
- 2022年11月,ChatGPT(对话能力)：面向对话进行优化。
- 2023年3月,GPT-4(推理能力、多模态能力)：据说采用MoE架构；推理能力显著提升,建立可预测的训练框架；可支持多模态信息的大语言模型。
- 2024年9月,01(深度思考能力提升)：长思维链推理能力。
- 2025年1月,03(深度思考能力进一步增强)：类似人类“慢思考”过程。

GPT-3 架构的形状（如何分配1750亿个参数）：

- 隐藏状态的维度：dmodel=12288
- 中间前馈层的维度：dff=4dmodel
- 注意头的数量：nheads=96
- 上下文长度：L=2048

$$
GPT-3(x_{1:L})=TransformerBlock^{96}(EmbedTokenWithPosition(x_{1:L}))
$$

不同版本的Transformer之间存在重要但详细的差异：

- 层归一化“后归一化”（原始Transformer论文）与“先归一化”（GPT-2），这影响了训练的稳定性（[Davis等人，2021](http://proceedings.mlr.press/v139/davis21a/davis21a.pdf)）。
- 应用了丢弃（Dropout）以防止过拟合。
- GPT-3使用了[sparse Transformer](https://arxiv.org/pdf/1904.10509.pdf)（稀释 Transformer）来减少参数数量，并与稠密层交错使用。
- 根据Transformer的类型（Encdoer-Only, Decoder-Only, Encdoer-Decoder），使用不同的掩码操作。

### 交叉熵
假设：假设一个城市 75% 的时间都是晴天：

<img width="84" alt="image" src="https://github.com/user-attachments/assets/5fa5db4e-14a6-4b98-b982-757521e84ff0" />

某A有 38% 的时间穿着外套：

<img width="76" alt="image" src="https://github.com/user-attachments/assets/f07b6fb8-97f3-4359-af94-2d68b8b010f9" />

如果这两件事情相对独立（互不影响）：

<img width="274" alt="image" src="https://github.com/user-attachments/assets/054515dd-c08f-472a-91db-06f4bac90fe0" />

图中直线、垂直线和水平线都贯穿始终。这就是独立，A穿着外套并且下周会下雨的概率就是A穿着外套的概率乘以下雨的概率。他们互不影响。

如果相互影响的话；

<img width="270" alt="image" src="https://github.com/user-attachments/assets/5de38dbe-5958-48c6-b7d1-2b972d04209c" />

当关注一个变量，比如天气，知道晴天或下雨的可能性有多大。对于这两种情况，都可以查看条件概率：如果天气晴朗，A穿 T 恤的可能性有多大？如果下雨，A穿外套的可能性有多大？

<img width="377" alt="image" src="https://github.com/user-attachments/assets/9e7b6501-2642-4bf7-bf7e-9ff2788e5e68" />

$p(rain,coat)=p(rain)⋅p(coat | rain)$ 即： $p(x,y)=p(x)⋅p(y|x)$ 

反过来关注穿衣的话： $p(rain,coat)=p(coat)⋅p(rain | coat)$ 

<img width="290" alt="image" src="https://github.com/user-attachments/assets/fa026e16-eda0-48a7-b307-9d7197e7d469" />

扩展：辛普森悖论（Simpson’s Paradox）

- 编码

<img width="239" alt="image" src="https://github.com/user-attachments/assets/615f5625-9879-48d5-b370-65b661e16991" />

将句子中每个单词替换为相应的码字，然后连接成编码字符串：

<img width="241" alt="image" src="https://github.com/user-attachments/assets/40fd9570-e7db-469e-954c-39d381d5e4be" />

根据频率：

<img width="270" alt="image" src="https://github.com/user-attachments/assets/37963044-3d1d-43ad-b530-6a35aebf1f6b" />
<img width="302" alt="image" src="https://github.com/user-attachments/assets/94242f13-2e9c-4bd6-96bd-a5d3ac03a470" />

短码字压缩后（目标常用的 codeword 简短）：

<img width="233" alt="image" src="https://github.com/user-attachments/assets/bdf40eeb-07bb-4ead-9763-daac84e7683d" />

$$
\begin{aligned}
-log_2{\frac{1}{2}} = 1 \\
-log_2{\frac{1}{4}} = 2 \\
-log_2{\frac{1}{8}} = 3 \\
\end{aligned}
$$

<img width="415" alt="image" src="https://github.com/user-attachments/assets/5f3befac-5d54-47b4-90df-bcccc1ad3cca" />

$$
L_{avg} = \sum_x {p(x)L(x)} = (\frac{1}{2} ⋅ 1) + (\frac{1}{4} ⋅ 2) + (\frac{1}{8} ⋅ 3) + (\frac{1}{8} ⋅ 3) = 1.75 bits 
$$

<img width="246" alt="image" src="https://github.com/user-attachments/assets/94e6d2cc-218d-4af1-b74a-5fa250c7e803" />

- 码字空间
添加的每个位都会使可能的代码数量增加一倍：

<img width="297" alt="image" src="https://github.com/user-attachments/assets/b9b366af-61b5-431a-b9f3-1f3c0d99da86" />

短码字后需要增加前缀（prefix codes，以免构成编码字符串的歧义）：

<img width="356" alt="image" src="https://github.com/user-attachments/assets/3958497d-1f5f-419e-8e9c-72b37490981b" />

花在获取短码字上的预算（通过牺牲一小部分可能的码字来支付一个码字的费用）是有限的。

<img width="521" alt="image" src="https://github.com/user-attachments/assets/53fca0f8-b59c-4acb-8c3a-ac43b8bf6014" />

如果成本呈（自然）指数衰减，则它既是高度，也是面积。如果需要在 50% 的时间内发送一个 4 位长的码字，那么平均消息长度比不发送该码字时长 2 位。

<img width="185" alt="image" src="https://github.com/user-attachments/assets/b553b506-f434-4718-87a4-2c6a0d213c08" />

支付的金额决定了码字的长度。码字的长度控制它对平均消息长度的增加程度。

<img width="410" alt="image" src="https://github.com/user-attachments/assets/1901e581-4d13-40b4-93ce-9679d03a4a4e" />

短码字会缩短平均消息长度，但成本高昂，而长码字会增加平均消息长度，但成本低廉。

<img width="485" alt="image" src="https://github.com/user-attachments/assets/18cbcc94-b6cb-463f-97fa-5956d470ffb5" />

- 信息熵
长度 $L$ 的消息的开销为 $\frac{1}{2^L}$ 。反过来获得花费给定金额的消息的长度为 $L(x) = log_2 (\frac{1}{cost})$ ，由于希望符号 $𝑥$ 的出现概率越高，占用的资源就越大，因此 $cost = p(x)$ ， $L(x) = log_2 (\frac{1}{p(x)})=-log_2 {p(x)}$ 。

<img width="421" alt="image" src="https://github.com/user-attachments/assets/a3d21097-d323-4079-a29a-7af868064555" />

信息熵：在信息论中，给定一个真实分布 $𝑝(𝑥)$ ，最佳编码方案应该使用熵 $𝐻(𝑝)$ 作为理论上的最优平均编码长度。

$H(p)=\sum_x {p(x)log_2(\frac{1}{p(x)})} = - \sum_x {p(x)log_2{p(x)}}$

- Cross-Entropy  交叉熵：

<img width="306" alt="image" src="https://github.com/user-attachments/assets/495cb124-777e-44fc-b846-627d89d36ee1" />

使用同一套编码时：

$$
\begin{aligned}
L_\text{avg} = \sum_x {p(x)L(x)} = (\frac{1}{2} ⋅ 1) + (\frac{1}{4} ⋅ 2) + (\frac{1}{8} ⋅ 3) + (\frac{1}{8} ⋅ 3) = 1.75 bits \\
L_\text{avg} = \sum_x {p(x)L(x)} = (\frac{1}{8} ⋅ 1) + (\frac{1}{2} ⋅ 2) + (\frac{1}{4} ⋅ 3) + (\frac{1}{8} ⋅ 3) = 2.25 bits \\
\end{aligned}
$$

将一个分布中的事件与另一个分布的最佳代码进行通信的平均长度，称为交叉熵。也就是不了解真实分布 $𝑝(𝑥)$ 情况下，选择了一个替代分布 $𝑞(𝑥)$ 进行编码，那么编码的平均长度变为：

$$
H_q(p)=- \sum_x {p(x)log_2{q(x)}}
$$

<img width="365" alt="image" src="https://github.com/user-attachments/assets/0561c05b-bee2-4abe-9a7d-91b7999a1ba3" />

以下每个子图代表这 4 种可能性中的一种：

<img width="483" alt="image" src="https://github.com/user-attachments/assets/6ee0f5e1-5633-457e-bd4d-4df118ae787f" />

可以看出交叉熵不是对称的。交叉熵提供了一种表达两种概率分布差异的方法：分布p和q的差异越大，p关于q的交叉熵就越大于p的熵。

<img width="184" alt="image" src="https://github.com/user-attachments/assets/60327ece-4ad9-4817-926b-d07a8dd2086d" />

- Kullback-Leibler 散度
KL 散度就是熵和交叉熵之间的差异。

$$
D_KL(p||q)=H(p,q)-H(p)=\sum_x{p(x)log{\frac{p(x)}{q(x)}}}
​$$

- 联合熵
$X$ 和 $Y$ 的联合熵定义为：

$$
H(p,q)=- \sum_{x,y} {p(x,y)log_2{p(x,y)}}
$$

例子：以穿衣为关注变量，T恤和外套现在是边际概率，即不考虑天气就穿着那件衣服的概率；另一方面，rain和sunny标签，它们的概率分别以穿T恤和穿外套为条件。

<img width="460" alt="image" src="https://github.com/user-attachments/assets/9b98fd89-8175-437d-b326-672b3a1758ef" />

这些概率事件的最佳码字，并计算平均消息长度：

<img width="470" alt="image" src="https://github.com/user-attachments/assets/624322a4-d4ec-4c6d-8fff-4ef45d23f056" />

将代码长度视为第三个维度，现在熵就是体积：

<img width="232" alt="image" src="https://github.com/user-attachments/assets/d7d7f258-80c4-41b5-8587-efe930026c89" />

- 条件熵

$$
H(X|Y)=- \sum_y p(y) \sum_x {p(x|y)log_2{p(x|y)}} = - \sum_{x,y} {p(x,y)log_2{p(x,y)}} 
$$

<img width="310" alt="image" src="https://github.com/user-attachments/assets/6f5ceda8-bc95-448d-8909-8184cb810c78" />

- 互信息

<img width="314" alt="image" src="https://github.com/user-attachments/assets/aa38ba34-8aa4-4e9d-8462-e1c6b909b607" />

<img width="460" alt="image" src="https://github.com/user-attachments/assets/9fb7bf91-f3b2-4da5-9e33-556af4f986f4" />

$H(X,Y)=H(Y)+H(X|Y)$

互信息 $I(X,Y)$ 定义为：

$I(X,Y)=H(X)+H(Y)−H(X,Y)$

信息变化量：

$V(X,Y)=H(X,Y)−I(X,Y)$

<img width="295" alt="image" src="https://github.com/user-attachments/assets/d004f96e-7fa2-42f6-a299-e3eeed4fe8fb" />

#### 理想编码长度
$L(x) = -log_2 {p(x)}$

可以向上取整（Ceil），确保每个符号可以被唯一编码并正确解码。也可以用四舍五入（Round），更贴近真实平均编码长度，但编码实现时仍需调整。

## 新的模型架构
### 混合专家模型 (MoE)
创建一组专家，每个输入只激活一小部分专家。类似一个由专家组成的咨询委员会，每个人都有不同的背景（如历史、数学、科学等）。

$$
\text{input} \quad\quad\Rightarrow\quad\quad \text{expert}_1 \quad \text{expert}_2 \quad \text{expert}_3 \quad \text{expert}_4 \quad\quad\Rightarrow\quad\quad \text{output}.
$$

#### MoE架构
MoE（Mixture of Experts，混合专家）架构是一种先进的深度学习模型设计方法，通过将复杂任务分解为多个子任务，并由多个专家网络协同处理，从而提高模型的灵活性、可扩展性和效率。旨在不显著提升计算成本的同时实现对于模型参数的拓展。它两个主要部分组成：门控网络（Gating Network）和专家网络（Expert Networks）。

- 门控网络：负责根据输入数据动态选择激活哪些专家网络。它通过路由机制决定每个token或输入数据被发送到哪个专家网络，从而实现稀疏激活。
- 专家网络：每个专家网络是一个独立的神经网络，专门处理特定类型的输入数据。它们可以是小型神经网络或特定任务的优化模型

<img width="440" alt="image" src="https://github.com/user-attachments/assets/af99b5c2-6596-45c4-a067-86a0dd24f5d6" />

MoE架构的工作流程：
- 输入分配：输入数据被分配给不同的专家网络。
- 专家处理：每个专家网络独立处理其负责的任务。
- 结果汇总：所有专家网络的输出结果被加权汇总，形成最终输出

<img width="673" alt="image" src="https://github.com/user-attachments/assets/92641eee-c509-46a3-be57-d5d93b3f8a6f" />

<img width="761" alt="image" src="https://github.com/user-attachments/assets/43246051-c5b8-49d1-b517-8b3eafe7473f" />

NLP：

<img width="580" alt="image" src="https://github.com/user-attachments/assets/bb3e6ea1-7d80-4791-8c64-09fa72eab088" />

混合专家架构：

- 包含 $K$ 个由前馈神经网络构成的专家组件 $E_i$ 
- 通过路由网络 $G$ 计算词元 $\boldsymbol{x}_t$ 对应于各个专家的权重

$$
G(\boldsymbol{x}_t)=\text{softmax}(\text{topk}(\boldsymbol{x}_t\cdot\boldsymbol{W}^G))
$$

  - $\boldsymbol{W}^G$ ：将词元映射为 $K$ 个专家的得分
  - $\text{topk}$ ：选择出得分最高的 $k$ 个专家进行激活
  - $\text{softmax}$ ：计算专家权重，未被选择的专家权重被置为0

- 被选择专家的输出加权和，作为该混合专家网络层的最终输出 $\boldsymbol{o}_t$ 

$$
o_t = \text{MoELayer}(x_t)=\sum_{i = 1}^{K}G(x_t)_i\cdot E_i(x_t)
$$

<img width="463" alt="image" src="https://github.com/user-attachments/assets/92068ef2-25b9-433e-bf1e-910158919a8e" />

步骤：模型通过多个专家来做决策，然后根据输入数据决定哪个专家的输出最应该被采纳。
1. 定义专家

设定有 $E$ 个专家，每个专家 $e \in \{1,2,\ldots,E\}$ ，每个专家 $e$ 都有自己的嵌入向量 $w_e \in \mathbb{R}^d$ ，这是专家的特征参数，通常是在训练过程中学习到的。

2. 专家的“门控机制”
通过门控机制来决定选用哪个专家。函数 $g_e(x)$ 来表示专家 $e$ 对输入 $x$ 的重要性（即该专家对输入 $x$ 贡献的概率）。门控函数公式如下：

$$
g_e(x)=\frac{\exp(w_e\cdot x)}{\sum_{e' = 1}^{E}\exp(w_{e'}\cdot x)}
$$

- 这个公式由 softmax 函数派生出来。
-  $w_e\cdot x$ 表示专家 $e$ 和输入 $x$ 的内积，是专家 $𝑒$ 对输入 $𝑥$ 的线性响应，、，反映了专家 $𝑒$ 在给定输入 $𝑥$ 下的重要性，衡量了专家与输入之间的“匹配”程度。即内积的结果越大，说明专家 $e$ 对该输入的重要度越高。
- 每个专家的指数响应 $\exp(w_{e'}\cdot x)$ 会放大匹配程度较大的专家。也就是说，若某个专家 $𝑒$ 与输入 $𝑥$ 的匹配度很高（即 $w_{e'}\cdot x$ 较大），那么 $\exp(w_{e'}\cdot x)$ 就会比其他专家的响应更大，从而使得这个专家在最终的加权和中占更大的比例。
-  $softmax$ 可以确保输出是一个有效的概率分布，即每个专家的贡献（权重）会根据输入 $𝑥$ 的特点动态调整。通过 $softmax$ 操作（分母是对所有专家的响应进行归一化），得到每个专家的权重，即它表示了对输入 $𝑥$ 的“关注度”，通过对每个专家的指数进行归一化，确保每个专家的权重在 0 和 1 之间，并且所有专家的权重之和为 1： $\sum{e = 1}^{E}g_e(x) = 1$ 
-  平衡专家：通过门控函数可以实现平衡专家，即只选择贡献较大的几个专家来进行计算(有效节省计算资源) 或 降低某专家2的权重以避免其被过度使用。 $\tilde g(x) = (\tilde g_1(x), \dots, \tilde g_E(x))$ ，其中大多数专家都是零。因此，在前向和反向传播时，只需要使用非零 $\tilde g_e(x)$ 的专家 $e$ 。

也就是每个专家都会给出一个评分，然后根据这些评分（权重）来决定每个专家对最终输出的贡献程度。

3. 每个专家的参数

每个专家都有自己的参数集 ${\theta}^e = (W_1^e, W_2^e)$ ，其中：
-  $W_1^e$ 是专家 $e$ 的第一部分参数。
-  $W_2^e$ 是专家 $e$ 的第二部分参数。

4. 专家的输出函数

根据每个专家的参数，定义每个专家的输出函数 $h_{\theta_e}(x)$ 。这个函数基于专家的参数 $W_1^e$ 和 $W_2^e$ 来计算输出：

$$
h_{\theta_e}(x) = W_2^e \cdot \max(W_1^e \cdot x,0)
$$

- 这相当于一个带有ReLU激活的线性变换，其中 $\max(W_1^e\cdot x,0)$ 表示如果 $W_1^e\cdot x$ 小于零，则输出为零，反之则输出 $W_1^e\cdot x$ 。
-  $ReLU$ 激活作用是 去除负值

5. 最终混合模型

最终模型的输出是所有专家输出的加权和，也就是说，模型结合了所有专家的贡献。权重由门控函数 $g_e(x)$ 来确定。公式如下：

$$
f(x) = \sum_{e=1}^E \underbrace{g_e(x)}_ \text{gating} \underbrace {h_{\theta_e}(x)}_\text{expert}.
$$

这表示最终模型的输出是每个专家的输出 $h_{\theta_e}(x)$ 按照门控函数 $g_e(x)$ 的权重加权平均得到的。具体来说：
-  $g_e(x)$ 为每个专家的门控函数，表示该专家对输入 $x$ 的贡献程度。
-  $h_{\theta_e}(x)$ 为每个专家的输出函数，它根据专家的参数 $\theta_e$ 计算输出。
-  加权和：每个专家的输出 $ℎ_{\theta_e}(x)$ 被它的门控函数 $g_e(x)$ 权重加权。也就是说，输入 $𝑥$ 会由多个专家共同决定，但各个专家的贡献不同。

#### 训练
训练可以通过反向传播来学习混合专家模型。反向传播的目标是根据损失函数（或目标函数）相对于模型参数的梯度来更新模型参数。在传统的神经网络中，反向传播计算的是模型所有参数的梯度。但在混合专家模型中，因为每个输入 $𝑥$ 并不是都依赖于所有专家，而是根据门控函数 $𝑔_𝑒(𝑥)$ 来动态选择某些专家，这就引入了一个选择性梯度计算的问题。根据链式法则，可以得到：

$$
\nabla f(x) = \sum_{e=1}^E g_e(x) (\nabla (\log g_e(x)) h_{\theta_e}(x) + \nabla h_{\theta_e}(x)).
$$

注意到，梯度与 $g_e(x)$ 成比例，并且同时更新门控函数和专家。

- $\nabla f(x)$ 是要优化的目标函数 $f(x)$ 关于输入 $x$ 的梯度。通常，目标函数是模型的损失函数，通过梯度下降来最小化来优化。
- $g_e(x)$ 是每个专家 $e$ 对输入 $x$ 的门控函数（即权重），通过计算门控函数的梯度，可以调整各个专家对最终输出的贡献。
- $h_e(x)$ 是每个专家 $e$ 的函数输出（根据专家的参数来计算），梯度 $\nabla h_e(x)$ 表示专家输出如何随输入 $x$ 变化。
- $\nabla(\log g_e(x))$ 是门控函数的梯度，它会影响每个专家在最终计算中的权重。它表示门控函数的权重随着输入 $𝑥$ 的变化如何调整，从而影响该专家对最终输出的贡献。通过门控函数的梯度，可以知道输入如何影响专家的选择，即哪个专家对模型输出的重要性更大。
- $\nabla h_{\theta_e}(x)$  是专家函数关于输入 $𝑥$ 的梯度，表示专家函数如何响应输入的变化。

简要来说，这个公式的意思是：通过反向传播计算梯度时，每个专家的贡献会根据其门控函数 $g_e(x)$ 和专家函数 $h_e(x)$ 的梯度进行调整。通过这种方式，反向传播可以更新每个专家的参数，使得最终模型的表现更好。 

- Tips 推导过程：

$$
\begin{aligned}
\nabla_{x} \log(g_e(x)) = \frac{\nabla_{x}g_e(x)}{g_e(x)} \\
g_e(x) \cdot \frac {\nabla_{x} g_e(x)} {g_e(x)} = \nabla_{x} g_e(x) \\ 
\Rightarrow {g_e(x)} \cdot \nabla_{x} \log(g_e(x)) = {\nabla_{x} g_e(x)} 
\end{aligned}
$$

$$
f(x) = \sum_{e = 1}^{E} g_e(x) \cdot h_{\theta_e}(x)
$$

$$
\begin{aligned}
\nabla_{\theta_e} f(x) = \sum_{e = 1}^{E} \nabla_{\theta_e} (g_e(x) \cdot h_{\theta_e}(x)) \\ 
 = \sum_{e = 1}^{E} \nabla_{\theta_e} (g_e(x) \cdot h_{\theta_e}(x) + g_e(x) \cdot \nabla_ {\theta_e}h_{\theta_e}(x)) \\
 = \sum_{e = 1}^{E} {g_e(x)} \cdot \nabla_{\theta_e} \log(g_e(x)) \cdot h_{\theta_e}(x) + g_e(x) \cdot \nabla_ {\theta_e}h_{\theta_e}(x) \\
 = \sum_{e = 1}^{E} {g_e(x)} (\nabla_{\theta_e} \log(g_e(x)) \cdot h_{\theta_e}(x) + \nabla_ {\theta_e}h_{\theta_e}(x))
\end{aligned}
$$

#### 平衡专家
- 设 $c_e = \sum_{i=1}^B \mathbf{1}[\tilde g_e(x_i) > 0]$ 是专家 $e$ 被选中的次数。
- 注意，处理完一个batch后， $\sum_e c_e = B$ 。
- 如果所有专家都是平衡的，那么 $c_e = \frac{B}{E}$ 。
- 溢出：如果 $c_e > 2 \frac{B}{E}$ ，则设 $f(x) = x$ （带残差的旁路），其中2是容量系数。
- 辅助损失：我们期望 $c = [c_1, \dots, c_E]$ 接近均匀分布。
- 我们可以惩罚 $\|c\|_2^2 = \sum_{e=1}^E c_e^2$ ，但这是不可微分的。
- 定义 $m_e = \sum_{i = 1}^B g_e(x_i)$ （这是 $c_e$ 的软版本)。
- 相反，我们在目标函数中添加 $\text{load-balancing-loss} = \sum_{e=1}^E m_e c_e$ 。这样，通过 $m_e$ 的梯度将为非零。

$\text{loss} = \text{negative-log-likelihood} + \lambda \text{load-balancing-loss}.$

例如，可以取 $\lambda = \frac{0.01}{B}$ 。

### 基于检索的模型
有一个原始数据存储库，给定一个新的输入，检索存储库中和它相关的部分，并使用它们来预测输出。类似提问一个问题，然后进行网络搜索，并阅读搜索得到的文档以得出答案。

$$
\text{store} \quad\quad|\quad\quad \text{input} \quad\quad\Rightarrow\quad\quad \text{relevant data from store} \quad \quad\quad\Rightarrow\quad\quad \text{output}.
$$

### DeepSeek

- 训练框架：HAI-LLM
- 语言大模型：DeepSeek LLM/V2/V3、Coder/Coder-V2、Math
- 多模态大模型：DeepSeek-VL
- 推理大模型：DeepSeek-R1

<img width="909" alt="image" src="https://github.com/user-attachments/assets/491fabaf-e3bc-4256-b564-8ce7c980dc24" />

#### HAI-LLM
HAI-LLM(High-Flyer）是一个由DeepSeek团队开发，高效且轻量级的分布式训练框架，主要用于训练和评估大型语言模型（LLM）。具有以下特点和优势：
1. 并行策略：集成了多种并行策略，包括数据并行（Data Parallel）、张量并行（Tensor Parallel）、流水线并行（Pipeline Parallel）以及1F1B流水线并行（1F1B Pipeline Parallel）；数据并行通过ZeRO-1技术优化了激活参数的存储，减少了内存占用。
   
   <img width="716" alt="image" src="https://github.com/user-attachments/assets/ed12fd97-ab35-454a-920d-d4c35fd06ede" />

2. 硬件利用率优化：利用FlashAttention等加速算子提高硬件利用率；通过自研的高性能算子haiscale，显著提升了显存和计算效率。
3. 训练效率：通过优化计算和通信的排程，HAI-LLM有效减少了分布式训练过程中的通信开销，从而提升了训练效率。
4. 灵活性与扩展性：支持数万亿参数规模的超大模型训练，并可扩展至数千个GPU。

- DeepSeek进行了重要的网络架构、训练算法、性能优化探索
    - V1探索了scaling law分析(考虑了数据质量影响),用于预估超参数性能
    - V2提出了MLA高效注意力机制,提升推理性能
    - V2、V3都针对MoE架构提出了相关稳定性训练策略
    - V3使用了MTP(多token预测)训练
    - Math提出了PPO的改进算法GRPO
    - V3详细介绍Infrastructure的搭建方法,并提出了高效FP8训练方法

- DeepSeek-V3
    - 671B参数(37B激活),14.8T训练数据
    - 基于V2的MoE架构,引入了MTP和新的复杂均衡损失
    - 对于训练效率进行了极致优化,共使用2.788M H800 GPU时

<img width="724" alt="image" src="https://github.com/user-attachments/assets/612f1a72-6156-4bff-9789-d442384d979a" />

- DeepSeek R1

<img width="917" alt="image" src="https://github.com/user-attachments/assets/1508d34b-e746-4730-ab41-9bd596dd5c7f" />

### 参数化状态空间模型(State Space Model，SSM)
- RNN和CNN的结合体，利用卷积计算并行编码
- 仅依赖于前一个状态循环推理
- 相比于Transformer长文本建模效率得到极大改进

<img width="813" alt="image" src="https://github.com/user-attachments/assets/ef63d2f3-cec0-41da-ae14-97f4fc77086b" />

<img width="868" alt="image" src="https://github.com/user-attachments/assets/35cb3ca9-c5e4-47b2-914e-4a2d9ee3bdfc" />

- 递归分解当前时刻的输出

<img width="738" alt="image" src="https://github.com/user-attachments/assets/f065b261-4cc6-4cc9-be5b-1dbf6d499593" />

- 相关变种
1. Mamba
    - 引入基于当前输入的信息选择机制
    - 矩阵 $A,B,C$ 表示成基于输入 $x_t$ 的非线性函数,对历史信息进行选择过滤
  
<img width="702" alt="image" src="https://github.com/user-attachments/assets/f08886ac-8ede-4565-8699-fb1515fd722c" />

2. RWKV

    - 词元偏移(Token Shift)：将当前词元和前一个词元线性插值代替当前词元作为输入。
    
    <img width="380" alt="image" src="https://github.com/user-attachments/assets/3456ed98-8379-43c2-8ba9-1091b7606117" />

    - 时间混合模块(Time-Mixing)：代替Transformer中的注意力层；门控RNN，对词元偏移进行更新。

    <img width="362" alt="image" src="https://github.com/user-attachments/assets/fe87e087-b8d0-450e-bf09-8d474bf1be51" />

    - 频道混合模块(Channel-Mixing)：代替Transformer中的前馈网络层；对词元偏移进行映射。
  
    <img width="373" alt="image" src="https://github.com/user-attachments/assets/83e1bf56-acb5-4434-beb5-80fd2a3acd78" />

3. RetNet
使用多尺度保留模块(Multi-scale Retention,MSR)替换多头注意力。

<img width="670" alt="image" src="https://github.com/user-attachments/assets/1fde803f-2b17-4e48-8722-d5749b3f36af" />

4. Hyena
长卷积模块(Long Convolution)替换多头注意力。

- 每层包含 $N$ 个滤波器,每个相对位置索引设置对应的滤波器组成卷积核 $K=(h(1), ... ,h(T))$ 

<img width="785" alt="image" src="https://github.com/user-attachments/assets/823d40b5-8275-40c7-bba9-6c7f9f624410" />

- 对于每个卷积核的输出的中间表示 $z_t$ ，使用门控函数 $g(t)$ 进行加权:

<img width="778" alt="image" src="https://github.com/user-attachments/assets/12d368c7-5cb4-454b-a4f1-28503db479b1" />

# Qwen整体介绍

Qwen的架构：
![2bd108a0a25f60fd7baad3a6ae0d4148_framework](https://github.com/superkong001/learning_in_datawhale/assets/37318654/5cadb050-43c8-48be-b87a-b895db72c411)

其中:
- tokenizer将文本转为词表里面的数值。
- 数值经过embedding得到一一对应的向量。
- attention_mask是用来看见左边、右边，双向等等来设定。
- 各类下游任务，Casual,seqcls等，基本都是基础模型model后面接对应的Linear层，还有损失函数不一样。

```bash
# 拉取huggingface上代码到当前目录
git clone https://github.com/huggingface/transformers.git 

# 安装依赖包
pip install huggingface_hub
pip install transformers
```

```bash
def run_qwen2():
    qwen2config = Qwen2Config(
        vocab_size=151936,
        hidden_size=4096//2,
        intermediate_size=22016//2,
        num_hidden_layers=32//2,
        num_attention_heads=32, #每一头的hidden_dim=2048/32=64
        max_position_embeddings=2048//2      
    )
    qwen2model = Qwen2Model(config=qwen2config)

    input_ids = torch.randint(0, qwen2config.vocab_size, (4,30))

    res = qwen2model(input_ids)
    print(type(res))

if __name__=="__main__":
    run_qwen2()
```

# Qwen2Config
Qwen2Config中包含一些自定义的超参数

```bash
# 初始化参数配置
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0
```

# Qwen2Model类

## 初始化

```bash
输入：tensor[4,30]
input_ids = torch.randint(0, qwen2config.vocab_size, (4,30))

class Qwen2Model(Qwen2PreTrainedModel):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id #指定填充标记的索引
        self.vocab_size = config.vocab_size  #词汇表的大小

        # 嵌入层将输入的标记映射成密集的向量表示
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # 解码器层，包含多个解码器层（16层）
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # 归一化层使用的是 Root Mean Square Layer Normalization
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False #用来节省显存
        # Initialize weights and apply final processing
        self.post_init()  # 对参数进行初始化，以及初始化梯度检查点作用
```

```bash
def post_init(self):
    """
    A method executed at the end of each Transformer model initialization, to execute code that needs the model's
    modules properly initialized (such as weight initialization).
    """
    self.init_weights()
    # 梯度检查点的基本思想是在网络的前向传播过程中不保存所有层的中间激活值（即每一层输出的结果），只有选定的“检查点”层的输出会被保存，从而减少内存占用。
    # 未保存的，需要在反向传播期间重新计算输出。
    self._backward_compatibility_gradient_checkpointing()
```

## 主干Forward, Embedding+Layers(Qwen2DecoderLayer)+Norm

```bash
inputs_embeds = self.embed_tokens(input_ids)  #input: tensor[4,30] （4行30列）, output: tensor[4,30,2048]
# embed positions
hidden_states = inputs_embeds

for decoder_layer in self.layers: #16层循环处理
    # 将所有的hidden_states保存成tuple
    if output_hidden_states:
        all_hidden_states += (hidden_states,)
    # 将hs送入每一层decoder_layer
    if self.gradient_checkpointing and self.training:
        layer_outputs = self._gradient_checkpointing_func(
            decoder_layer.__call__,
            hidden_states,
            attention_mask,
            position_ids,
            past_key_values,
            output_attentions,
            use_cache,
        )
    else:
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
    # 取出上一层decoder_输出的hs,再传入下一个layer
    # 只要第一个,第二个是cache的一个类，然后进入下一个layer
    hidden_states = layer_outputs[0]

    if use_cache:
        next_decoder_cache = layer_outputs[2 if output_attentions else 1]

    if output_attentions:
        all_self_attns += (layer_outputs[1],)
# 将最后layers输出后的hidden_states进行标准化  
hidden_states = self.norm(hidden_states)

# 加上最后一层的hidden_states
if output_hidden_states:
    all_hidden_states += (hidden_states,)
```

- 如果保存output_hidden_states的话，就是第一个为input_ids进行emb，然后保存到n-1层的decoder_layer的输出hs，再加上最后一层layer的输出hs进行过norm后的hs.
- 最后是以BaseModelOutputWithPast的形式输出。

# Qwen2DecoderLayer, attn+MLP+norm

![1725eb39a3bb2bc6b1908c4d6f585a89_decoderlayer](https://github.com/superkong001/learning_in_datawhale/assets/37318654/708fc8ae-d732-4064-9c24-adcff8b5f9ed)

## 初始化

```bash
QWEN2_ATTENTION_CLASSES = {
    "eager": Qwen2Attention, # 一般情况下是这个
    "flash_attention_2": Qwen2FlashAttention2,
    "sdpa": Qwen2SdpaAttention,

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = QWEN2_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

input_layernorm和post_attention_layernorm内容是一样的，只是应用的顺序不一样。

## Forward, Norm+attn+(+residual)+Norm+mlp+(+residual)

- 首先复制一份hidden_states为残差,然后将hidden_states送入Norm,再送入attn模块。
- 得到attn的输出后与前面残差相加（向量逐位相加），再复制一份作为残差，再将hidden_states送入Norm和mlp，再与residual进行相加。最后输出的就是这个hidden_states。

```bash
def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
            )
        residual = hidden_states
        #  RMSNorm标准化后送入attn
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        # 残差与新的hidden_states相加
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
```

# Qwen2Attention

![eb0bcb521d1c092d05a30351a3a3b641_Qwen2Attention](https://github.com/superkong001/learning_in_datawhale/assets/37318654/b2e66e42-8c5a-4da8-8c12-c90223976145)

- num_key_value_heads:表示键值对的头数
- num_key_value_groups:表示键值对的组数，计算为num_heads // num_key_value_headsGQA的实现！！
- q_proj,k_proj,v_proj,o_proj四个Linear操作。后续LoRa也基本都对他动的刀子.

## 初始化

```bash
def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads        
        # max_position_embeddings (`int`, *optional*, defaults to 32768):The maximum sequence length that this model might ever be used with.
        self.max_position_embeddings = config.max_position_embeddings
        # rope_theta (`float`, *optional*, defaults to 10000.0):The base period of the RoPE embeddings.
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
```

## forward, q&k&v proj(nn.Linear) + reshape + rotary_pos_emb  +k&v expand(GQA) + q*kT/hd_d^0.5 + attn_weights加上attention_mask + (softmax + dropout + values_states相乘) + reshape + o_proj

<img width="672" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/c05cdca1-aed1-43b8-ada4-7801fb135bc1">

<img width="661" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/3078c3d0-97da-42bb-a19e-7324b23c9ebd">

<img width="667" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/2061d0a9-fbe5-414e-bbd5-19602a2bbe57">

- 首先将hidden_states送入Linear中得到query、key与value。
- 使用旋转位置嵌入操作rotary_emb，使用了旋转位置嵌入的余弦和正弦部分，将他们与query和key相乘，并将结果相加，从而实现旋转位置嵌入的效果。
- 将key_states和value_states重复group次，再执行dot attn操作。
- 在dot attn操作后得到attn_weights,加上attention_mask从而实现读取掩盖操作，在经过softmax与value_states相乘。得到attn_output。
- 再将上述的attn_output进行reshape操作，送入o_proj，得到最终的输出。

![b6fceb434fbc46d94b0cf3683ff4ea4a_GQA](https://github.com/superkong001/learning_in_datawhale/assets/37318654/43f9acf2-389a-439c-afcf-103567b03389)

主旨:GQA和MQA不需要在推理的过程存储那么多的kv cache, 那么kv cache占用的显存就变小，那么我们LLM serving可以处理的请求数量就更多

解析：

1) 初始张量

```bash
输入：tensor[4, 30](shape:[batch, seq_len]) , headers=32
input_ids = torch.randint(0, qwen2config.vocab_size, (4, 30))
embedding后: tensor[4, 30, 2048](shape:[batch, seq_len, dim]) 


.view(bsz, q_len, self.num_heads, self.head_dim)后: tensor[4, 30, 32, 64](shape:[batch, seq_len, head, head_dim]) 
.transpose(1, 2)后: tensor[4, 32, 30, 64](shape:[batch, head, seq_len, head_dim]) #每一头的hidden_dim=2048/32=64
分别输入到q、k、v
```

```bash
# GQA(grouped-query)情况:
import torch

# shape:(batch, seq_len, head, head_dim)
query = torch.randn(10, 128, 8, 128)
key = torch.randn(10, 128, 2, 128)
value = torch.randn(10, 128, 2, 128)

## 在此设置组数为8/2=4
groups = query.shape[-2] // key.shape[-2]

# key和value都要比query小group倍，但是为在后续做矩阵乘法时方便，我们需要先把key和value的head重复到和query相同的维度。方便后续计算。
# 定义输入x， n_rep是需要重复的次数，在这里一般是组数，输入shape:(batch, head, seq_len, head_dim)
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:

    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    # dont need repeat here means multi head attention
    if n_rep == 1:
        return hidden_states
    # first we expand x to (bs, seq_len, head, group, head_dim)
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    # reshape make head -> head * group
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
```

tips: 为什么要用expand之后再reshape而不能直接用tensor自带的repeat?

- expand 方法用于对张量进行扩展，但不实际分配新的内存。它返回的张量与原始张量共享相同的数据
- repeat 方法通过实际复制数据来扩展张量。它返回的新张量不与原始张量共享数据，扩展后的张量占用了更多的内存。

2) pos_emb, Qwen2RotaryEmbedding + apply_rotary_pos_emb

### Qwen2RotaryEmbedding

相关知识：

- 卷积神经网络（CNN）使用卷积核来捕获单词之间的相对位置信息，但其仅能捕获固定大小的局部上下文信息。
- 循环神经网络（RNN）在处理序列信息上会有更好的效果，其依靠循环结构，将序列信息逐步传递，这其中就引入了单词的位置和顺序信息。但随着序列长度的增加，RNN 会慢慢忘记早前的信息，这就导致了长期依赖问题。除此之外，循环结构也使得 RNN 无法并行计算，这使得 RNN 的训练速度十分缓慢。
- Transformer：由于 Transformer 不包含任何循环结构，各个单词在 Transformer 中都同时经过 Decoder-Encoder 的变换，这就导致了 Transformer 无法捕获单词的位置信息。

Transformer采用的是静态的正弦和余弦波函数的组合，主要提供绝对位置信息。

<img width="533" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/3ee73abe-a686-429e-8491-fa6dc46d9f5d">

这里 𝑝𝑜𝑠 是词在序列中的位置，𝑖 是位置向量中的维度索引，𝑑 是位置向量的维度（通常与模型的隐藏层维度相同，例如512）。这个公式中的 ${10000^{2n/d}}$ 是一个缩放因子，它随 𝑖 的增大而增大，这样对于不同的 𝑖，正弦和余弦函数的波长会随之增长。这种设计使得模型能够在每个维度捕捉到不同频率的位置信息。

旋转位置编码（RoPE）：引入旋转矩阵的位置编码，位置编码的含义是对每一个token的每一个dim赋予不同的位置信息。 公式定义:

![image](https://github.com/superkong001/learning_in_datawhale/assets/37318654/58f0f9f6-4d7b-4762-b4b5-826af5259975)

概念：通过旋转编码，使得每个token既有相对位置信息，又有绝对位置信息。

- 既能以自注意力矩阵偏置的形式作用于,直接反映两个token的相对位置信息，又能拆解到向量和上，通过直接编码token的绝对位置实现。
- RoPE本质是实现对特征向量的旋转操作，如果以二维特征向量举例，对于相邻两个token来说，其对应同一个,其定义为:

![bcfcb5136238da2cca5641a70169cc23_ROPE2](https://github.com/superkong001/learning_in_datawhale/assets/37318654/3e698be4-2a31-43cf-af96-6e50a8b859cd)

可得，其本质就是: $q_{t}$, $k_{s}$ 旋转后的结果，就是 $q_{t}$, $k_{s}$乘上cos再加上 $q_{t}$, $k_{s}$翻转维度并取反一维后乘上sin。
- 对于高纬向量，由于奇、偶数维度两两交错实现较为复杂，则现在可简化为将特征维度一切二，如下图所示，在实现过程中对前后各半进行的操作即为rotate_half操作：

![b9732c2d7d6e7e265bfd933fb481cc9b_ROPE3](https://github.com/superkong001/learning_in_datawhale/assets/37318654/2204dd5d-2fae-4455-9fb0-600c17c3aa11)

```bash
# Copied from transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding with Mistral->Qwen2
class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        # 定义初始值
        self.dim = dim # 旋转嵌入的维度
        self.max_position_embeddings = max_position_embeddings # 最大的位置索引，用于定义最大的序列长度
        self.base = base # 默认10000，计算频率的基数，通常用于调节位置编码的周期性
        # 定义旋转角θn=10000^(−2n/d)，其中n表示维度数，其取值范围为[0, 1, ..., d/2-1]
        # 如：2/64=0.0312，10000^0.0312=1.3335，1/1.3335=7.4989e-01
        # torch.arange(0, self.dim, 2, dtype=torch.int64)生成从0开始到self.dim（但不包括self.dim），步长为2的序列。
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))

        # 注册缓冲区（buffer）. 第一个参数"inv_freq"缓冲区名字，第二个参数 (inv_freq)缓冲区的实际数据，第三个参数 (persistent=False)不保存这个缓冲区
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    # 为seq里面的每个token形成独一无二的旋转角嵌入(外积)
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        # 将前面生成角度(inv_freq)与每一个位置乘积，区分一个seq中各个词
        # torch.outer表示两个向量外积，即第一个向量逐个元素与第二个向量相乘得到每个结果单独保存为一行。
        #  t 的长度为 L（代表序列长度）且 inv_freq 的长度为 D/2（假设 dim=D），那么 freqs 的形状是 L x (D/2)。最终形状为(1024,32)
        freqs = torch.outer(t, self.inv_freq)
        # 生成角度信息(利用注册机制生成self.cos_cached与sin_cached)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # emb将二者cat起来(列方向拼接)，得到dim维度，每dim/2一循环。为一个形状为 L x D (1024, 64)的矩阵，其中 L 是序列长度，D 是编码的完整维度。
        # 通过拼接两份 freqs，可以确保对于每个位置索引 i，有足够的频率值来同时计算其正弦和余弦。
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [batch_size, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # 在取出位置编码信息cos与sin的时候，就是将seq的部分切出来，原先设置的1024是最大pos编码，每次用的时候只取当下seq_len的即可，
        # 之前求得外积，是为了保证seq里面得每一个词都能有不同的1024个位置编码。
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
```

a) 生成角度: $$\theta = \left(\frac{1}{10000^{2n/d}}\right)$$

其中，n表示维度数，其取值范围为[0, 1, ..., d/2-1]

b) 将上述生成角度与每一个位置乘积，区分一个seq中各个词：其实等价于:
$$\theta = \left(\frac{i}{10000^{2n/d}}\right)$$  
其中: `i`为行数。

c) emb将二者cat起来，得到dim维度，每dim/2一循环。

d) 在取出位置编码信息cos与sin的时候，就是将seq的部分切出来，原先设置的1024是最大pos编码，每次用的时候只取当下seq_len的即可.之前求得外积，是为了保证seq里面得每一个词都能有不同的1024个位置编码。

e) 进行旋转嵌入。

将 𝑞 视为复数，其中实部和虚部分别是 𝑞 向量的两个分量。 $𝑒^𝑖𝜃$ 是由 cos⁡(𝜃)+𝑖sin⁡(𝜃) 表示的单位复数

复数乘法可以表示为两个复数相乘。如果你把一个复数 𝑎+𝑏𝑖 与另一个复数 𝑐+𝑑𝑖 相乘，结果是 𝑎𝑐−𝑏𝑑+(𝑎𝑑+𝑏𝑐)𝑖。

```bash
# 后半部分和前半部分进行了交换，并且将后半部分的符号取反。
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    # x1 被定义为张量 x 最后一个维度的前半部分。
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    query and key tensors rotated using the Rotary Position Embedding.
    """
    # unsqueeze(-1)  # 负1表示，在最后一维上添加一个维度
    # 使得维度与查询和键张量匹配，从而可以执行元素级乘法（广播）。
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

### apply_rotary_pos_emb

```bash
# cos.shape(head, head_dim), sin.shape(head, head_dim)
# Copied from transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

4) 矩阵乘法得到score与output 后面就是真正的kqv相乘了

```bash
# GQA(grouped-query)情况:
# 初始shape:(batch, seq_len, head, head_dim) => shape:(batch, head, seq_len, head_dim)
query = query.transpose(1, 2)
# 输入shape:(batch, seq_len, head, head_dim) => shape:(batch, head * n_rep, seq_len, head_dim)
key = repeat_kv(key, 4).transpose(1, 2)
value = repeat_kv(value, 4).transpose(1, 2)

# q*kT/head_dim^0.5 
scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(head_dim) # shape:(batch, head, seq_len, kv_seq_len)
scores = torch.nn.functional.softmax(scores, dim=-1)

# (batch, head, seq_len, kv_seq_len)*(batch, head, seq_len, head_dim)=(batch, head, seq_len, head_dim)
out = torch.matmul(scores, value)
#上一步转置了，还得转回去(batch, seq_len, head, head_dim)
out = out.transpose(1, 2)
```

完整代码：

```bash
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        # 获取形状信息,hidden_states输入的为(bs,T,hd)
        bsz, q_len, _ = hidden_states.size()

        # 对hidden_states进行Linear生成query、key、value
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # reshape多头处理--分块--(bs,T,heads,hd_d)，交换数组的第二个维度（索引为1）和第三个维度
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2] # = q_len
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        # 将旋转位置嵌入应用于查询和键张量。使用了旋转位置嵌入的余弦和正弦部分，将它们与查询和键张量相乘，并将结果相加，从而实现旋转位置嵌入的效果
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # 先将key_states和value_states重复了num_key_value_groups次（GQA）
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 使用dot attn实现q*kT/hd_d^0.5
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # 然后 attn_weights 加上 attention_mask，实现读取顺序
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # softmax + dropout + values_states相乘
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # 转置，修改形状等reshape操作
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # 最后在进行一次o_proj
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        # 返回结果
        return attn_output, attn_weights, past_key_value
```

# Qwen2 MLP

![38d5a025fe702e2d3b1aa624355d90c4_MLP1](https://github.com/superkong001/learning_in_datawhale/assets/37318654/d236cc58-f3bd-4b2b-a591-e5757f211fa7)

输入hidden_state并行送入两个Linear层，其中一个激活一下，再与另一个相乘，最终再经过一个Linear，输出最终结果。

```bash
# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Qwen2
class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

# Qwen2RMSNorm, 根均方归一化

![7d54bafe8e22779a9b9b169b66fe2cea_RMSNorm_formulation](https://github.com/superkong001/learning_in_datawhale/assets/37318654/42f21607-de36-407c-a8d7-75adbacedf3c)

```bash
# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Qwen2
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        # .pow(2).mean(-1, keepdim=True)表示对每个元素求平方，然后计算张量在最后一个维度（由 -1 指定）上的平均值（每一行的平均值）并保持维度
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # rsqrt表示开根的导数
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
```







