引用参考：

- https://github.com/datawhalechina/tiny-universe
- https://github.com/huggingface/transformers/tree/v4.39.3/src/transformers/models/qwen2
- [https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch0](https://github.com/datawhalechina/so-large-lm/tree/main)

# 基础概念
## 语言模型
语言模型两类问题：

1. 输入序列问题：输入是文本信号，而计算机能进入神经网络处理和计算的是数值，所以需要讲字符通过一定方式转化为数值。 如：独热编码。
2. 输出序列问题：输出要求是文本，而神经网络的输出是数值类型的（分类问题：二分类问题对应01输出，多分类对应多个01输出；回归问题：对应数值类型输出），所以需要建立神经网络的数值类型输出和最终字符输出的映射关系。如：构建神经网络的输出独热编码后每个字符的概率，选取最高的那个。

语言模型（LM）的经典定义是一种对词元序列(token)的概率分布。假设我们有一个词元集的词汇表 $V$ 。语言模型p为每个词元序列 $x_{1},...,x_{L}$ ∈ $V$ 分配一个概率（介于0和1之间的数字）：

$$
p(x_1, \dots, x_L)
$$

概率直观地告诉我们一个标记序列有多“好（good）”。例如，如果词汇表为{ate, ball, cheese, mouse, the}，语言模型可能会分配以下概率（演示）：

$$
p(\text{the, mouse, ate, the, cheese}) = 0.02,
$$

$$
p(\text{the, cheese ate, the, mouse}) = 0.01,
$$

$$
p(\text{mouse, the, the, cheese, ate}) = 0.0001,
$$

- 语言模型是序列  $x_{1:L}$ 的概率分布 p。
- 一个好的语言模型应具有语言能力和世界知识。
- 自回归语言模型允许有效地生成给定提示 $x_{1:i}$ 的补全 $x_{i+1:L}$。
- 温度可以用来控制生成中的变异量。

如：语言模型应该隐含地赋予"𝗆𝗈𝗎𝗌𝖾 𝗍𝗁𝖾 𝗍𝗁𝖾 𝖼𝗁𝖾𝖾𝗌𝖾 𝖺𝗍𝖾"一个非常低的概率，因为它在语法上是不正确的（句法知识）。由于世界知识的存在，语言模型应该隐含地赋予"𝗍𝗁𝖾 𝗆𝗈𝗎𝗌𝖾 𝖺𝗍𝖾 𝗍𝗁𝖾 𝖼𝗁𝖾𝖾𝗌𝖾"比"𝗍𝗁𝖾 𝖼𝗁𝖾𝖾𝗌𝖾 𝖺𝗍𝖾 𝗍𝗁𝖾 𝗆𝗈𝗎𝗌𝖾"更高的概率。这是因为两个句子在句法上是相同的，但在语义上却存在差异，而语言模型需要具备卓越的语言能力和世界知识，才能准确评估序列的概率。

语言模型也可以做生成任务。如定义所示，语言模型p接受一个序列并返回一个概率来评估其好坏。我们也可以根据语言模型生成一个序列。最纯粹的方法是从语言模型$p$中以概率 $p(x_{1:L})$ 进行采样，表示为：

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
\begin{align*} p({the}, {mouse}, {ate}, {the}, {cheese}) = \, & p({the}) \\ & p({mouse} \mid {the}) \\ & p({ate} \mid {the}, {mouse}) \\ & p({the} \mid {the}, {mouse}, {ate}) \\ & p({cheese} \mid {the}, {mouse}, {ate}, {the}). \end{align*}
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

### 大模型理论

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

如: trigram（n=3）模型会定义：

$$
p(𝖼𝗁𝖾𝖾𝗌𝖾∣𝗍𝗁𝖾,𝗆𝗈𝗎𝗌𝖾,𝖺𝗍𝖾,𝗍𝗁𝖾)=p(𝖼𝗁𝖾𝖾𝗌𝖾∣𝖺𝗍𝖾,𝗍𝗁𝖾)
$$

概率是基于各种 $\text{n-gram}$（例如，𝖺𝗍𝖾 𝗍𝗁𝖾 𝗆𝗈𝗎𝗌𝖾和𝖺𝗍𝖾 𝗍𝗁𝖾 𝖼𝗁𝖾𝖾𝗌𝖾）在大量文本中出现的次数计算的，并且适当地平滑以避免过拟合（例如，Kneser-Ney平滑）。

存在问题，例如以下的前缀：

```
𝖲𝗍𝖺𝗇𝖿𝗈𝗋𝖽 𝗁𝖺𝗌 𝖺 𝗇𝖾𝗐 𝖼𝗈𝗎𝗋𝗌𝖾 𝗈𝗇 𝗅𝖺𝗋𝗀𝖾 𝗅𝖺𝗇𝗀𝗎𝖺𝗀𝖾 𝗆𝗈𝖽𝖾𝗅𝗌. 𝖨𝗍 𝗐𝗂𝗅𝗅 𝖻𝖾 𝗍𝖺𝗎𝗀𝗁𝗍 𝖻𝗒 ___
```

如果 $n$ 太小，那么模型将无法捕获长距离的依赖关系，下一个词将无法依赖于𝖲𝗍𝖺𝗇𝖿𝗈𝗋𝖽。然而，如果 $n$ 太大，统计上将无法得到概率的好估计（所有合理的长序列都出现0次）：

$$
count(𝖲𝗍𝖺𝗇𝖿𝗈𝗋𝖽,𝗁𝖺𝗌,𝖺,𝗇𝖾𝗐,𝖼𝗈𝗎𝗋𝗌𝖾,𝗈𝗇,𝗅𝖺𝗋𝗀𝖾,𝗅𝖺𝗇𝗀𝗎𝖺𝗀𝖾,𝗆𝗈𝖽𝖾𝗅𝗌)=0
$$

因此，语言模型被限制在如语音识别和机器翻译等任务中，其中声音信号或源文本提供了足够的信息，只捕获局部依赖关系（而无法捕获长距离依赖关系）。

### 神经语言模型

神经语言模型，其中 $p(x_{i}∣x_{i−(n−1):i−1})$ 由神经网络给出：

$$
p(cheese∣ate,the)=\text{some-neural-network(ate,the,cheese)}
$$

训练神经网络在统计上是高效的，但在计算上是低效的，因为它在计算上要昂贵得多，而在相同数据量上优于n-gram模型。但由于n-gram模型的扩展性更好，且数据并非瓶颈，所以n-gram模型在至少接下来的十年中仍然占主导地位。

- **Recurrent Neural Networks**（RNNs），包括长短期记忆（LSTMs），使得一个词元 $x_{i}$ 的条件分布可以依赖于整个上下文 $x_{1:i−1}$ （有效地使 $n=∞$ ），但这些模型难以训练。
- **Transformers**是2017年为机器翻译开发，再次返回固定上下文长度n，但更易于训练（并利用了GPU的并行性）。

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

所以，在 $\epsilon=0$ 处：

$$
\frac{1}{1 - \epsilon} \approxeq 1 + \epsilon + \epsilon^2 + \epsilon ^3 + ...
$$

在 $\epsilon$ 很小的时候， $\epsilon ^2$ 和更高阶的项变得非常小，因此可以忽略它们。

## 模型架构
根据输入需求的语言描述（prompt）生成符合需求的结果（completion），形式表达为：

$$
prompt \overset{model}{\leadsto} completion \ \ or \ \ model(prompt) = completion
$$

#### 分词
分词（Tokenization）：即如何将一个字符串拆分成多个词元。分词方法：

1. 基于空格的分词。如：`text.split(' ')`
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
4. Unigram model (SentencePiece工具)
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

#### 向量化

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

### 语言模型架构
上下文向量表征 (Contextual Embedding): 作为模型处理的先决条件，其关键是将词元序列表示为响应的上下文的向量表征：

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

- 解码端（Decoder-Only）
    如：GPT系列模型。这些是常见的自回归语言模型，给定一个提示  $x_{1:i}$ ，它们可以生成上下文向量表征，并对下一个词元 $x_{i+1}$ （以及递归地，整个完成 
 $x_{i+1:L}$） 生成一个概率分布。 $x_{1:i}⇒ϕ(x_{1:i}),p(x_{i+1}∣x_{1:i})$ 。以自动补全任务来说，输入与输出的形式为， $[[CLS], 他们, 移动, 而]⇒强大$ 。与编码端架构比，其优点为能够自然地生成完成文本，有简单的训练目标（最大似然）。缺点也很明显，对于每个  $xi$ ，上下文向量表征只能单向地依赖于左侧上下文  ($x_{1:i−1}$) 。

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

<img width="405" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/d291e84d-3431-45b8-a786-fe5c95c439d5">

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

Transformer采用的是静态的正弦和余弦波函数的组合，主要提供绝对位置信息:

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

<img width="530" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/a3664578-23ad-46b8-9f6e-d0e43f04e9bb">






