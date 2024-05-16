- https://github.com/datawhalechina/tiny-universe
- https://github.com/huggingface/transformers/tree/v4.39.3/src/transformers/models/qwen2

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
class Qwen2Model(Qwen2PreTrainedModel):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id #指定填充标记的索引
        self.vocab_size = config.vocab_size  #词汇表的大小

        # 嵌入层将输入的标记映射成密集的向量表示
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # 解码器层，包含多个解码器层
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # 归一化层使用的是 Root Mean Square Layer Normalization
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False #用来节省显存
        # Initialize weights and apply final processing
        self.post_init()  # 对参数进行初始化，以及初始化梯度检查点作用
```


