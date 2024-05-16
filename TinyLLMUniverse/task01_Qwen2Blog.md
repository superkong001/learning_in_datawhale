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

## forward, 

<img width="672" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/c05cdca1-aed1-43b8-ada4-7801fb135bc1">

<img width="661" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/3078c3d0-97da-42bb-a19e-7324b23c9ebd">

<img width="667" alt="image" src="https://github.com/superkong001/learning_in_datawhale/assets/37318654/2061d0a9-fbe5-414e-bbd5-19602a2bbe57">

- 首先将hidden_states送入Linear中得到query、key与value。
- 使用旋转位置嵌入操作rotary_emb，使用了旋转位置嵌入的余弦和正弦部分，将他们与query和key相乘，并将结果相加，从而实现旋转位置嵌入的效果。
- 将key_states和value_states重复group次，再执行dot attn操作。
- 在dot attn操作后得到attn_weights,加上attention_mask从而实现读取掩盖操作，在经过softmax与value_states相乘。得到attn_output。
- 再将上述的attn_output进行reshape操作，送入o_proj，得到最终的输出。

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

        kv_seq_len = key_states.shape[-2]
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

        # 先将key_states和value_states重复了num_key_value_groups次
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

# Qwen2RMSNorm

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
        # .pow(2).mean(-1, keepdim=True)表示对最后一个维度平方并取均值
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # rsqrt表示开根的导数
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
```





