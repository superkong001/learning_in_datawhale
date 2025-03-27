参考：
> https://github.com/datawhalechina/tiny-universe/tree/main/content/TinyAgent
> https://github.com/InternLM/Tutorial/blob/main/helloworld/hello_world.md

> 论文：***[ReAct: Synergizing Reasoning and Acting in Language Models](http://arxiv.org/abs/2210.03629)***

![c7c01c11d4f51136806f9dd55c6c934a_Lagent](https://github.com/superkong001/learning_in_datawhale/assets/37318654/c124a941-cfe2-4a7b-8b50-ce0e57032048)

## 规划
### 无反馈规划
规划模块是智能体理解问题并可靠寻找解决方案的关键，它通过分解为必要的步骤或子任务来回应用户请求。任务分解的流行技术包括思维链（COT）和思维树（TOT），分别可以归类为单路径推理和多路径推理。

- “思维链（COT）”的方法：
  它通过分步骤细分复杂问题为一系列更小、更简单的任务，旨在通过增加计算的测试时间来处理问题。这不仅使得大型任务易于管理，而且帮助我们理解模型如何逐步解决问题。

- 思维树（TOT）”方法：
  通过在每个决策步骤探索多个可能的路径，形成树状结构图。这种方法允许采用不同的搜索策略，如宽度优先或深度优先搜索，并利用分类器来评估每个可能性的有效性。

<img width="709" alt="image" src="https://github.com/user-attachments/assets/ed686a71-6f1b-4798-9ca1-e9172b9ab705" />

为了进行任务分解，可以通过不同的途径实现，包括直接利用LLM进行简单提示、采用针对特定任务的指令，或者结合人类的直接输入。这些策略能够根据不同的需求，灵活调整任务的解决方案。而另一种方法则是结合了经典规划器的LLM（简称为LLM+P），该策略依赖外部规划器来进行长期规划。这种方法首先将问题转化为PDDL格式，然后利用规划器生成解决方案，最终将这一解决方案转化回自然语言。这适用于需要详细长期规划的场景，尽管依赖特定领域的PDDL和规划器，可能限制了其适用范围。

![b41226b0942a12faeca517c1af91256c_llm+p](https://github.com/user-attachments/assets/54b4f1ee-acdc-4af3-b1a3-fd0c7bd4ce30)

#### 有反馈规划
上述规划模块不涉及任何反馈，这使得实现解决复杂任务的长期规划变得具有挑战性。为了解决这一挑战，可以利用一种机制，使模型能够根据过去的行动和观察反复思考和细化执行计划。目标是纠正并改进过去的错误，这有助于提高最终结果的质量。这在复杂的现实世界环境和任务中尤其重要，其中试错是完成任务的关键。这种反思或批评机制的两种流行方法包括[ReAct](https://github.com/ysymyth/ReAct)和[Reflexion](https://github.com/noahshinn/reflexion)。

![10aa1e82d91c9e227af965672e8e2d07_act](https://github.com/user-attachments/assets/1d81bb5f-b5e7-4e7b-a38a-d614034babcd)

[ReAct](https://github.com/ysymyth/ReAct)方法提出通过结合特定任务的离散动作与语言描述，实现了在大规模语言模型（LLM）中融合推理与执行的能力。离散动作允许LLM与其环境进行交互，如利用Wikipedia搜索API，而语言描述部分则促进了LLM产生基于自然语言的推理路径。这种策略不仅提高了LLM处理复杂问题的能力，还通过与外部环境的直接交互，增强了模型在真实世界应用中的适应性和灵活性。此外，基于自然语言的推理路径增加了模型决策过程的可解释性，使用户能够更好地理解和校验模型行为。ReAct设计亦注重模型行动的透明度与控制性，旨在确保模型执行任务时的安全性与可靠性。因此，ReAct的开发为大规模语言模型的应用提供了新视角，其融合推理与执行的方法为解决复杂问题开辟了新途径。

[Reflexion](https://github.com/noahshinn/reflexion)是一个框架，旨在通过赋予智能体动态记忆和自我反思能力来提升其推理技巧。该方法采用标准的强化学习（RL）设置，其中奖励模型提供简单的二元奖励，行动空间遵循ReAct中的设置，即通过语言增强特定任务的行动空间，以实现复杂的推理步骤。每执行一次行动后，智能体会计算一个启发式评估，并根据自我反思的结果，可选择性地重置环境，以开始新的尝试。启发式函数用于确定轨迹何时效率低下或包含幻觉应当停止。效率低下的规划指的是长时间未成功完成的轨迹。幻觉定义为遭遇一系列连续相同的行动，这些行动导致在环境中观察到相同的结果。

![bfadbcf4e38a3f56e9234f8a536ab3a0_reflection](https://github.com/user-attachments/assets/b27158a7-866e-46a8-ac54-613110595bcd)


# Step 1: 构造大模型

```python
import os
from typing import Any, List, Optional, Dict, Tuple, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
class BaseModel:
    def __init__(self, model_path: str = '') -> None:
        self.model_path = model_path

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        pass

    def load_model(self):
        pass

    @property
    def _llm_type(self) -> str:
        return "BaseModel"
```

创建一个InternLM2类，这个类继承自BaseModel类，类中实现chat方法和load_model方法。

```python
class InternLM2Chat(BaseModel):
    def __init__(self, model_path: str = '') -> None:
        # model_path: InternLM 模型路径
        # 从本地初始化模型
        super().__init__(model_path)
        self.load_model()

    def chat(self, prompt: str, history: List = [], content: str='', meta_instruction:str ='') -> str:
        prompt = PROMPT_TEMPLATE['InternLM_PROMPT_TEMPALTE'].format(question=prompt, context=content)
        response, history = self.model.chat(self.tokenizer, prompt, history, temperature=0.1, meta_instruction=meta_instruction)
        return response, history


    def load_model(self):
        print("正在从本地加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.float16, trust_remote_code=True).cuda()
        self.model = self.model.eval()
        print("完成本地模型的加载")

    @property
    def _llm_type(self) -> str:
        return "InternLM"
```

# Step 2: 构造工具

Tips: 使用Google搜索功能需要去`serper`官网申请一下`token`: https://serper.dev/dashboard， 然后在tools.py文件中填写你的key，这个key每人可以免费申请一个，且有2500次的免费调用额度，足够做实验用啦~

构造一个Tools类，其中添加一些工具的描述信息和具体实现方式。添加工具的描述信息，是为了在构造system_prompt的时候，让模型能够知道可以调用哪些工具，以及工具的描述信息和参数。

```python
class Tools:
    def __init__(self) -> None:
        self.toolConfig = self._tools()
    
    def _tools(self):
        tools = [
            {
                'name_for_human': '谷歌搜索',
                'name_for_model': 'google_search',
                'description_for_model': '谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。',
                'parameters': [
                    {
                        'name': 'search_query',
                        'description': '搜索关键词或短语',
                        'required': True,
                        'schema': {'type': 'string'},
                    }
                ],
            }
        ]
        return tools

    def google_search(self, search_query: str):
        pass
```

# Step 3: 构造Agent

Agent是一个React范式的Agent，Agent类中实现了text_completion方法，这个方法是一个对话方法，通过调用大语言模型，然后根据React的Agent的逻辑，来调用Tools中的工具。

## 构造system_prompt

system_prompt告诉大模型，可以调用哪些工具，以什么样的方式输出，以及工具的描述信息和工具应该接受什么样的参数。

```python
def build_system_input(self):
    tool_descs, tool_names = [], []
    for tool in self.tool.toolConfig:
        tool_descs.append(TOOL_DESC.format(**tool))
        tool_names.append(tool['name_for_model'])
    tool_descs = '\n\n'.join(tool_descs)
    tool_names = ','.join(tool_names)
    sys_prompt = REACT_PROMPT.format(tool_descs=tool_descs, tool_names=tool_names)
    return sys_prompt
```

React结构：第一次解析用户的提问，选择调用的工具和参数，第二次将工具返回的结果与用户的提问整合。

## 构造Agent

```python
class Agent:
    def __init__(self, path: str = '') -> None:
        pass

    def build_system_input(self):
        # 构造上文中所说的系统提示词
        pass
    
    def parse_latest_plugin_call(self, text):
        # 解析第一次大模型返回选择的工具和工具参数
        pass
    
    def call_plugin(self, plugin_name, plugin_args):
        # 调用选择的工具
        pass

    def text_completion(self, text, history=[]):
        # 整合两次调用
        pass
```

# Step 4: 运行Agent

```python
from Agent import Agent


agent = Agent('/root/share/model_repos/internlm2-chat-20b')

response, _ = agent.text_completion(text='你好', history=[])
print(response)

# Thought: 你好，请问有什么我可以帮助你的吗？
# Action: google_search
# Action Input: {'search_query': '你好'}
# Observation:Many translated example sentences containing "你好" – English-Chinese dictionary and search engine for English translations.
# Final Answer: 你好，请问有什么我可以帮助你的吗？ 

response, _ = agent.text_completion(text='周杰伦是哪一年出生的？', history=_)
print(response)

# Final Answer: 周杰伦的出生年份是1979年。 

response, _ = agent.text_completion(text='周杰伦是谁？', history=_)
print(response)

# Thought: 根据我的搜索结果，周杰伦是一位台湾的创作男歌手、钢琴家和词曲作家。他的首张专辑《杰倫》于2000年推出，他的音乐遍及亚太区和西方国家。
# Final Answer: 周杰伦是一位台湾创作男歌手、钢琴家、词曲作家和唱片制作人。他于2000年推出了首张专辑《杰伦》，他的音乐遍布亚太地区和西方国家。他的音乐风格独特，融合了流行、摇滚、嘻哈、电子等多种元素，深受全球粉丝喜爱。他的代表作品包括《稻香》、《青花瓷》、《听妈妈的话》等。 

response, _ = agent.text_completion(text='他的第一张专辑是什么？', history=_)
print(response)

# Final Answer: 周杰伦的第一张专辑是《Jay》。 
```

