参考：
> https://github.com/datawhalechina/tiny-universe/tree/main/content/TinyAgent
> https://github.com/InternLM/Tutorial/blob/main/helloworld/hello_world.md

> 论文：***[ReAct: Synergizing Reasoning and Acting in Language Models](http://arxiv.org/abs/2210.03629)***

![c7c01c11d4f51136806f9dd55c6c934a_Lagent](https://github.com/superkong001/learning_in_datawhale/assets/37318654/c124a941-cfe2-4a7b-8b50-ce0e57032048)

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

