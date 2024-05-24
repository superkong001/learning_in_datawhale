参考：
> https://github.com/datawhalechina/tiny-universe/tree/main/content/TinyAgent
> https://github.com/InternLM/Tutorial/blob/main/helloworld/hello_world.md

> 论文：***[ReAct: Synergizing Reasoning and Acting in Language Models](http://arxiv.org/abs/2210.03629)***

![c7c01c11d4f51136806f9dd55c6c934a_Lagent](https://github.com/superkong001/learning_in_datawhale/assets/37318654/c124a941-cfe2-4a7b-8b50-ce0e57032048)

# Step 1: 构造大模型

class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict]):
        pass

    def load_model(self):
        pass

> 使用Google搜索功能的话需要去`serper`官网申请一下`token`: https://serper.dev/dashboard， 然后在tools.py文件中填写你的key，这个key每人可以免费申请一个，且有2500次的免费调用额度，足够做实验用啦~

