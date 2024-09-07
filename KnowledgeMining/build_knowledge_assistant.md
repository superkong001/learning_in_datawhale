# 参考

- 茴香豆

> [茴香豆Github](https://github.com/InternLM/HuixiangDou/tree/main)
> [[茴香豆web搭建]](https://github.com/InternLM/HuixiangDou/tree/main/web)
> [茴香豆本地搭建实战](https://github.com/InternLM/Tutorial/blob/camp3/docs/L2/Huixiangdou/readme.md)
> [茴香豆零编程接入微信](https://zhuanlan.zhihu.com/p/686579577)
> [茴香豆web](https://openxlab.org.cn/apps/detail/tpoisonooo/huixiangdou-web)

- 医疗知识问答

> [[以疾病为中心的一定规模医药领域知识图谱，并以该知识图谱完成自动问答与分析服务]](https://github.com/TommyZihao/QASystemOnMedicalKG/blob/master/README.md)

- Coze搭建A1客服助手

# 基于茴香豆

<img width="528" alt="image" src="https://github.com/user-attachments/assets/c3c957f3-5857-4306-a78d-4c5a958c348e">

<img width="470" alt="image" src="https://github.com/user-attachments/assets/d2ce9137-2b25-418e-8fe9-22331c1ae3e2">

# [基于Coze](https://www.coze.cn/home)

- 人设

你是xxxx有限公司的智能客服; 

- 提示词

请注意！当业务资料不能直接回答客户提问的时候，你只能回复:抱歉，这个问题我不知道；
任务：根据业务资料【{{knowledge}}】来回答客户的提问【{{input}}】；
要求：输出内容需要你整理一下格式,用emoji表情让内容更加易读；

Tips:

- 测试工作流的三个步骤

1) 直接问业务，要求智能客服介绍；
2) 描述需求匹配业务，但不要出现关键词；
3) 问一个原本没有的业务

- 四个锦囊

1) 请注意！当业务资料里没有能回答问题的资料时，一律回答：抱歉，这个问题我不知道
2) 根据情况添加禁止语句
3) 单独加一页概述/目录页
4) 上传到知识库，最好的是Q&A文档


