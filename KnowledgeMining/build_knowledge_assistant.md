# 参考

- 茴香豆

> [茴香豆Github](https://github.com/InternLM/HuixiangDou/tree/main)
> [茴香豆web搭建](https://github.com/InternLM/HuixiangDou/tree/main/web)
> [茴香豆本地搭建实战](https://github.com/InternLM/Tutorial/blob/camp3/docs/L2/Huixiangdou/readme.md)
> [茴香豆零编程接入微信](https://zhuanlan.zhihu.com/p/686579577)
> [茴香豆web](https://openxlab.org.cn/apps/detail/tpoisonooo/huixiangdou-web)

- 医疗知识问答

> [以疾病为中心的一定规模医药领域知识图谱，并以该知识图谱完成自动问答与分析服务](https://github.com/TommyZihao/QASystemOnMedicalKG/blob/master/README.md)

- Coze搭建A1客服助手

# 基于茴香豆

## web 版茴香豆

<img width="435" alt="image" src="https://github.com/user-attachments/assets/a55a7ac4-3a06-4e1a-bc89-9887370b96ac">

三阶段 Pipeline （前处理、拒答、响应）:

<img width="528" alt="image" src="https://github.com/user-attachments/assets/c3c957f3-5857-4306-a78d-4c5a958c348e">

<img width="470" alt="image" src="https://github.com/user-attachments/assets/d2ce9137-2b25-418e-8fe9-22331c1ae3e2">

## 茴香豆本地标准版搭建


# [基于Coze智能客服](https://www.coze.cn/home)

- 人设

你是xxxx有限公司的智能客服;

你是一名擅长民法典、刑法、道路交通安全法、数据安全法相关法律问题咨询专家；

- 提示词

请注意！当业务资料不能直接回答客户提问的时候，你只能回复:抱歉，这个问题我不知道；
任务：根据业务资料【{{knowledge}}】来回答客户的提问【{{input}}】；
要求：输出内容需要你整理一下格式,用emoji表情让内容更加易读；

请注意！当法律资料不能直接回答客户提问的时候，你只能回复:抱歉，这个问题我不知道； 
任务：根据法律资料【{{KDD_output}}】来回答客户的提问【{{USER_INPUT}}】； 
要求：输出内容需要你整理一下格式,用emoji表情让内容更加易读；

Q：老人过世，父母双亡，只留有一妻子和一个女儿，有多个兄弟姐妹，在老人没有留下遗嘱情况下，财产如何继承？

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

实战：
1. 建立知识库

  创建知识库：
  
   <img width="401" alt="image" src="https://github.com/user-attachments/assets/937bd39f-8352-423a-8808-15f96c3239cb">

  上传文本：

    <img width="469" alt="image" src="https://github.com/user-attachments/assets/0b72ebe2-2ffa-453c-97bb-ad7a05838635">

  分段：

    <img width="769" alt="image" src="https://github.com/user-attachments/assets/d8ad52b7-371a-47b5-98d5-c61e761aaf34">

  手动调整：

    ![image](https://github.com/user-attachments/assets/e3ce83a4-a0c5-4aae-a8fc-ea874a489a2e)
    
2. 创建工作流

 <img width="835" alt="image" src="https://github.com/user-attachments/assets/bf6d4b82-90fe-4d1a-8813-c209b0aa2a57">

 <img width="835" alt="image" src="https://github.com/user-attachments/assets/2597c354-d8b4-438b-863b-64ecdf5c9706">

3. 创建Bots

  ![image](https://github.com/user-attachments/assets/c7a4e078-144f-419f-b769-81fa4e5ff1bf)

  选择工作流模型，并添加工作流

4. 发布Bots

   <img width="605" alt="image" src="https://github.com/user-attachments/assets/fa51ade1-712a-4615-b105-e0ebbff7fce0">

