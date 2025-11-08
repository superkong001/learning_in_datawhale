参考： [隐语社区secretflow](https://www.secretflow.org.cn/community/bootcamp/2narwgw4ub8rabq/overview)

# 政策与合规

![9ea0c6f76d097dc67df27e57f5f2515b](https://github.com/user-attachments/assets/dd0e3566-4ba8-441a-bb44-c57ac5339b87)

## 隐私计算
<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/2123433c-94af-49b8-93ec-160eca489d31" />

<img width="647" height="305" alt="image" src="https://github.com/user-attachments/assets/7052631b-8deb-47e3-a42d-f858393e185a" />

- 一级防护等级：安全目标—————非参与方无法通过攻击获得非约定信息；
- 二级防护等级：安全目标—————参与方无法通过一般水平攻击获取非约定信息，并且有审计追溯能力；
- 三级防护等级：安全目标—————参与方无法通过已知攻击获取非约定信息；
- 四级防护等级：安全目标—————通过纵深防御思想构建完善的安全防御体系，能够抵御一定未知攻击；
- 五级防护等级：安全目标—————能够通过形式化验证等手段证明系统不存在除结果、输入元信息以外的信息泄露；

<img width="800" height="332" alt="image" src="https://github.com/user-attachments/assets/6cece3c3-bedb-48e4-9481-0e59a3f88077" />

<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/7cf03ab4-db7c-4c12-8d9e-4e82a58319c6" />

<img width="246" height="241" alt="image" src="https://github.com/user-attachments/assets/a4dc4561-e682-4e25-997f-cdf701bd1e59" />

# 可信数据空间

定义：可信数据空间是基于共识规则，联接多方主体，实现数据资源共享共用的一种数据流通利用基础设施，是数据要素价值共创的应用生态,是支撑构建全国一体化数据市场的重要载体。

<img width="496" height="306" alt="image" src="https://github.com/user-attachments/assets/158ac4ed-bdf7-4b4d-93f5-a104d4522024" />

<img width="557" height="223" alt="image" src="https://github.com/user-attachments/assets/8e90d6d2-f910-4f3f-a693-8a7fad9b2784" />

<img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/b7840c59-0686-4bde-aa26-ce667bb01d03" />

- 可信数据空间服务平台应具备身份管理、接入连接器管理、目录管理、数字合约管理、可信数据空间管理、数据使用控制、国际空间互通网关7个功能,其中身份管理、接入连接器管理、目录管理可复用区域/行业功能节点相关能力,并可在此基础上结合可信数据空间业务需求进行扩展。国际空间互通网关为可选功能,由服务平台按需建设。
- 接入连接器是用户接入可信数据空间服务平台、访问和使用可信数据空间资源的入口。接入连接器加入可信数据空间时应遵循NDI-TR-2025-05,并采取扩展模式的接入连接器。其中需扩展的功能包括:数据交付、数据资源管理、数据产品管理、数字合约管理、数据使用控制5项功能。

## 相关标准

- 《可信数据空间 技术架构》
- 共性技术类：
  - 国家数据局指导全国数据标准化技术委员会于2025年3月形成并发布《数据基础设施 参考架构（试行）》《数据基础设施 互联互通基本要求（试行）》《数据基础设施 用户身份管理和接入规范（试行）》《数据基础设施 标识管理规范（试行）》《数据基础设施 接入连接器技术要求（试行）》《数据基础设施 数据目录描述规范（试行）》等6项技术文件
  - 2025年8月，全国数据标准化技术委员会发布《数据基础设施 区域/行业功能节点技术要求》《数据基础设施 接入管理》《数据基础设施 安全能力通用要求》共3项技术文件
  - 《数据基础设施 数据凭证技术要求》《数据匿名化流通实施指南及评估指南》《数据基础设施 公共数据沙箱技术要求》《数据基础设施 隐私保护计算公共服务平台》等技术文件也在编制中
- 特性技术类：《可信数据空间 数字合约技术要求》、《可信数据空间 使用控制技术要求》
- 业务运营类：《数据基础设施 接入管理要求》，待编制《数据基础设施 运营管理要求》、《可信数据空间 业务实施指南》
- 安全保障类：《数据基础设施安全能力通用要求》，待编制《数据基础设施 运行日志管理技术要求》、《数据基础设施 密码应用技术要求》
- 能力评价类：《可信数据空间 技术能力评价规范》，待编制《可信数据空间 运营管理能力评价》、《可信数据空间 应用服务成效评价》、《可信数据空间 安全保障能力评价》
- 应用服务类：待编制《可信数据空间 应用服务分级要求》，《可信数据空间 应用服务指南》

## 密态计算

蚂蚁密算技术架构：

<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/3cc4f467-21b2-4ce0-a08b-142bc6f497c8" />

数据从“内循环”转变成“外循环”。“外循环”中，数据将脱离持有方的管控边界，进入新的信息系统，新的运维方有作恶可能。运维方是权利中枢，其作恶带来的威胁是颠覆性的。原本的安全措施，即便做些增强，也是无法抵御的。

<img width="814" height="338" alt="image" src="https://github.com/user-attachments/assets/62898812-89e1-4ed7-8a05-da09a5921aaf" />

技术理念变革：技术信任、管理平权、内生安全

<img width="870" height="335" alt="image" src="https://github.com/user-attachments/assets/ea9a98bc-819f-4b7d-84c6-1359be9e56a5" />

国家数据局《数据领域名词解释》密态计算定义：通过综合利用密码学、可信硬件和系统安全的隐私保护计算技术，其计算过程实现数据可用不可见，计算结果能够保持密态化，实现计算全链路保障，防止数据泄漏和滥用。

<img width="887" height="286" alt="image" src="https://github.com/user-attachments/assets/c9eb21c3-688b-4f1f-b293-63f5243df617" />

Tips：密态通用计算成本已降至明文分布式计算成本的1.5倍以下，而密态大模型推理成本更降至传统明文推理的1.2倍以内。

<img width="386" height="290" alt="image" src="https://github.com/user-attachments/assets/acdf109d-dd10-4aa9-80b7-102c4993b633" />

- 与MPC、FL的区别：MPC（安全多方计算）、FL（联邦学习）是基于分布式协议，多个参与方共同计算一个目标函数。在实际使用过程中，为了加速计算过程，往往会将部分中间结果以明文的形式交给某些参与方，而且一些协议的最终结果也是明文的，导致这些协议无法用于组成更大的密态计算过程。此外，分布式的网络交互也导致性能受到限制。
- 与TEE（Trusted Execution Environment，可信执行环境）的区别：TEE 只是CPU 的一个原子能力，仅提供了最基础的隔离认证能力，缺乏构建产业化应用的其他很多核心能力。这好比是，在金库、运钞车之上构建银行系统，在汽车、火车之上构建运输系统，还需要很多体系化构建能力。（蚂蚁密算采用安全编程语言Rust实现。）
- 与沙箱的区别：沙箱是防止恶意程序逃逸，即防止程序攻击主机，不保护程序。运维人员仍能窥探应用程序。
- 与区块链的区别：区块链的控制能力，是通过拒绝异常结果达到的，无法防止滥用。

<img width="836" height="281" alt="image" src="https://github.com/user-attachments/assets/fa4e8e28-2675-4df7-84f4-1855af7169b3" />

- 可信机制层：依托硬件可信根，构建远程度量、可信应用身份、可信密钥服务等，并通过相互验证实现信任的传递，最终用户可以对大型集群进行远程度量，并判断其安全性。同时，硬件机制会确保被验证过的集群不会被运维人员篡改。
  <img width="398" height="303" alt="image" src="https://github.com/user-attachments/assets/b8159403-c057-42e6-b43b-44eff1d8cde1" />

  - 可信根实现了一个非常重要的能力，远程度量与验证：

  <img width="629" height="333" alt="image" src="https://github.com/user-attachments/assets/46ea2384-fc84-4b18-8409-e31109833458" />

  - 可信应用身份：基于可信执行环境等，可以实现不受运维人员操纵的应用，叫做“可信应用”。
  - 可信集群、可信执行网络：多个可信应用相互协作，每个可信应用需要判断其他可信应用提供的服务是否能够满足其安全目标。
  - 可信密钥服务（TKMS）：采用“可信应用身份”作为用户账号，在可信体系下闭环。
  - 可信日志审计：日志产出时即进行保护，没有任何攻击间隙；产出的日志使用可信应用专属密钥保护，外界无法篡改。
  - 安全运维服务：当操作系统在TCB外时，运维操作也发生在TCB外，运维不影响安全；当操作系统在TCB内部时，需要对系统进行改造，限制运维人员的权限。

- 全流程密态保障：基于硬件隔离环境或密码协议，构建机密虚拟机、TEE OS、机密容器、密态胶囊等，结合密态互联实现全链路密态。
    
    <img width="472" height="349" alt="image" src="https://github.com/user-attachments/assets/25411965-54e8-4c6a-a4df-97092ffaa076" />

  - 可信执行环境基础原理：
    
    <img width="827" height="346" alt="image" src="https://github.com/user-attachments/assets/5965b670-75ba-42f2-9701-864c723a7c1b" />

  - 密态胶囊：数据与使用策略（数字合约）绑定胶囊；用户通过加密确保只有指定可信应用能接收到；用户通过检查代码，确认可信应用会严格检查策略。
    
    <img width="447" height="281" alt="image" src="https://github.com/user-attachments/assets/e75417f7-9985-4486-b05b-f5cfd5676b10" />

  - 可信鉴权:可信应用要从自己的安全目标出发，对所有对应关系进行验证。（1.验证数据与提供者ID之间的对应关系;2.验证策略的签发者是否是数据提供者;3.验证所加载的数据、算法与相应ID 是否对应）
    
    <img width="397" height="357" alt="image" src="https://github.com/user-attachments/assets/8d6ba580-8732-44fe-9235-77fdc7e0e61b" />

- 数据互通层:主要提供数据流通全生命周期功能，从参与方身份鉴别、数据目录查询，一直到数据授权、鉴权、析出管控。并通过受控匿名化解决个人信息保护问题，通过安全分级让不同平台的安全性显性化。
  
  <img width="532" height="198" alt="image" src="https://github.com/user-attachments/assets/30eecf3a-97a0-41cc-adce-812814af0fee" />

  - 参与方身份：权威方颁发、参与方自己保管；可信应用验证；验证不可绕过。
  - 受控匿名化：应对匿名化失败的原因（数据在开放空间中与其他来源的数据交叉比对），将数据通过技术手段限制在特定空间内 ，切断这种关联。无身份标识输入、无个体粒度非授权数据输出。
    
    <img width="403" height="262" alt="image" src="https://github.com/user-attachments/assets/ee9d299e-0b4c-4a81-9fc0-68cd84ed42f2" />

- 密态计算层：密态计算能够高效集成现有主流计算框架与应用工具，支持包括大模型在内的多种计算范式，使得开发者无需深入了解底层密码学或可信硬件的原理，也不需要改变原有的编程习惯，即可快速构建高安全等级的数据流通应用。
  - 大数据密态：将计算框架放入到TEE环境内运行。通过插件的方式，将大数据框架与密态Pass融合。
    
    <img width="614" height="263" alt="image" src="https://github.com/user-attachments/assets/86420e95-f0ad-46de-853d-688a12d19395" />

  - 密态数据库：数据库管理员不再拥有“特权”，由数据提供方决定谁能使用数据。
    
    <img width="612" height="226" alt="image" src="https://github.com/user-attachments/assets/5eca953e-b06d-4181-92f9-6afb5387b9b6" />



