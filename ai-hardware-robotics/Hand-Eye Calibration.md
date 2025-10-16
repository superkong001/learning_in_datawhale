## 参考

[1.手眼标定算法---Sai-Lenz(A New Technique for Fully Autonomous and Efficient 3D Robotics Hand/Eye Calibrati)](https://blog.csdn.net/u010368556/article/details/81585385)

[2.机器人手眼标定使用指南](https://docs.mech-mind.net/1.5/zh-CN/SoftwareSuite/MechVision/CalibrationGuide/CalibrationGuide.html)

[3.标定学习笔记（四）-- 手眼标定详解](https://blog.csdn.net/qq_45006390/article/details/121670412)

[4.Camera Calibration and 3D Reconstruction](https://docs.opencv.org/4.10.0/d9/d0c/group__calib3d.html#gad10a5ef12ee3499a0774c7904a801b99)

[5.3D视觉工坊-手眼标定（附opencv实现代码）](https://blog.csdn.net/z504727099/article/details/115494147)

[6.Hand-Eye Calibration](https://github.com/datawhalechina/ai-hardware-robotics/blob/main/02-%E6%9C%BA%E5%99%A8%E4%BA%BA%E5%9F%BA%E7%A1%80%E5%92%8C%E6%8E%A7%E5%88%B6%E3%80%81%E6%89%8B%E7%9C%BC%E5%8D%8F%E8%B0%83/Hand-Eye%20Calibration.md)

## 手眼标定的定义

手眼标定（Hand-Eye Calibration）是机器人视觉应用中的一个基础且关键的问题，主要用于统一视觉系统与机器人的坐标系，具体来说，就是确定摄像头与机器人的相对姿态关系。

> 当我们希望使用视觉引导机器人去抓取物体时，需要知道三个相对位置关系，即
> 
> 1. 末端执行器与机器人底座之间相对位置关系
> 2. 摄像头与末端执行器之间相对位置关系
> 3. 物体与摄像头之间的相对位置和方向
> 
> 手眼标定主要解决其中第二个问题，即确定“手”与安装在其上“眼”之间的空间变换关系，即求解相机坐标系和机器人坐标系之间的变换矩阵。  这里的机器人末端执行器称为手，摄像头称为眼。

根据摄像头安装方式的不同，手眼标定分为两种形式：
1. 摄像头安装在机械手末端，称之为眼在手上（Eye in hand）
2. 摄像头安装在机械臂外的机器人底座上，则称之为眼在手外（Eye to hand）

![1990c1350e8a97d48b571af7b48c04b0_6_image_0_raw=true](https://github.com/user-attachments/assets/1ae6a904-31c0-4207-84c3-91291fb19d98)

## 手眼标定的数学模型

不管是眼在手上还是眼在手外，它们求解的数学方程是一样的，手眼标定的核心方程都为 $AH = HB$ 。

它表示：当机械臂从一个姿态 A 变到另一个姿态 B 时，相机看到的标定板的变化也会发生对应的变化。也就是机械臂动多少，相机看到的物体也要动相应的量。手眼标定的目标，就是求出这中间固定的“变换关系” $𝐻 = T^e_c$ 

- 𝐴：机械臂末端执行器在两次拍摄间的位姿变化；
- 𝐵：标定板在相机视角下的变化；
- 𝐻：相机与机械臂末端之间的固定关系（这是要标定的目标）。

首先定义如下坐标系：

$F_b$: **基座坐标系**(Base Frame)，固定在机械臂的底座上，是机械臂运动的全局参考坐标系。

$F_e$: **末端执行器坐标系**(End-Effector Frame)，固定在机械臂末端，比如机械爪、工具尖端等。

$F_c$: **相机坐标系**（Camera Frame），固定在相机光心处，用于视觉感知的坐标系。

$F_t$: **标定目标坐标系**（Calibration Target Frame），固定在标定目标（如棋盘格、圆点板）上。

坐标系之间的关系通常用齐次变换矩阵（刚体变换）T表示：

$$
T^i_j =
\begin{bmatrix}
R^i_j & t^i_j \\
0 & 1
\end{bmatrix}
$$

其中：
- $R^i_j$ ：3×3 旋转矩阵，描述坐标系 j 相对于 i 的旋转；
- $t^i_j$ ：3×1 平移向量，描述坐标系 j 相对于 i 的平移；
- $𝑖,𝑗$ ：分别代表起点与终点坐标系。
- $R ∈ SO(3)$ 旋转矩阵(3x3)
- $t ∈ R^{3}$ 平移向量

T的上下标表示变换是对于哪两个坐标系，例如：
- $T^e_c$：表示将相机坐标系转换到末端执行器坐标系的变换，也表示相机在末端执行器坐标系下的位姿，在眼在手上这种情形下，就是要求的目标矩阵。

在任意时刻 𝑘 ，四个坐标系间的几何关系为： $T^b_t=T^b_eT^e_cT^c_t$ ，也就是说，标定板（目标）在基座坐标系下的位置，可以由以下三段变换组合得到。

| 模型                    | 相机位置                | 相机是否随机械臂一起动  | 举例场景            |
| --------------------- | ------------------- | ------------ | --------------- |
| **Eye-in-Hand（手眼合一）** | 相机安装在机械臂末端          | ✅ 相机会随着机械臂运动 | 工业抓取机器人、焊接机器人   |
| **Eye-to-Hand（手眼分离）** | 相机固定在外部环境中（如三脚架、墙壁） | ❌ 相机不动       | 生产线检测、固定相机监控抓取区 |

### Eye In Hand 推导核心方程（Derivation of the Core Equation）

当相机固定在机械臂末端时，相机与末端执行器之间的变换是固定的进行该类手眼标定时，会将标定板固定在一处，然后控制机械臂移动到不同位置，使用机械臂上固定的相机，在不同位置对标定板拍照，拍摄多组不同位置下标定板的照片。

![09ccdf1018bf828fa6cb4f57816131b9_6_image_1_raw=true](https://github.com/user-attachments/assets/c19e3d68-2856-4cf8-9bf1-f3eb051ae72c)

当机械臂从姿态 1 移动到姿态 2 时，分别记录两次的位姿：

$$
(T^b_{e1}, T^{c1}_t)
(T^b_{e2}, T^{c2}_t)
$$

由于标定板与机器人底座是固定的，二者之间相对位姿关系不变，由几何关系则有：

$$T^b_t = T^b_{e1}T^{e1}_{c1}T^{c1}_{t} = T^b_{e2}T^{e2}_{c2}T^{c2}_{t}$$

| 符号                | 含义                         | 说明                |
| ----------------- | -------------------------- | ----------------- |
|  $(T^b_{e1})$        | 末端执行器在“姿态1”时的位姿（从末端到基座的变换） | 把末端坐标系下的点变换到基座坐标系 |
|  $(T^b_{e2})$        | 末端执行器在“姿态2”时的位姿            | 同上                |
|  $((T^b_{e2})^{-1})$ | 从“基座”到“末端2”的逆变换            | 即末端2坐标系相对于基座的变换   |
|  $(T^{e}_c)$         | 相机相对于末端执行器的固定变换            | 是我们要标定的“手眼矩阵”     |

其中 $(T^b_{e2})^{-1} T^b_{e1} T^{e}_c$ 等价于：

先从相机坐标系出发（在姿态1时），
- → 转到末端1坐标系（通过 $T^{e}_c$ ），
- → 再从末端1转到基座坐标系（通过 $T^{e1}_b$ ），
- → 最后从基座转回末端2坐标系（通过 $(T^b_{e2})^{-1}$ ）。

由于 $T^e_c$ 是固定的（相机固定安装在末端），即 $T^{e1}_c = T^{e2}_c = T^e_c$ ，因此定义：

- 末端的相对运动：
  $A=T^{e2}_ {e1}=(T^b_ {e2})^{-1}T^b_ {e1}$
- 目标在相机下的相对运动：
  $B=T^{c2}_ {c1}=T^{c2}_ t (T^{c1}_ t)^{-1}$
- 常量刚体：相机刚性安装在末端
  $H=T^{e1}_c=T^{e2}_c=T^e_c$ 

对上述等式进行变换：

$$
\begin{align*}
T^b_{e1} T^{e}_ {c} T^{c1}_ t 
&= T^b_ {e2} T^{e}_ {c} T^{c2}_ t \\
(T^b_ {e2})^{-1}T^b_ {e1}T^{e1}_ {c1}T^{c1}_ {t} 
&=T^{e2}_ {c2}T^{c2}_ {t} \\
(T^b_ {e2})^{-1}T^b_ {e1}T^{e1}_ {c1} 
&=T^{e2}_ {c2}T^{c2}_ {t}(T^{c1}_ {t})^{-1} \\
T^{e2}_ {e1}T^{e1}_ {c1} 
&=T^{e2}_ {c2}T^{c2}_ {c1} \\
T^{e2}_ {e1}T^{e}_ {c} 
&=T^{e}_ {c}T^{c2}_ {c1} \\
\Rightarrow \quad A H &= H B
\end{align*}
$$

其中 $T^e_c$ 就是最终需要求解的 $H$ 。

### Eye To Hand

当相机固定在机械臂以外时，相机与末端执行器的相对位置会随着机械臂的运动而改变。进行该类手眼标定时，会将标定板固定在机械臂末端，然后控制机械臂拿着标定板，围绕着固定的相机拍照。为了求解的准确性，一般需要拍摄多于10组的照片。

![3e06961ddbc9c5469e4361deab822e8f_6_image_2_raw=true](https://github.com/user-attachments/assets/4ef59887-b71e-49f0-a9a1-31c053fa91b5)

由于此时标定板是固定在机械臂末端的，二者相对位置在拍摄不同照片时值不变，所以有：

$$T^e_t = T^{e1}_ bT^{b}_ {c}T^{c}_ {t} = T^{e2}_ bT^{b}_ {c}T^{c}_ {t}$$

定义：
- $A = T^b_ {e2}T^{e1}_ b$    （末端相对于基座的相对运动）
- $B = T^{c}_ {t2}T^{t1}_ c$    （相机视角下目标的相对运动）
- $X = T^b_ c$                        （相机与基座之间的固定关系）

$$ 
\begin{align*}
T^{e1}_ bT^{b}_ {c}T^{c}_ {t1} 
&= T^{e2}_ bT^{b}_ {c}T^{c}_ {t2} \\
(T^{e2}_ b)^{-1}T^{e1}_ bT^{b}_ {c}T^{c}_ {t1} 
&= T^{b}_ {c}T^{c}_ {t2} \\
(T^{e2}_ b)^{-1}T^{e1}_ bT^{b}_ {c} 
&= T^{b}_ {c}T^{c}_ {t2}(T^{c}_ {t1})^{-1} \\
(T^b_ {e2}T^{e1}_ b)T^{b}_ {c} 
&= T^{b}_ {c}(T^{c}_ {t2}T^{t1}_ c) \\
\Rightarrow \quad A X &= X B
\end{align*}
$$

## 求解 $AH = HB$ 

$AH = HB$ 求解方法，目前比较常用的是分步解法，即将方程组进行分解，然后利用旋转矩阵的性质，先求解出旋转，然后将旋转的解代入平移求解中，再求出平移部分。

常见的两步经典算法有将旋转矩阵转为旋转向量求解的Tsai-Lenz方法，基于旋转矩阵李群性质（李群的伴随性质）进行求解的Park方法等。

### Park方法求解旋转

1. 手眼标定的基本方程

手眼标定问题的几何关系可以表示为：

$$
A X = X B
$$

其中， $A, B, X \in SE(3)$ ，均为齐次变换矩阵；每个矩阵由旋转矩阵 𝑅 和平移向量 𝑡 构成。

$$
H =
\begin{bmatrix}
R & t \\
0 & 1
\end{bmatrix},
\quad R \in SO(3),\; t \in \mathbb{R}^3
$$

分别表示旋转和平移。将 $A, B, X$ 展开：

$$
A =
\begin{bmatrix}
R_A & t_A\\
0 & 1
\end{bmatrix},
\quad
B =
\begin{bmatrix}
R_B & t_B\\
0 & 1
\end{bmatrix},
\quad
X =
\begin{bmatrix}
R_X & t_X\\
0 & 1
\end{bmatrix}.
$$

原方程三个变量均为齐次变换矩阵（homogeneous transformation：将旋转和平移变换写在一个4x4的矩阵中），表示两个坐标系之间的变换。

2. 展开方程并分解旋转和平移部分

将上述形式代入 $A X = X B$ ：

$$
\begin{align*}
\begin{bmatrix}
R_A & t_A\\
0 & 1
\end{bmatrix}
\begin{bmatrix}
R_X & t_X\\
0 & 1
\end{bmatrix}
&=
\begin{bmatrix}
R_X & t_X\\
0 & 1
\end{bmatrix}
\begin{bmatrix}
R_B & t_B\\
0 & 1
\end{bmatrix} \\
\Rightarrow \quad
\begin{bmatrix}
R_A R_X & R_A t_X + t_A\\
0 & 1
\end{bmatrix}
&=
\begin{bmatrix}
R_X R_B & R_X t_B + t_X\\
0 & 1
\end{bmatrix}
\end{align*}
$$

由此得到两组独立方程：

$$
\begin{align*}
R_A R_X &= R_X R_B \quad &(\text{旋转部分}) \\
R_A t_X + t_A &= R_X t_B + t_X \quad &(\text{平移部分})
\end{align*}
$$

$R_A R_X = R_X R_B$ 表示机械臂先转 $𝑅_𝐴$ ，再乘以 $𝑅_𝑋$ ，等价于先应用相机的固定关系 $𝑅_𝑋$ ，再让标定板转 $𝑅_𝐵$ ，也就是 $R_X$ 是那个把「相机看到的旋转轴」对齐到「机械臂旋转轴」的旋转。

3. 旋转部分的共轭关系

对旋转矩阵部分进行整理：

$$
\begin{align*}
R_A R_X &= R_X R_B \\
\Rightarrow \quad R_A &= R_X R_B R_X^{-1}
\end{align*}
$$

说明 $R_A$ 与 $R_B$ 为相似变换（共轭关系），即它们共享同一旋转轴，但角度不同。

4. 李群与李代数表示

旋转矩阵属于李群 $SO(3)$ ————3D旋转群，是一个特殊正交三维（三维空间中所有正交、行列式为 1 的矩阵，如：旋转矩阵有3个自由度），其李代数 $so(3)$ 是一个三维欧式空间，
旋转矩阵与旋转向量的关系为：

tips:
- 李群的本质：连续可微的变换群，是连续的变换集合。李群的特征是：群运算（矩阵乘法）；连续、可微；可以用少数参数描述（比如 SO(3) 只有 3 个自由度）。

| 名称    | 记号    | 含义           |
| ----- | ----- | ------------ |
| SO(3) | 3D旋转群 | 所有旋转矩阵的集合    |
| SE(3) | 刚体变换群 | 所有旋转+平移矩阵的集合 |
| SO(2) | 2D旋转群 | 平面旋转集合       |
| GL(n) | 一般线性群 | 所有可逆矩阵       |

- 李代数：是李群的‘局部线性化’（微分近似）。从“单位点”出发，小幅移动（小角度旋转、小距离平移）时，变化的方向

| 方向       | 数学形式                      | 含义            |
| -------- | ------------------------- | ------------- |
| 李代数 → 李群 | $R = \exp(\phi^\wedge)$ | 从角速度向量生成旋转矩阵  |
| 李群 → 李代数 | $\phi^\wedge = \log(R)$ | 从旋转矩阵提取旋转轴和角度 |

如果绕 z 轴旋转角度 θ：

$$
\phi =
\begin{bmatrix}
0\\
0\\
\theta
\end{bmatrix},
\quad
\phi^\wedge =
\begin{bmatrix}
0 & -\theta & 0\\
\theta & 0 & 0\\
0 & 0 & 0
\end{bmatrix}.
$$

指数映射：

$$
R = \exp(\phi^\wedge) = 
\begin{bmatrix}
\cos\theta & -\sin\theta & 0\\
\sin\theta & \cos\theta & 0\\
0 & 0 & 1
\end{bmatrix}.
$$

$$
R = \exp(\phi^\wedge)
$$

其中 $\phi^\wedge$ 表示反对称矩阵：

$$
\phi^\wedge =
\begin{bmatrix}
0 & -\phi_3 & \phi_2\\
\phi_3 & 0 & -\phi_1\\
-\phi_2 & \phi_1 & 0
\end{bmatrix}.
$$

因此，旋转矩阵与旋转向量通过指数映射与对数映射互相转换。

5. 变换到李代数空间

将旋转方程 $R_A R_X = R_X R_B$ 将它写成指数形式：

$$
\exp(\alpha^\wedge)\exp(x^\wedge)
= \exp(x^\wedge)\exp(\beta^\wedge)
$$

根据李群的共轭性质：

$$
\exp(\alpha^\wedge)
= \exp(x^\wedge)\exp(\beta^\wedge)\exp(-x^\wedge)
= \exp((R_X \beta)^\wedge)
$$

因此可得：

$$
\boxed{\alpha = R_X \beta}
$$

即在旋转向量空间中，
$A$ 的李代数向量等于 $R_X$ 作用于 $B$ 的李代数向量。

旋转矩阵为SO3群，SO3群为李群，每一个李群都有对应的李代数，其李代数处于低维的欧式空间（线性空间），是李群局部开域的切空间表示，李群与李代数可以通过指数映射与对数映射相互转换：

![4e3bb1be12b047a43c05806404948432_6_image_3_raw=true](https://github.com/user-attachments/assets/fcdc720c-b9ea-4717-a4b8-7a63b1ff386a)

对于旋转矩阵R，与对应的李代数**Φ **变换关系可以如下表示：

$R = \exp(Φ^{\wedge}) = \exp [Φ]$

其中[]符号表示^操作，及转为反对称矩阵，或者说叉积。

对于SO(3)，其伴随性质为：

![3838c668f5e8bc2d24ce96ecd1045fda_6_image_3_1_raw=true](https://github.com/user-attachments/assets/6b51b4bc-e957-40aa-9f61-de6d9ee117e2)

$$
\begin{aligned}
\theta_{A}  & =\theta_{X} \theta_{B} \theta_{X}^T \\
\exp [\alpha] & = \theta_{X}\exp [\beta]\theta_{X}^T  \\
\exp [\alpha] & = \exp [\theta_{X}\beta]  \\
\alpha &= \theta_{X}\beta
\end{aligned}
$$

6. 多组姿态下的最小二乘估计

对于多组观测 $(A_i, B_i)$ ，有：

$$
\alpha_i = R_X \beta_i
$$

当存在多组观测时，上述问题可以转化为如下最小二乘问题，目标是最小化误差：

$$
\min_{R_X} \sum_{i=1}^k \| R_X \beta_i - \alpha_i \|^2
$$

这是一个三维点云刚体对齐问题。

定义矩阵：

$$
M = \sum_{i=1}^k \beta_i \alpha_i^T
$$

其最优解为：

$$
R_X = U V^T,
\quad \text{其中 } M = U \Sigma V^T
$$

等价写法：

$$
R_X = (M^T M)^{-\frac{1}{2}} M^T
$$

当存在多组观测时，上述问题可以转化为如下最小二乘问题：

![b840125b35f6bc12306073bd6093816a_6_image_3_2_raw=true](https://github.com/user-attachments/assets/93d853bf-287f-4399-b718-cb9dfb5bd2d8)

α与β为对应旋转的李代数，它们都是三维向量，可以看作一个三维点，那么上述问题等同于一个点云配准问题：  

![13ad8648ce9f2aadffc13c6012c95a09_6_image_3_4_1_raw=true](https://github.com/user-attachments/assets/03b0ba18-c12d-4955-aa09-9de4d2bce250)

该问题有最小二乘解为：  

![b88b4acfd1f4c64dba3e125dae837974_6_image_3_3_raw=true](https://github.com/user-attachments/assets/6f23ac30-f952-417f-8152-0854bd094b5b)

其中：  

![84f507ddb5f10f0c5d99ad7f28cf8054_6_image_3_4_raw=true](https://github.com/user-attachments/assets/8d22a27d-2729-496f-94cb-a67300dccd2d)

7. 平移部分的解

回到平移方程：

$$
R_A t_X + t_A = R_X t_B + t_X
$$

化简为：

$$
(R_A - I)t_X = R_X t_B - t_A
$$

当存在多组姿态时，利用最小二乘法求解：

$$
t_X =
\left(\sum_i (R_{Ai}-I)^T(R_{Ai}-I)\right)^{-1}
\left(\sum_i (R_{Ai}-I)^T(R_X t_{Bi}-t_{Ai})\right)
$$

8. 最终结果

$$
X =
\begin{bmatrix}
R_X & t_X\\
0 & 1
\end{bmatrix}
$$

原方程进行变换：

$$
AH  = HB \\
\begin{array}{l}
\left[\begin{array}{cc}
\theta_{A} & b_{A} \\
0 & 1
\end{array}\right]\left[\begin{array}{cc}
\theta_{X} & b_{X} \\
0 & 1
\end{array}\right]  =\left[\begin{array}{cc}
\theta_{X} & b_{X} \\
0 & 1
\end{array}\right]\left[\begin{array}{cc}
\theta_{B} & b_{B} \\
0 & 1
\end{array}\right]\\
\end{array}
$$

所以有（乘积结果旋转与平移部分对应位置相等）：

$$
\begin{aligned}
\theta_{A} \theta_{X} & =\theta_{X} \theta_{B} \\
\theta_{A} b_{X}+b_{A} & =\theta_{X} b_{B}+b_{X}
\end{aligned}
$$

首先求解第一个只包含旋转矩阵的方程。

$$
\begin{aligned}
\theta_{A} \theta_{X}  =\theta_{X} \theta_{B} \\
\theta_{A}  =\theta_{X} \theta_{B} \theta_{X}^T 
\end{aligned}
$$

### Park方法求解平移

在求解得到旋转矩阵后，将旋转矩阵值代入第二个方程：

$$
\begin{aligned}
\theta_{A} b_{X}+b_{A} & =\theta_{X} b_{B}+b_{X} \\
\theta_{A} b_{X} - b_{X} & =\theta_{X} b_{B}- b_{A} \\
(\theta_{A} - I)b_{X}  & =\theta_{X} b_{B}- b_{A} \\
Cb_{X} &= D
\end{aligned}
$$

其中C与D均为已知值，由于C不一定可逆，原方程做如下变换：  

$$
\begin{aligned}
Cb_{X} &= D \\
C^TCb_{X} &= C^TD \\
b_{X} &= (C^TCX)^{-1}C^TD
\end{aligned}
$$

即可求得平移部分。

当有多组观测值时  

<img width="616" height="68" alt="f2f1e0b9be6bf7d205df0c915c6aa785_6_image_3_5_raw=true" src="https://github.com/user-attachments/assets/1cac07e0-9d34-4c02-b13c-e5eeca849c3b" />

最终的解为： 

$$
\begin{aligned}
H = \left[\begin{array}{cc}
\theta_{X} & b_{X} \\
0 & 1
\end{array}\right]
\end{aligned}
$$

## OpenCV中的手眼标定方法及标定注意事项

手眼标定算法按照实现原理可以将所有方法分为三类：独立闭式解，同时闭式解，迭代方法 

> a.独立闭式解 **（seperable closed-form solutions）**: 与位移分量分开求解旋转分量。
> 
> 缺点: 旋转分量Rx的计算误差会被带入位移分量tx的计算中。
> 
> b.同时闭式解 **（simultaneous closed-form solutions）**： 同时求解位移分量和旋转分量
> 
> 缺点: 由于噪声的影响，旋转分量Rx的求解可能不一定是正交矩阵。因此，必须对旋转分量采取正交化步骤。然而，相应的位移分量没有被重新计算，这会导致求解错误。
> 
> c.迭代方法 **（iterative solutions）**： 使用优化技术迭代求解旋转分量和平移分量。
> 
> 缺点: 这种方式计算量可能很大，因为这些方法通常包含复杂的优化程序。此外，随着方程数量(n)变大，迭代解与封闭式解之间的差异通常会变小。因此，使用此方法前必须决定迭代解决方案的准确性是否值得计算成本。

### OpenCV中的手眼标定接口

![13d5bcd32e242de6e47a911a51c483cb_6_image_4_raw=true](https://github.com/user-attachments/assets/7911e478-417e-45d6-a286-af82ed122af7)

OpenCV主要实现了前两类方法，其中默认方法为TSAI。PARK，HORAUD也是独立解方法。ANDREFF，DANIILIDIS是同时闭式解，（从同一组数据实验中得到结论，认为独立解中TSAI方法求得解的误差较大）。

采集多组标定数据，传入相应的机械臂数据与摄像头标定板定位数据，就可以获得标定结果。

### 标定注意事项

当新手进行第一次标定时，需要注意如下事项：

一：用calibrateCamera求标定板到相机的R,t，跟抓取用的内参不同，造成误差

二：标定板角点方向反了。默认从左到右，有时会出现从右到左，导致不在统一坐标系。

三：标定板面积过小，而且只在中心移动。会导致边缘不准。

四：标定时要旋转

五：图片太少。要10张以上

六。某些相机有问题，rgbd的变换矩阵要注意

## 评价手眼标定效果

通过测量几组不同位置的目标点，我们可以比较：

测量值：

$$
P_{measure}
$$

和预测值：

$$
P_{predict} = T^{robot}_{camera} \, P_{camera}
$$

之间的距离误差：

$$
err = P_{measure} - T^{robot}_{camera} \, P_{camera}
$$

通过计算各个点的误差的**均值**和**标准差**，最终我们可以计算出标定结果的系统误差（均值）和随机误差（标准差）。

