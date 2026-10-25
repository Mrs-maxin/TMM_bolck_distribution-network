# 基于TMM的配电网交流概率潮流计算方法（理论方法部分）

日期：2026-10-24。本文件按论文正文格式撰写理论方法，第1—5节为正文，附录A—C为正文附录，第6节为内部的实现与证据对应，不进入论文。正文涉及三种电压联合分布模型：作为比较基础的GMM（AQGMM）、标准TMM，以及针对配电网的改进TMM（分组t混合）。

---

## 1 交流概率潮流计算框架

### 1.1 源荷场景与节点电压

考虑$n$节点辐射状配电网，节点1为平衡节点，其余为PQ节点，拓扑和平衡节点电压固定。非平衡节点的净注入功率（发电减负荷，p.u.）记为

$$
\mathbf p=\big[\mathbf P_{\mathcal N}^{\mathsf T},\ \mathbf Q_{\mathcal N}^{\mathsf T}\big]^{\mathsf T}\in\mathbb R^{d},\qquad d=2(n-1).
\tag{1}
$$

$\mathbf p$的概率分布由源荷功率规律给定，例如光伏、风电的预测出力加预测误差，以及负荷的波动。节点电压采用直角坐标

$$
\mathbf x=\big[\Re(\mathbf V_{\mathcal N})^{\mathsf T},\ \Im(\mathbf V_{\mathcal N})^{\mathsf T}\big]^{\mathsf T}\in\mathbb R^{d},
\tag{2}
$$

交流潮流方程$\mathbf p=\mathbf F(\mathbf x)$关于$\mathbf x$为精确的二次关系。

由源荷场景计算节点电压沿用AQGMM的快速映射。取离场景平均注入最近的场景为锚点$a$，求一次交流潮流得到$\mathbf x_a$、$\mathbf p_a$并分解雅可比矩阵$\mathbf J_a$。对任一场景$\mathbf p$，先作二次残差修正

$$
\mathbf h_1=\mathbf J_a^{-1}(\mathbf p-\mathbf p_a),\quad
\mathbf h_2=-\mathbf J_a^{-1}\big[\mathbf F(\mathbf x_a+\mathbf h_1)-\mathbf p\big],\quad
\hat{\mathbf x}=\mathbf x_a+\mathbf h_1+\mathbf h_2+\mathbf J_a^{-1}\big[\mathbf p-\mathbf F(\mathbf x_a+\mathbf h_1+\mathbf h_2)\big],
\tag{3}
$$

再以$\hat{\mathbf x}$为初值作固定雅可比修正

$$
\mathbf x^{(m+1)}=\mathbf x^{(m)}+\alpha_m\mathbf J_a^{-1}\big[\mathbf p-\mathbf F(\mathbf x^{(m)})\big],
\tag{4}
$$

直到各节点功率残差不超过$10^{-8}$ p.u.。未达到要求的场景按残差最大者追加锚点，锚点数上限12；全部锚点均未达到要求的场景才逐场景求解交流潮流。式(3)、(4)只需锚点处的一次交流潮流和一次雅可比分解，其余为对全部场景的批量矩阵运算。由此得到的节点电压用于估计电压联合分布的参数。

### 1.2 电气量的电压二次型

节点电压幅值平方、支路两端有功和无功功率以及系统有功网损，均可写为$\mathbf x$的二次型[14]

$$
q(\mathbf x)=\mathbf x^{\mathsf T}\mathbf A_q\mathbf x+\mathbf b_q^{\mathsf T}\mathbf x+c_q,\qquad \mathbf A_q=\mathbf A_q^{\mathsf T},
\tag{5}
$$

系数由支路导纳和平衡节点电压确定。支路两端功率均以流入支路为正，网损为全部在运支路两端有功功率之和。电压幅值的分布由幅值平方的分布换算。式(5)保留交流功率中电压的乘积项和平方项，节点电压同时出现较大偏差时，支路功率和网损的尾部由这些项决定。

### 1.3 由电压联合分布计算电气量分布

设节点电压的联合分布为$K$个分量的混合

$$
f(\mathbf x)=\sum_{k=1}^{K}\pi_k f_k(\mathbf x),\qquad \sum_k\pi_k=1 .
\tag{6}
$$

若每个分量在给定一个标量随机变量（下文的波动强度）后为高斯分布，则式(5)的条件特征函数为闭式（附录A），对该标量作一维积分并按分量加权，即得电气量的特征函数，再经反演得到概率分布函数和概率密度。因此，电气量尾部的计算精度由电压联合分布的尾部决定，本文的核心是电压联合分布模型。第2—4节依次给出GMM、标准TMM和分组t混合三种模型，三者共用第1.1节的快速映射和式(5)的二次型。

参数估计在按各方向标准差归一化的电压坐标中进行：

$$
\boldsymbol\xi=\mathbf W(\mathbf x-\bar{\mathbf x}),\qquad \mathbf W=\mathbf D^{-1}\mathbf E^{\mathsf T},
\tag{7}
$$

$\bar{\mathbf x}$为用于估计参数的源荷场景所对应电压的均值，$\mathbf E$和$\mathbf D$分别为其协方差的特征向量矩阵和沿各特征方向标准差构成的对角阵。配电网中光伏驱动方向与负荷小波动方向的电压标准差可相差5至6个数量级，式(7)使各方向在参数估计中处于同一数值尺度。

## 2 基于GMM的电压联合分布

AQGMM取式(6)中$f_k=\mathcal N(\boldsymbol\mu_k,\mathbf C_k)$，参数由期望最大化估计，分量$k$对场景$i$的归属概率记为$\gamma_{ik}$，分量数$K$按另一组源荷场景上的电压似然选择。给定分量后电压为高斯分布，式(5)的特征函数直接为附录A的闭式。GMM在分量内只能表示高斯尾部，电压分布中心集中而尾部延伸较远时，需要增加分量来表示尾部。

## 3 标准TMM

### 3.1 模型

标准TMM[15]的第$k$个分量为多元t分布，写成给定波动强度的条件形式：

$$
\boldsymbol\xi\mid C=k,\,S=s\ \sim\ \mathcal N(\boldsymbol\mu_k,\ s\,\mathbf C_k),\qquad
S=\frac{\nu_k-2}{\chi^2_{\nu_k}},\quad \mathrm E[S]=1 .
\tag{8}
$$

$C$为分量归属，$\nu_k>4$为自由度。$S$为该分量内同一场景的波动强度，同一个$S$同时放大全部$d$个电压方向的偏差；按式(8)归一后$\mathbf C_k$即为分量协方差。$\nu_k$越小，$S$的变化越大，电压分布中心越集中而外侧概率越大；$\nu_k\to\infty$时$S\equiv1$，退化为GMM的高斯分量。

### 3.2 参数估计

参数按矩确定，分量归属取同一$K$的AQGMM。记$N_k=\sum_i\gamma_{ik}$，

$$
\pi_k=\frac{N_k}{N},\quad
\boldsymbol\mu_k=\frac{1}{N_k}\sum_i\gamma_{ik}\boldsymbol\xi_i,\quad
\mathbf C_k=\frac{1}{N_k}\sum_i\gamma_{ik}(\boldsymbol\xi_i-\boldsymbol\mu_k)(\boldsymbol\xi_i-\boldsymbol\mu_k)^{\mathsf T}.
\tag{9}
$$

自由度由分量内的多元峰度[17]确定。记$\delta_{ik}=(\boldsymbol\xi_i-\boldsymbol\mu_k)^{\mathsf T}\mathbf C_k^{-1}(\boldsymbol\xi_i-\boldsymbol\mu_k)$，

$$
b_k=\frac{\sum_i\gamma_{ik}\delta_{ik}^2}{N_k\,d(d+2)},\qquad
\nu_k=\frac{4b_k-2}{b_k-1}\ \ (b_k>1),
\tag{10}
$$

$b_k\le1$时该分量取高斯分布，$\nu_k$下限取4.1以保证电气量方差存在。式(10)来自$\mathrm E[\delta^2]=d(d+2)\,\mathrm E[S^2]$及$\mathrm E[S^2]=(\nu-2)/(\nu-4)$。式(9)使各分量的均值和协方差与AQGMM相同，由式(5)，电气量的均值也与AQGMM相同，两者的差别只在电压分布的形状和尾部。最大似然估计[15]会降低偏差较大场景的权重，使光伏驱动方向的电压方差被低估，按矩确定可避免这一偏差。

### 3.3 电气量分布

给定分量$k$和强度$s$，$\boldsymbol\xi$为协方差$s\mathbf C_k$的高斯分布，式(5)的条件特征函数$\psi_q(t\mid k,s)$由附录A给出。对$S$积分得

$$
\psi_q(t)=\sum_k\pi_k\int_0^\infty\psi_q(t\mid k,s)\,g_{\nu_k}(s)\,\mathrm ds,
\tag{11}
$$

$g_{\nu_k}$为式(8)中$S$的概率密度，积分采用广义Gauss–Laguerre求积。

### 3.4 配电网中的局限

配电网中受天气共同放大的通常只是部分注入，例如同一云区的光伏，其余负荷为相互独立的小幅波动。设式(7)坐标中$r$个方向受共同强度放大，分量内多元峰度为$b_u>1$，其余$d-r$个方向近似为高斯分布且与前者独立。由$\delta=\delta_u+\delta_w$，$\mathrm E[\delta_u]=r$，$\mathrm E[\delta_w]=d-r$，$\mathrm E[\delta_w^2]=(d-r)(d-r+2)$，得

$$
b-1=\frac{r(r+2)}{d(d+2)}\,(b_u-1).
\tag{12}
$$

全网电压的峰度超出量被压缩为天气驱动方向的$r(r+2)/[d(d+2)]$。IEEE 33节点中$d=64$，8个光伏站对应$r=8$，压缩系数为0.019；$b_u=2$时$b=1.019$，由式(10)得$\nu\approx109$，标准TMM退化为接近GMM。另一方面，式(8)中同一个$S$若真正放大全部方向，也会把负荷小波动方向错误地放大。两方面说明，在只有部分注入受天气放大时，波动强度应只作用于天气驱动的电压方向，这是第4节改进的出发点。

## 4 分组t混合

### 4.1 物理依据

天气过程同时改变一个区域内多个光伏站（或风电场、受气温影响的负荷）的预测误差幅度：云层较厚时，同一区域各站出力偏差同时变大。该区域的注入变化经网络灵敏度改变节点电压，主要落在由潮流雅可比逆矩阵对应列确定的少数电压方向上；偏差较大时，潮流的二次关系还使一部分变化落在这些方向之外。相距较远的区域受各自云团影响，偏差幅度不完全相同，但受同一天气形势驱动而同时增减。光伏出力不低于零且不超过出力上限，偏差幅度有上界。预测出力接近晴空出力时，增发受晴空上限限制，云遮则可使出力骤降，偏差偏向减发一侧：云层越厚，偏差越大，电压也越整体偏向减发一侧。潮流二次响应是一阶电压偏差的平方效应，方向固定、幅度随强度增大，同样偏向一侧。

据此，改进TMM在每个分量内设若干组电压方向，每组一个波动强度：同一组的方向由同一强度放大并带动均值向同一侧移动，不同组的强度由同一天气形势同步决定，强度受出力限值截断，其余方向保持高斯。统计文献中，各组强度由同一均匀随机数决定的这类分布称为分组t分布或分组正态方差混合[18]—[20]。

### 4.2 天气驱动电压方向的确定

分组只用源荷注入场景、第1.1节已算得的节点电压和平均注入处的一次潮流，不使用MC方法结果，也不按电气量误差调整。步骤如下。

**步骤1：判别受同一天气过程放大的注入。** 记注入$i$在各场景中的偏差为$e_i$（减去其均值），对任意两个注入计算共同放大量

$$
c_{ij}=\mathrm{corr}\big(e_i^2,e_j^2\big)-\mathrm{corr}\big(e_i,e_j\big)^2 .
\tag{13}
$$

两注入偏差为联合高斯分布时$c_{ij}=0$；被同一强度放大时，两者的大偏差同时出现，$c_{ij}>0$。判别阈值$c^*$取与实际注入协方差相同、场景数相同的高斯场景中$c_{ij}$的最大值（独立生成5组取最大）。$c_{ij}>c^*$的注入相连，相连的注入构成一组$\mathcal W_g$，每组至少两个注入，一组对应一个天气过程。判别使用全部已生成的注入场景，不需要潮流计算。

**步骤2：计算一阶电压方向。** 在平均注入处求一次交流潮流，得到雅可比矩阵$\mathbf J_0$。第$g$组注入引起的电压变化方向为

$$
\mathbf H_g=\mathbf J_0^{-1}\mathbf L_g,
\tag{14}
$$

$\mathbf L_g$为选出$\mathcal W_g$各注入的0-1矩阵，$\mathbf H_g$即潮流雅可比逆矩阵中这些注入对应的列。将$\mathbf W\mathbf H_g$依次对已得方向正交化并作奇异值分解，保留奇异值大于最大奇异值$10^{-10}$倍的左奇异向量，得到第$g$组的一阶方向$\mathbf Q_g$，维数$r_g\le|\mathcal W_g|$。方向必须在实际运行点计算；用额定电压下的节点阻抗矩阵代替$\mathbf J_0^{-1}$，原始坐标中百分之几的方向偏差会在式(7)坐标中被放大到主导地位。

**步骤3：检出潮流二次响应方向。** 记一阶方向之外的电压坐标为$\mathbf w$，计算四阶矩阵

$$
\mathbf M=\frac1N\sum_i\|\mathbf w_i\|^2\,\mathbf w_i\mathbf w_i^{\mathsf T}
\tag{15}
$$

的特征向量[21]。某一特征方向同时满足以下三个条件时，判为潮流二次响应方向：（a）其特征值超过同协方差、同场景数高斯场景的最大特征值；（b）该方向承担的电压方差不低于全部电压方差的$10^{-3}$；（c）该方向坐标的平方与一阶方向坐标平方和的相关系数超过高斯情形的最大值。条件（c）排除出力限值等与天气强度无关的非高斯变化。每个二次响应方向归入其坐标平方与之相关程度最高的一阶组。

**步骤4：确定强度上限。** 第$g$组的强度上限取组内各注入最大可能偏差与偏差标准差之比的平方的中位数，

$$
s_{\max,g}=\mathrm{median}_{i\in\mathcal W_g}\left(\frac{\max|e_i|}{\sigma_{e_i}}\right)^2,
\tag{16}
$$

最大可能偏差由出力限值和场景平均注入确定，二次响应方向沿用其所属一阶组的上限。

分组结果为$G$组方向，每个天气过程一组，包含其一阶方向和归入的二次响应方向。二次响应方向与一阶方向共用同一强度，其偏向同一侧的特点由第4.3节的均值移动项表示。各组方向合并为$\mathbf Q_u=[\mathbf Q_1,\ldots,\mathbf Q_G]$，维数$r=\sum_g r_g$，$\mathbf Q_w$为其正交补。

表1汇总分组所需的输入、固定阈值、输出和计算量。

**表1 天气驱动电压方向的确定**

| 项目 | 内容 |
|---|---|
| 输入 | 源荷注入场景；快速映射得到的节点电压；平均注入处的一次交流潮流及其雅可比矩阵 |
| 固定阈值 | 高斯阈值的生成组数5；奇异值相对阈值$10^{-10}$；电压方差占比$10^{-3}$；每组至少2个注入 |
| 输出 | 各组注入$\mathcal W_g$；各组电压方向$\mathbf Q_g$；强度上限$s_{\max,g}$ |
| 计算量 | 式(13)：$O(Nm^2)$，$m$为注入数；式(14)：取已有逆矩阵的列；正交化与奇异值分解：$O(dr^2)$；式(15)：$O(N(d-r)^2)$与一次$(d-r)$阶特征分解 |
| 不需要 | 额外的逐场景潮流计算；按MC方法结果或电气量误差调整的参数 |

分组的两种极端情形与已有模型一致：步骤1不形成任何组时，波动强度不作用于任何方向，模型退化为GMM；全部注入同属一组且$r=d$时，模型退化为标准TMM。

### 4.3 模型

记$\mathbf u=\mathbf Q_u^{\mathsf T}\boldsymbol\xi\in\mathbb R^{r}$，$\mathbf w=\mathbf Q_w^{\mathsf T}\boldsymbol\xi\in\mathbb R^{d-r}$，$g(j)$为第$j$个方向所属的组。第$k$个分量写为

$$
\mathbf u=\mathbf m_k+\boldsymbol\gamma_k\odot\big(\mathbf s(U)-\mathbf 1\big)+\mathbf D(U)\,\mathbf z,\qquad \mathbf z\sim\mathcal N(\mathbf 0,\tilde{\boldsymbol\Sigma}_k),\qquad
\mathbf D(U)=\mathrm{diag}\Big(\sqrt{S_{g(1)}(U)},\ldots,\sqrt{S_{g(r)}(U)}\Big),
\tag{17}
$$

$\mathbf s(U)$的第$j$个元素为$S_{g(j)}(U)$，$\boldsymbol\gamma_k$为各方向的均值移动量，$\odot$为逐元素相乘。

$$
S_g(U)=F_{g,k}^{-1}(U),\qquad U\sim\mathrm{Uniform}(0,1),
\tag{18}
$$

$$
\mathbf w\mid\mathbf u,C=k\ \sim\ \mathcal N\big(\mathbf a_k+\mathbf G_k\mathbf u,\ \boldsymbol\Omega_k\big).
\tag{19}
$$

$U$表示同一场景中天气形势的强弱，同一个$U$决定各组强度，各组强度同时增减，大小可以不同。$F_{g,k}$为截断并归一的逆伽马分布：$V=\nu_{g,k}/\chi^2_{\nu_{g,k}}$限制在$V\le v_{\max}$，$S_g=V/\mathrm E[V]$，$v_{\max}$使$S_g$的上限等于式(16)的$s_{\max,g}$（附录B）。强度大于1时，偏差放大，电压沿$\boldsymbol\gamma_k$向光伏减发一侧移动；强度小于1时，偏差缩小，电压略向另一侧回移。$\mathrm E[S_g]=1$，分量电压均值仍为$\mathbf m_k$。式(19)中，与天气驱动方向相关的电压变化通过$\mathbf G_k\mathbf u$随强度放大，负荷独立小波动引起的其余变化保持高斯。

$\boldsymbol\gamma_k=\mathbf 0$、只有一组且不截断时，式(17)—(19)为一个强度只作用于天气驱动方向的TMM；$\boldsymbol\gamma_k=\mathbf 0$且各组强度不截断时，$\mathbf u$为分组t分布[18]—[20]；强度同时决定均值与方差的形式在统计文献中称为正态均值方差混合【文献待核查】。$(\mathbf u,\mathbf w)$整体不是多元t分布。

### 4.4 参数估计

分量归属取同一$K$的AQGMM的$\gamma_{ik}$，各参数按矩确定：

1. **位置与协方差。** $\pi_k$按式(9)；$\mathbf m_k$和$\mathbf u$的分量协方差$\mathbf C_{u,k}$为$\gamma_{ik}$加权的均值和协方差。
2. **各组自由度。** 对第$g$组，$\delta_{ik,g}=(\mathbf u_{i,g}-\mathbf m_{k,g})^{\mathsf T}\mathbf C_{u,k,gg}^{-1}(\mathbf u_{i,g}-\mathbf m_{k,g})$，
$$
b_{g,k}=\frac{\sum_i\gamma_{ik}\delta_{ik,g}^2}{N_k\,r_g(r_g+2)},
\tag{20}
$$
$\nu_{g,k}$由截断强度满足$\mathrm E[S_g^2]=b_{g,k}$的一维求根确定（附录B）；$b_{g,k}\le1.05$时该组取高斯。
3. **均值移动量。** 记$v_g=\mathrm{Var}(S_g)$、$\kappa_g=\mathrm E[(S_g-1)^3]$，方向$j$的分量方差$C_{u,k,jj}$和三阶中心矩$\mu_{3,j}$为$\gamma_{ik}$加权统计值。由式(17)，
$$
\mu_{3,j}=\gamma_{k,j}^3\big(\kappa_{g(j)}-3v_{g(j)}^2\big)+3\gamma_{k,j}\,v_{g(j)}\,C_{u,k,jj},
\tag{21a}
$$
取绝对值最小、且使高斯部分方差$C_{u,k,jj}-\gamma_{k,j}^2v_{g(j)}$不低于$C_{u,k,jj}$的10%的实根；无此根时该方向不设均值移动。该组取高斯时$\gamma_{k,j}=0$。
4. **散布矩阵。** 为使$\mathbf u$的协方差等于$\mathbf C_{u,k}$，
$$
\tilde\Sigma_{k,jl}=\frac{C_{u,k,jl}-\gamma_{k,j}\gamma_{k,l}\,\mathrm{Cov}\big(S_{g(j)},S_{g(l)}\big)}{\mathrm E\big[\sqrt{S_{g(j)}S_{g(l)}}\big]},\qquad
\mathrm E\big[\sqrt{S_gS_h}\big]=\int_0^1\sqrt{F_{g,k}^{-1}(U)\,F_{h,k}^{-1}(U)}\,\mathrm dU .
\tag{21}
$$
强度的方差、三阶矩和组间协方差均为$[0,1]$上关于$U$的一维积分。若$\tilde{\boldsymbol\Sigma}_k$不正定，先按比例缩小组间元素，仍不正定时按比例减小$\boldsymbol\gamma_k$。
5. **其余方向。** $[\mathbf a_k,\mathbf G_k]$为$\mathbf w$对$[1,\mathbf u^{\mathsf T}]$的$\gamma_{ik}$加权最小二乘解，$\boldsymbol\Omega_k$为其加权残差协方差。

以上各步均为闭式或一维求根，不需要迭代。由式(9)、(21)，各分量的电压均值和协方差与AQGMM相同，电气量均值也相同；式(21a)另使天气驱动方向的三阶矩与估计参数场景一致，从而表示偏向减发一侧的尾部。

每个分量的参数为：$\mathbf m_k$、$\tilde{\boldsymbol\Sigma}_k$、$\mathbf a_k$、$\mathbf G_k$、$\boldsymbol\Omega_k$，数目与同$K$标准TMM的$\boldsymbol\mu_k$、$\mathbf C_k$相同（只是同一协方差按方向重新组织），另加$G$个自由度和$r$个均值移动量。$s_{\max,g}$由源荷出力限值确定，不是估计参数。

### 4.5 电气量分布

给定分量$k$和$U$，由式(17)、(19)，$\boldsymbol\xi$为高斯分布，

$$
\mathrm{Cov}(\boldsymbol\xi\mid k,U)=\mathbf Q\begin{bmatrix}\boldsymbol\Sigma_U & \boldsymbol\Sigma_U\mathbf G_k^{\mathsf T}\\ \mathbf G_k\boldsymbol\Sigma_U & \mathbf G_k\boldsymbol\Sigma_U\mathbf G_k^{\mathsf T}+\boldsymbol\Omega_k\end{bmatrix}\mathbf Q^{\mathsf T},\qquad
\boldsymbol\Sigma_U=\mathbf D(U)\tilde{\boldsymbol\Sigma}_k\mathbf D(U),
\tag{22}
$$

$\mathbf Q=[\mathbf Q_u,\mathbf Q_w]$，均值为$\mathbf Q[\mathbf m_U;\ \mathbf a_k+\mathbf G_k\mathbf m_U]$，$\mathbf m_U=\mathbf m_k+\boldsymbol\gamma_k\odot(\mathbf s(U)-\mathbf 1)$。经式(7)换回$\mathbf x$后，式(5)的条件特征函数$\psi_q(t\mid k,U)$按附录A计算，

$$
\psi_q(t)=\sum_k\pi_k\int_0^1\psi_q(t\mid k,U)\,\mathrm dU .
\tag{23}
$$

各组强度由同一个$U$决定，无论分为几组，强度积分均为$[0,1]$上的一维积分，采用Gauss–Legendre求积；截断使$S_g(U)$有界，被积函数在$U\to1$处保持光滑。由式(23)经Gil-Pelaez反演[22]得到电气量的概率分布函数，求导得到概率密度。节点低电压概率为幅值平方低于下限平方的概率，过电压概率为高于上限平方的概率；支路功率按实际功率方向和规定限值计算。

### 4.6 三种模型的关系

**表2 三种电压联合分布模型**

| | GMM（AQGMM） | 标准TMM | 分组t混合 |
|---|---|---|---|
| 每分量波动强度 | 无 | 1个，作用于全部$d$个方向 | $G$组，作用于天气驱动的$r$个方向，同步、截断，并带动均值移动 |
| 强度作用范围的确定 | 无 | 无 | 第4.2节，由注入共同放大和运行点雅可比确定 |
| 每分量参数 | $d+d(d+1)/2$ | GMM加1个自由度 | GMM加$G$个自由度和$r$个均值移动量 |
| 参数估计 | 期望最大化 | 在AQGMM分量归属上按矩确定 | 在AQGMM分量归属上按矩确定 |
| 电气量计算的积分维数 | 0 | 1 | 1 |
| 退化关系 | | $\nu_k\to\infty$为GMM | 无分组为GMM；全部注入一组且$r=d$为标准TMM |

## 5 计算步骤

1. **源荷场景与节点电压。** 按源荷功率规律生成注入场景，由第1.1节快速映射计算节点电压，并另生成一组场景用于选择分量数。
2. **确定天气驱动电压方向。** 按第4.2节步骤1—4得到各组方向和强度上限。
3. **估计AQGMM。** 对给定$K$由期望最大化得到分量归属$\gamma_{ik}$；$K$按第二组场景上的电压似然选择。
4. **确定TMM参数。** 标准TMM按式(9)、(10)，分组t混合按第4.4节。
5. **构造电气量二次型。** 按式(5)由网络参数得到节点电压幅值平方、支路功率和网损的系数。
6. **计算电气量分布。** 按式(11)或(23)计算特征函数，经反演得到概率分布函数、概率密度和限值概率。

步骤1中锚点处的交流潮流和步骤2中平均注入处的交流潮流是全部计算中仅有的完整交流潮流求解（锚点可直接取在平均注入处，此时两者为同一次）。MC方法（逐场景求解交流潮流）只用于检验计算精度。

---

## 附录A 高斯电压下电气量二次型的特征函数

设$\mathbf x\sim\mathcal N(\boldsymbol\mu,\mathbf C)$，$\mathbf C=\mathbf L\mathbf L^{\mathsf T}$，$\mathbf L^{\mathsf T}\mathbf A_q\mathbf L=\mathbf P\boldsymbol\Lambda\mathbf P^{\mathsf T}$，$\boldsymbol\Lambda=\mathrm{diag}(\lambda_1,\ldots,\lambda_d)$。令

$$
c_0=\boldsymbol\mu^{\mathsf T}\mathbf A_q\boldsymbol\mu+\mathbf b_q^{\mathsf T}\boldsymbol\mu+c_q,\qquad
\boldsymbol\beta=\mathbf P^{\mathsf T}\mathbf L^{\mathsf T}(2\mathbf A_q\boldsymbol\mu+\mathbf b_q),
$$

则$q=c_0+\sum_j(\beta_jy_j+\lambda_jy_j^2)$，$y_j$相互独立且服从标准正态分布[23]，

$$
\psi_q(t)=\exp\!\Big(\mathrm itc_0-\frac{t^2}{2}\sum_j\frac{\beta_j^2}{1-2\mathrm i\lambda_jt}\Big)\prod_j(1-2\mathrm i\lambda_jt)^{-1/2}.
\tag{A1}
$$

概率分布函数由Gil-Pelaez反演[22]

$$
F_q(q_0)=\frac12-\frac1\pi\int_0^\infty\frac{\Im\big[\mathrm e^{-\mathrm itq_0}\psi_q(t)\big]}{t}\,\mathrm dt .
\tag{A2}
$$

式(22)的协方差随$U$变化，每个强度积分点需重新计算$\mathbf L$和$\boldsymbol\Lambda$。

## 附录B 截断强度分布

$V=\nu/\chi^2_\nu$限制在$V\le v_{\max}$，即$\chi^2_\nu\ge\nu/v_{\max}$，该事件概率$p_{\max}=\Pr(\chi^2_\nu\ge\nu/v_{\max})$。按均匀随机数$U$生成

$$
V(U)=\frac{\nu}{\chi^{2}_{\nu,\,\mathrm{isf}}(U\,p_{\max})},\qquad S(U)=\frac{V(U)}{\mathrm E[V]},
\tag{B1}
$$

$\chi^2_{\nu,\mathrm{isf}}(p)$为上侧$p$分位点。$U$由0增至1时$S$单调增大，上限$S(1)=v_{\max}/\mathrm E[V]$。给定$s_{\max}$和$b$，先对每个$\nu$求$v_{\max}$使$S(1)=s_{\max}$，再求$\nu$使$\mathrm E[S^2]=b$，两者均为单调一维求根。$b$取不超过$0.98\,s_{\max}$；不截断时$\mathrm E[S^2]=(\nu-2)/(\nu-4)$，即式(10)。

## 附录C 符号

| 符号 | 含义 | 单位或维数 |
|---|---|---|
| $\mathbf p$ | 非平衡节点净注入有功、无功 | p.u.，$d$ |
| $\mathbf x$ | 非平衡节点电压实部、虚部 | p.u.，$d=2(n-1)$ |
| $\boldsymbol\xi$ | 按各方向标准差归一化的电压坐标 | $d$ |
| $\mathbf J_a$, $\mathbf J_0$ | 锚点、平均注入处的潮流雅可比矩阵 | $d\times d$ |
| $q$ | 节点电压幅值平方、支路有功或无功、网损 | p.u.²、MW、MVAr |
| $K$, $\pi_k$ | 分量数、分量权重 | |
| $\gamma_{ik}$ | AQGMM中场景$i$归属分量$k$的概率 | |
| $S$, $S_g$ | 波动强度（均值为1） | |
| $\nu_k$, $\nu_{g,k}$ | 自由度 | |
| $c_{ij}$, $c^*$ | 注入$i$、$j$的共同放大量及其判别阈值 | |
| $\mathcal W_g$ | 第$g$个天气过程放大的注入集合 | |
| $\mathbf Q_g$, $r_g$ | 第$g$组电压方向及其维数 | $d\times r_g$ |
| $U$ | 同一场景的天气形势强弱 | $(0,1)$ |
| $s_{\max,g}$ | 第$g$组强度上限 | |
| $\mathbf a_k$, $\mathbf G_k$, $\boldsymbol\Omega_k$ | 其余方向的截距、回归系数、残差协方差 | |

## 参考文献（正文新增，编号接引言）

[14] 吴涵. 基于随机变量二次型的配电网交流概率潮流建模和解析计算方法(一)：从电量到功率量的推导[J]. 中国电机工程学报, 2026, 46(3): 928–941.

[15] PEEL D, MCLACHLAN G J. Robust mixture modelling using the t distribution[J]. Statistics and Computing, 2000, 10(4): 339–348.

[17] MARDIA K V. Measures of multivariate skewness and kurtosis with applications[J]. Biometrika, 1970, 57(3): 519–530.

[18] DAUL S, DE GIORGI E, LINDSKOG F, et al. The grouped t-copula with an application to credit risk[J]. Risk, 2003, 16(11): 73–76.【页码文献待核查】

[19] DEMARTA S, MCNEIL A J. The t copula and related copulas[J]. International Statistical Review, 2005, 73(1): 111–129.

[20] HINTZ E, HOFERT M, LEMIEUX C. Grouped normal variance mixtures[J]. Risks, 2020, 8(4): 103.

[21] CARDOSO J-F. Source separation using higher order moments[C]//Proc. IEEE ICASSP. 1989: 2109–2112.

[22] GIL-PELAEZ J. Note on the inversion theorem[J]. Biometrika, 1951, 38(3–4): 481–482.

[23] IMHOF J P. Computing the distribution of quadratic forms in normal variables[J]. Biometrika, 1961, 48(3–4): 419–426.

AQGMM：课题组前作《解析交流概率潮流中的电压联合分布估计》，正式书目待作者补齐。

---

## 6 实现与证据对应（内部，不进入论文）

| 内容 | 实现 | 证据状态 |
|---|---|---|
| 快速映射（式(3)、(4)） | `+BTMM_core/+inverse/map_samples_with_atlas.m`；Python调用`tests/python/fast_voltage_map.py` | 已由数值实验支持：算例1、两片云区各12288个场景1个锚点，功率残差<1e-8 p.u.，与逐场景交流潮流电压最大差2.7e-8 p.u. |
| 二次型（式(5)） | 前作[14]；`tests/python/run_pv_fixed_forecast_case33.py`中`evaluate_outputs` | 已由代码实现支持 |
| 标准TMM按矩（式(9)、(10)） | `tests/python/moment_matched_tmm_case33.py`中`moment_models` | 已由数值实验支持（33节点） |
| 稀释关系（式(12)） | 推导 | 已由理论推导支持；与33节点标准TMM自由度30—200的结果一致 |
| 分组步骤1—4（第4.2节） | `tests/python/block_split_rule_check.py`中`split`；上限`tests/python/truncated_strength_case33.py`中`strength_cap` | 已由数值实验支持：6个已有场景、4个构造场景、AQGMM前作输入判别正确；8192个场景时2处临界误判 |
| 分组t参数估计（第4.4节，含均值移动式(21a)） | `tests/python/shift_grouped_t_case33.py`中`shift_grouped_model`；截断强度`tests/python/grouped_strength_tmm_case33.py`中`TruncatedStrength` | 已由代码实现支持；强度的各阶矩按$U$的2000点均匀网格计算 |
| 平均注入处潮流与锚点 | 当前`split`在平均注入处另求一次潮流，快速映射锚点取离平均注入最近的场景 | 两者尚未统一为同一次潮流【尚未实现】 |
| 解析计算（式(11)、(23)、附录A） | `tests/python/analytic_grouped_t_case33.py`（三种模型；分组t对$U$分4段各12点Gauss–Legendre，标准TMM 40点广义Gauss–Laguerre，Gil-Pelaez反演） | 已由数值实验支持（E-ANALYTIC-20261024）：与模型生成约105万组电压的累积概率最大差0.0006—0.0014，积分节点加密结果不变 |
| 两个核心论点 | 同K尾部更准；少分量达到相近精度 | 按实测记录构造的场景（E-MAINLINE-SHIFT-20261024）：S1（参与配置选择）与S3（独立检验）节点电压、支路有功、网损多数K下更准，S3中K=2的节点电压尾部AQGMM在K≤16内达不到；S2E1、S2E2（两种运行状态混合）只在K=1更准。单次运行，模型电压262144组 |
