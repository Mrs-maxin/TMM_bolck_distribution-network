# 基于TMM的配电网交流概率潮流计算方法（理论方法部分）

日期：2026-10-24。本文件按论文正文格式撰写理论方法，第1—4节为正文，附录A、B为正文附录，第5节为内部的实现与证据对应，不进入论文。正文涉及两种电压联合分布模型：作为比较基础的GMM（AQGMM）和标准TMM。2026-10-26由合并稿拆出，分组t混合（原第4节）移至分组t论文。

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

若每个分量在给定一个标量随机变量（下文的波动强度）后为高斯分布，则式(5)的条件特征函数为闭式（附录A），对该标量作一维积分并按分量加权，即得电气量的特征函数，再经反演得到概率分布函数和概率密度。因此，电气量尾部的计算精度由电压联合分布的尾部决定，本文的核心是电压联合分布模型。第2、3节依次给出GMM和标准TMM两种模型，两者共用第1.1节的快速映射和式(5)的二次型。

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

## 4 计算步骤

1. **源荷场景与节点电压。** 按源荷功率规律生成注入场景，由第1.1节快速映射计算节点电压，并另生成一组场景用于选择分量数。
2. **估计AQGMM。** 对给定$K$由期望最大化得到分量归属$\gamma_{ik}$；$K$按第二组场景上的电压似然选择。
3. **确定标准TMM参数。** 按式(9)、(10)。
4. **构造电气量二次型。** 按式(5)由网络参数得到节点电压幅值平方、支路功率和网损的系数。
5. **计算电气量分布。** 按式(11)计算特征函数，经Gil-Pelaez反演（附录A式(A2)）得到概率分布函数、概率密度和限值概率。

步骤1中锚点处的交流潮流是全部计算中唯一的完整交流潮流求解，电气量计算只用电压二次型。MC方法（逐场景求解交流潮流）只用于检验计算精度。

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

标准TMM给定强度$s$时协方差为$s\mathbf C_k$，$\mathbf L$按$\sqrt s$缩放，$\boldsymbol\Lambda$按$s$缩放，各强度积分点不需重新分解。

## 附录B 符号

| 符号 | 含义 | 单位或维数 |
|---|---|---|
| $\mathbf p$ | 非平衡节点净注入有功、无功 | p.u.，$d$ |
| $\mathbf x$ | 非平衡节点电压实部、虚部 | p.u.，$d=2(n-1)$ |
| $\boldsymbol\xi$ | 按各方向标准差归一化的电压坐标 | $d$ |
| $\mathbf J_a$ | 快速映射锚点处的潮流雅可比矩阵 | $d\times d$ |
| $q$ | 节点电压幅值平方、支路有功或无功、网损 | p.u.²、MW、MVAr |
| $K$, $\pi_k$ | 分量数、分量权重 | |
| $\gamma_{ik}$ | AQGMM中场景$i$归属分量$k$的概率 | |
| $S$ | 波动强度（均值为1） | |
| $\nu_k$ | 自由度 | |


## 参考文献（正文新增，编号接引言）

[14] 吴涵. 基于随机变量二次型的配电网交流概率潮流建模和解析计算方法(一)：从电量到功率量的推导[J]. 中国电机工程学报, 2026, 46(3): 928–941.

[15] PEEL D, MCLACHLAN G J. Robust mixture modelling using the t distribution[J]. Statistics and Computing, 2000, 10(4): 339–348.

[17] MARDIA K V. Measures of multivariate skewness and kurtosis with applications[J]. Biometrika, 1970, 57(3): 519–530.

[22] GIL-PELAEZ J. Note on the inversion theorem[J]. Biometrika, 1951, 38(3–4): 481–482.

[23] IMHOF J P. Computing the distribution of quadratic forms in normal variables[J]. Biometrika, 1961, 48(3–4): 419–426.

AQGMM：课题组前作《解析交流概率潮流中的电压联合分布估计》，正式书目待作者补齐。

---

---

## 5 实现与证据对应（内部，不进入论文）

| 内容 | 实现 | 证据状态 |
|---|---|---|
| 快速映射（式(3)、(4)） | `shared/+BTMM_core/+inverse/map_samples_with_atlas.m`；Python调用`shared/tests/python/fast_voltage_map.py` | 已由数值实验支持（合并版E-FASTMAP-20261024） |
| 二次型（式(5)） | 前作[14]；`shared/tests/python/run_pv_fixed_forecast_case33.py`中`evaluate_outputs` | 已由代码实现支持 |
| 标准TMM按矩（式(9)、(10)） | `shared/tests/python/moment_matched_tmm_case33.py`中`moment_models` | 已由代码实现支持 |
| 解析计算（式(11)、附录A） | `shared/tests/python/analytic_grouped_t_case33.py`（三种模型共用；标准TMM 40点广义Gauss–Laguerre，Gil-Pelaez反演） | 已由数值实验支持（合并版E-ANALYTIC-20261024） |
| 重尾分布注入算例 | `shared/tests/python/run_direct_heavytail_case33.py`；本文表格`paper_standardTMM/result/heavytail_injection/case33bw_20261026/` | 见`01_IDEA/evidence_map.md` E-STD-HEAVYTAIL-20261026 |
