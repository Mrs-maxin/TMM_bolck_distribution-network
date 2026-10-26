# 极端天气下配电网电气量尾部分布的解析概率潮流计算

## 0 引言

极端高温增多、强降水增强等气候变化增加了电力系统应对极端天气的运行压力[1]。新能源出力受天气影响，云层遮蔽和移出可引起光伏出力快速下降与恢复[2]。较大的出力变化会改变配电网受电、反送功率和沿线电压。电压越限较深或支路输送功率较大时，较少出现的运行状态也可能产生较严重的后果。因此，交流概率潮流需要准确描述这些状态对应的电气量尾部，为极端天气下的运行风险分析提供概率依据。

小概率运行状态的后果取决于电压偏离允许范围的程度及支路输送功率的大小，其发生概率由相应电气量分布在限值外侧的概率确定。网损随电流平方增大，也需要计及较高损耗水平的发生概率。分布主体的近似精度主要反映常见运行状态的描述程度，严重偏差对应的概率则需要由尾部分布计算。

现有概率潮流研究已针对小概率运行状态改进计算方法。交叉熵重要性抽样根据目标区域调整源荷场景的生成规律，再利用概率权重计算越限概率[3]。混合代理方法对尾部运行状态调用精确潮流模型修正，以减小近似潮流在小概率区域的偏差[4]。这些方法通过增加目标区域的潮流信息改善概率估计。解析概率潮流则需要建立电气量分布的计算表达式，使指定限值下的概率能够由分布参数直接计算。

高斯混合模型GMM能够描述非高斯分布及变量间的相关关系。已有研究利用输入输出联合GMM和分段线性潮流计算电气量分布[5]。基于电压二次型的研究保留了原始交流功率关系，并给出高斯电压及GMM电压下的功率量分布计算方法[6]。完整GMM交流概率潮流文稿进一步衔接了源荷注入、电压分布估计与电气量分布计算【完整GMM英文稿引文待补充】。本文沿用这套计算基础，扩展联合电压的尾部表示。

较大的电压偏差经交流功率关系改变支路功率和网损，因此联合电压的尾部描述会影响相应电气量的小概率计算。当常见电压波动集中于较窄范围，而少量偏差延伸较远时，有限GMM分量需要兼顾分布主体和尾部。增加分量可以调整分布形态，也增加参数估计和电气量计算的工作量。分量内的幅度变化为兼顾常见波动与较大偏差提供了进一步的表示手段。

t混合模型TMM通过分量内的随机尺度描述不同幅度的偏差[7]。自由度调节标准t分量的尾部衰减，电压相关关系则决定多个节点同时偏离的方式。配电网不同注入对应的电压变化幅度并不相同，电压升高与降低两侧也可能存在差异。将t分量用于交流概率潮流，需要使幅度变化的作用范围与网络电压响应相对应，并计及两侧偏差的不对称性。

本文面向极端天气下配电网电气量的小概率尾部，建立基于TMM的交流概率潮流计算方法。联合电压模型在分量内描述较大偏差，并计及电压升高与降低两侧的偏斜。幅度变化的作用范围由注入变化及运行点潮流关系确定。电气量分布由给定幅度下的交流功率关系计算，再按幅度的发生概率和混合分量权重合成，从而得到节点电压、支路有功和无功功率、系统有功网损的分布及限值外侧概率。

## 1 面向小概率尾部的联合电压建模

### 1.1 电压尾部分布的表示

考虑拓扑固定的平衡配电网，平衡节点电压保持给定值，其余节点采用PQ节点模型。非平衡节点净注入有功和无功均以发电减负荷为正。源荷场景反映给定运行条件下的联合功率规律，其对应电压采用已有快速交流潮流方法计算【快速潮流文稿引文待补充】。电压与功率均采用各自基准下的标幺值。

设非平衡节点集合为$\mathcal N$，节点总数为$n$，电压实部和虚部组成

$$
\mathbf x=\big[\Re(\mathbf V_{\mathcal N})^{\mathsf T},\ \Im(\mathbf V_{\mathcal N})^{\mathsf T}\big]^{\mathsf T}\in\mathbb R^d,\qquad d=2(n-1).
\tag{1}
$$
该联合电压包含各节点幅值与相角变化所需的信息。用于估计参数的电压均值记为$\bar{\mathbf x}$。在协方差正定的条件下，采用归一化电压$\boldsymbol\xi=\mathbf W(\mathbf x-\bar{\mathbf x})$，其中$\mathbf W=\mathbf D^{-1}\mathbf E^{\mathsf T}$，$\mathbf E$为协方差的正交特征向量矩阵，$\mathbf D$为各特征方向标准差组成的对角阵。归一化使不同幅度的电压变化采用相同的数值尺度，电气量计算时恢复原电压坐标。

联合电压分布由$K$个分量组成，分量权重为$\pi_k$，分量归属记为$C$，满足$\Pr(C=k)=\pi_k$及$\sum_k\pi_k=1$。标准TMM的第$k$个分量可写为[7]

$$
\boldsymbol\xi\mid C=k,S=s\sim\mathcal N(\boldsymbol\mu_k,s\mathbf C_k),\qquad
S=\frac{\nu_k-2}{\chi^2_{\nu_k}},\qquad \nu_k>4.
\tag{2}
$$
$S$为无量纲波动强度，均值为1；$\boldsymbol\mu_k$和$\mathbf C_k$分别为分量电压均值和协方差。强度较小时，电压偏差集中；强度较大时，较远偏差的概率增加。自由度$\nu_k$控制强度的变化程度，$\nu_k\to\infty$时分量趋于高斯分布。自由度大于4使电压四阶矩及相应二次电气量的方差存在。

较大出力偏差与普通负荷小幅变化可以同时出现，其对应的电压响应幅度有所不同。因此，联合电压采用分别作用于不同方向的波动强度，并保留这些方向与其余电压变化之间的联系。

### 1.2 电压变化方向的确定

大幅注入偏差经网络影响多个节点电压，运行点灵敏度给出这种影响的主要方向。幅度调整所涉及的注入先按平方偏差的相关程度确定。对标准差大于零的注入$i$和$j$，分别以$e_i$和$e_j$表示相对场景均值的偏差，计算

$$
c_{ij}=\operatorname{corr}(e_i^2,e_j^2)-\operatorname{corr}(e_i,e_j)^2.
\tag{3}
$$
联合高斯偏差的总体统计量为零。判别阈值取相同协方差、相同场景数的五组高斯源荷场景所得统计量的最大值。超过阈值的注入具有较强的平方偏差相关性，将其相连，每个包含至少两个注入的连通集合记为$\mathcal W_g$。这些集合用于确定幅度变化的作用范围。

在电压计算所采用的运行点处，记潮流雅可比矩阵为$\mathbf J_0$，选取$\mathcal W_g$中注入的矩阵为$\mathbf L_g$。相应电压灵敏度由线性方程确定，

$$
\mathbf J_0\mathbf H_g=\mathbf L_g.
\tag{4}
$$
$\mathbf H_g$各列给出一项注入变化对全网电压的影响。将$\mathbf W\mathbf H_g$按集合规模由大到小依次正交化，并作奇异值分解。保留奇异值大于本组最大值$10^{-10}$倍的方向，得到各组一阶电压变化方向。

交流潮流的非线性使一阶方向以外仍可能出现较大的电压偏差。记一阶方向外的归一化电压为$\mathbf w^{(0)}$，用$N$组电压计算

$$
\mathbf M=\frac1N\sum_{i=1}^{N}\|\mathbf w_i^{(0)}\|^2\mathbf w_i^{(0)}(\mathbf w_i^{(0)})^{\mathsf T}.
\tag{5}
$$
补充方向的特征值应超过相同协方差高斯电压的最大值，其电压方差占比按原坐标折算后不低于$10^{-3}$。此外，该方向平方幅度与一阶方向平方幅度之和的相关系数应超过高斯情形的最大值。满足三项要求的方向并入平方幅度相关程度最高的一阶组，使一阶灵敏度之外的相关大幅电压偏差也参与幅度调整。

各组方向组成$\mathbf Q_u$，维数为$r$；其余正交方向组成$\mathbf Q_w$。记

$$
\mathbf u=\mathbf Q_u^{\mathsf T}\boldsymbol\xi,\qquad
\mathbf w=\mathbf Q_w^{\mathsf T}\boldsymbol\xi,\qquad
\mathbf Q=[\mathbf Q_u,\mathbf Q_w].
\tag{6}
$$
一个电压方向包含多个节点实部、虚部的组合，反映一组相关电压变化。$\mathbf u$保留需要调整幅度的变化，$\mathbf w$保留其余电压变化。

### 1.3 计及偏斜的TMM电压分布

电压尾部在升高和降低两侧的延伸可能不同。仅对偏差作对称放大时，同一分量两侧采用相同的幅度规律。本文使分量内的均值随波动强度变化，以描述较大偏差偏向一侧的电压分布。

设共有$G$组电压方向，第$j$个方向所属组为$g(j)$。第$k$个分量内，$\mathbf u$写为

$$
\mathbf u=\mathbf m_k+\boldsymbol\gamma_k\odot[\mathbf s(U)-\mathbf1]+\mathbf D(U)\mathbf z,\qquad
\mathbf z\sim\mathcal N(\mathbf0,\tilde{\boldsymbol\Sigma}_k),
\tag{7}
$$
其中$\mathbf m_k$为均值，$\boldsymbol\gamma_k$为均值随强度变化的系数，$\odot$表示对应元素相乘。$\mathbf s(U)$的第$j$项为$S_{g(j),k}(U)$，幅度矩阵为

$$
\mathbf D(U)=\operatorname{diag}\!\left(\sqrt{S_{g(1),k}(U)},\ldots,\sqrt{S_{g(r),k}(U)}\right),\qquad
S_{g,k}(U)=F_{g,k}^{-1}(U),\quad U\sim\operatorname{Uniform}(0,1).
\tag{8}
$$
$F_{g,k}$为第$g$组强度的累积分布函数。各组非恒定强度由同一$U$确定，按相同次序增减，属于分组正态方差混合的一种依赖设定[8]。这一设定适用于组间偏差幅度具有同步变化特征的情形，并使各组幅度能够由一个变量积分合成。

输入源荷场景的取值范围用于确定幅度放大的上限。对第$g$组，令$e_{i,\max}$为第$i$项注入在输入场景中的最大绝对偏差，$\sigma_{e_i}$为偏差标准差，取强度上限

$$
s_{\max,g}=\operatorname{median}_{i\in\mathcal W_g}
\left(\frac{e_{i,\max}}{\sigma_{e_i}}\right)^2.
\tag{9}
$$
强度采用截断并归一化的逆伽马分布。令$V=\nu_{g,k}/\chi^2_{\nu_{g,k}}$，在$V\le v_{\max,g,k}$条件下取值，再令$S_{g,k}=V/\mathrm E[V]$。截断点使$\max S_{g,k}=s_{\max,g}$，并保持$\mathrm E[S_{g,k}]=1$。这一近似利用输入注入范围限制幅度放大程度，各给定强度下的电压采用高斯条件分布。

其余电压变化按条件高斯分布表示，

$$
\mathbf w\mid\mathbf u,C=k\sim
\mathcal N(\mathbf a_k+\mathbf G_k\mathbf u,\boldsymbol\Omega_k).
\tag{10}
$$
$\mathbf G_k\mathbf u$保留两类电压变化之间的线性相关部分，$\boldsymbol\Omega_k$描述条件残差的协方差。模型假设$\mathbf z$与$U$独立，条件残差与二者独立。幅度变化通过$\mathbf u$及$\mathbf G_k\mathbf u$共同影响节点电压，均值移动则调节两侧尾部的不对称性。

各组强度恒为1时，分量为高斯分布。全部电压方向同属一组、均值移动为零且强度采用未截断的标准t尺度时，分量为标准多元t分布。本文TMM保留这两种特殊情形，并采用分组幅度和均值移动描述所关注的电压尾部。

### 1.4 TMM参数估计

电压分布参数由源荷场景对应的电压确定。对于给定分量数$K$，先采用已有GMM参数估计方法得到场景$i$属于分量$k$的概率$\gamma_{ik}$【完整GMM英文稿引文待补充】。令$N_k=\sum_i\gamma_{ik}$，分量权重取$\pi_k=N_k/N$，$\mathbf m_k$和$\mathbf C_{u,k}$分别取$\mathbf u$按$\gamma_{ik}$加权的均值与协方差。相同分量归属使GMM与TMM使用相同的场景权重，新增参数进一步描述分量内较大偏差及两侧的不对称性。

各组强度先按对称近似下的多元峰度确定[9]。第$g$组有$r_g$个方向，记

$$
\delta_{ik,g}=(\mathbf u_{i,g}-\mathbf m_{k,g})^{\mathsf T}
\mathbf C_{u,k,gg}^{-1}(\mathbf u_{i,g}-\mathbf m_{k,g}),\qquad
b_{g,k}=\frac{\sum_i\gamma_{ik}\delta_{ik,g}^{\,2}}{N_kr_g(r_g+2)}.
\tag{11}
$$
当$b_{g,k}\le1.05$时，取$S_{g,k}=1$。其余情形以$\mathrm E[S_{g,k}^2]=\min(b_{g,k},0.98s_{\max,g})$近似确定截断强度参数。对于给定自由度，先求截断点满足强度上限，再调整自由度满足强度二阶矩。

标准TMM采用全维电压按分量归属加权的归一化多元峰度$b_k$。其未截断强度满足$\nu_k=(4b_k-2)/(b_k-1)$，并取$\nu_k\ge4.1$；$b_k\le1$时取高斯分量。截断强度的自由度在正值范围内确定，其有界幅度使高阶矩存在。

电压的三阶中心矩用于确定均值移动。对方向$j$，记$C_{jj}$为分量方差，$\mu_{3,j}$为分量三阶中心矩，$v_g=\operatorname{Var}(S_{g,k})$，$\kappa_g=\mathrm E[(S_{g,k}-1)^3]$。均值移动系数满足

$$
\mu_{3,j}=\gamma_{k,j}^{\,3}\big(\kappa_{g(j)}-3v_{g(j)}^2\big)
+3\gamma_{k,j}v_{g(j)}C_{jj}.
\tag{12}
$$
取绝对值最小且满足$C_{jj}-\gamma_{k,j}^2v_{g(j)}\ge0.1C_{jj}$的实根；无满足条件的实根时取零。强度恒定的方向取$\gamma_{k,j}=0$。均值移动与强度参数采用顺序估计，强度峰度关系作为对称近似使用，偏斜后的四阶矩由最终电压模型确定。

分量内的散布矩阵按协方差关系计算，

$$
\tilde\Sigma_{k,jl}=
\frac{C_{u,k,jl}-\gamma_{k,j}\gamma_{k,l}\operatorname{Cov}(S_{g(j),k},S_{g(l),k})}
{\mathrm E[\sqrt{S_{g(j),k}S_{g(l),k}}]}.
\tag{13}
$$
强度的各阶矩及组间矩由$U$上的一维积分计算。散布矩阵正定时，该关系保持目标电压协方差。出现非正定情形时，先按比例缩小组间散布元素，再根据需要减小均值移动系数并重新计算散布矩阵。修正后的电压协方差和三阶矩由最终参数确定。

$\mathbf a_k$和$\mathbf G_k$由$\mathbf w$对$\mathbf u$的分量归属加权最小二乘确定，$\boldsymbol\Omega_k$取相应残差协方差。由此形成包含均值、相关关系、偏斜和幅度变化的联合电压分布。

## 2 电气量尾部分布与越限概率的解析计算

### 2.1 电气量分布的计算

支路功率和网损取决于相关节点电压的组合。给定TMM的分量及强度后，联合电压为高斯分布，可以沿用已有交流概率潮流计算关系求取电气量分布[6]。

记$\mathbf T=\mathbf W^{-1}$，$\mathbf m_{k,U}=\mathbf m_k+\boldsymbol\gamma_k\odot[\mathbf s(U)-\mathbf1]$，$\boldsymbol\Sigma_{k,U}=\mathbf D(U)\tilde{\boldsymbol\Sigma}_k\mathbf D(U)$。令$\mathbf B_k=\mathbf Q_u+\mathbf Q_w\mathbf G_k$，原坐标电压的条件均值和协方差为

$$
\boldsymbol\mu_{k,U}=\bar{\mathbf x}+\mathbf T\big[\mathbf Q_u\mathbf m_{k,U}
+\mathbf Q_w(\mathbf a_k+\mathbf G_k\mathbf m_{k,U})\big],
\qquad
\mathbf C_{k,U}=\mathbf T\big[\mathbf B_k\boldsymbol\Sigma_{k,U}\mathbf B_k^{\mathsf T}
+\mathbf Q_w\boldsymbol\Omega_k\mathbf Q_w^{\mathsf T}\big]\mathbf T^{\mathsf T}.
\tag{14}
$$
条件均值随强度改变偏差的主要取值位置，条件协方差描述对应幅度下各节点电压的相关变化。两者共同决定支路有功、无功及网损的条件分布。

节点电压幅值平方、支路有功和无功功率、系统有功网损统一记为$q$，其交流电压关系为[6]

$$
q(\mathbf x)=\mathbf x^{\mathsf T}\mathbf A_q\mathbf x+\mathbf b_q^{\mathsf T}\mathbf x+c_q,
\qquad \mathbf A_q=\mathbf A_q^{\mathsf T}.
\tag{15}
$$
系数由网络导纳和平衡节点电压确定。支路两端功率分别以流入支路为正，总有功网损取各在运支路两端有功功率之和。节点电压幅值由其平方换算。各电气量使用同一联合电压，保留不同节点及支路之间的相关关系。

以$\Psi_q(t;\boldsymbol\mu,\mathbf C)$表示高斯电压下电气量$q$的特征函数，采用已有二次型计算表达式[6]。TMM电压下的电气量特征函数为

$$
\psi_q(t)=\sum_{k=1}^{K}\pi_k\int_0^1
\Psi_q(t;\boldsymbol\mu_{k,U},\mathbf C_{k,U})\,\mathrm dU.
\tag{16}
$$
强度积分将不同幅度下的电气量变化按其发生概率合成，分量加权进一步合成完整分布。分组强度采用同一$U$，因而幅度积分为一维。标准TMM对未截断强度积分时采用广义Gauss–Laguerre求积，本文分组形式采用Gauss–Legendre求积。

电气量的累积分布$F_q(q_0)=\Pr(q\le q_0)$采用Gil–Pelaez反演计算[10]。概率密度由同一特征函数作傅里叶反演，或对累积分布求导。对不同$q_0$计算累积概率，即得到从常见取值到较远尾部的连续分布描述。

### 2.2 尾部概率与运行限值

电压越限概率由电压上下限直接确定。对节点$i$，令$q_{V_i}=|V_i|^2$，电压下限和上限分别为$V_i^{\min}$、$V_i^{\max}$，则

$$
p_{V_i}^{\mathrm{low}}=F_{q_{V_i}}\big((V_i^{\min})^2\big),\qquad
p_{V_i}^{\mathrm{high}}=1-F_{q_{V_i}}\big((V_i^{\max})^2\big).
\tag{17}
$$
沿电压下限外侧降低电压阈值、沿上限外侧提高电压阈值，可以得到不同越限程度对应的发生概率。

支路功率概率按实际功率方向计算。对有功或无功功率$q$，给定区间$[q_{\min},q_{\max}]$外的概率为

$$
p_q^{\mathrm{out}}=F_q(q_{\min})+1-F_q(q_{\max}).
\tag{18}
$$
正向与反向功率的限值分别按支路有功或无功分析要求给定。系统有功网损$L$超过给定水平$\ell$的概率为$1-F_L(\ell)$，用于描述高网损取值的发生可能性。

对给定小概率$\alpha$，电气量下侧和上侧分位值分别为$q_{\alpha}=F_q^{-1}(\alpha)$和$q_{1-\alpha}=F_q^{-1}(1-\alpha)$。尾部概率给出指定电压或功率水平的发生可能性，分位值给出指定概率水平对应的电气量幅度。运行限值及其外侧的取值共同表征所关注的小概率严重偏差。

### 2.3 计算步骤

概率潮流输入包括网络参数、平衡节点电压、源荷场景及需要分析的电压和功率限值。完整计算按以下步骤进行。

1. 采用已有快速交流潮流方法计算源荷场景对应的节点电压，保留运行点潮流雅可比矩阵。
2. 完成电压归一化，根据注入偏差、运行点电压灵敏度及一阶方向外的电压变化，确定幅度调整的作用方向。
3. 对给定分量数估计GMM分量归属，再确定TMM的强度、均值移动、散布和其余电压变化参数。没有形成幅度调整方向时，采用原GMM电压分布。
4. 由各分量及强度下的电压均值和协方差计算电气量条件特征函数，完成强度积分与分量加权。
5. 反演得到电气量分布，计算给定限值下的概率及指定概率水平的分位值。

积分精度根据所关注的最小尾部概率设置，并通过加密积分点检查概率计算的稳定性。计算结果分别保留各节点电压、各支路功率及系统网损的分布和限值概率。

## 参考文献

[1] IPCC. Climate Change 2023: Synthesis Report[R]. Geneva: IPCC, 2023. DOI: [10.59327/IPCC/AR6-9789291691647](https://doi.org/10.59327/IPCC/AR6-9789291691647).

[2] WELLBY S J, ENGERER N A. Categorizing the Meteorological Origins of Critical Ramp Events in Collective Photovoltaic Array Output[J]. Journal of Applied Meteorology and Climatology, 2016, 55(6): 1323–1344. DOI: [10.1175/JAMC-D-15-0107.1](https://doi.org/10.1175/JAMC-D-15-0107.1).

[3] LEITE DA SILVA A M, DE CASTRO A M. Risk Assessment in Probabilistic Load Flow via Monte Carlo Simulation and Cross-Entropy Method[J]. IEEE Transactions on Power Systems, 2019, 34(2): 1193–1202. DOI: [10.1109/TPWRS.2018.2869769](https://doi.org/10.1109/TPWRS.2018.2869769).

[4] XU Y, KORKALI M, MILI L, et al. Risk Assessment of Rare Events in Probabilistic Power Flow via Hybrid Multi-Surrogate Method[J]. IEEE Transactions on Smart Grid, 2020, 11(2): 1593–1603. DOI: [10.1109/TSG.2019.2940928](https://doi.org/10.1109/TSG.2019.2940928).

[5] GAO Y, XU X, YAN Z, et al. Analytical Probabilistic Power Flow and Global Sensitivity Analysis of Distribution Systems Based on Gaussian Mixture Model of Input-Output Variables[J]. IEEE Transactions on Power Systems, 2024, 39(3): 5283–5296. DOI: [10.1109/TPWRS.2023.3329197](https://doi.org/10.1109/TPWRS.2023.3329197).

[6] 吴涵. 基于随机变量二次型的配电网交流概率潮流建模和解析计算方法(一)：从电量到功率量的推导[J]. 中国电机工程学报, 2026, 46(3): 928–941. DOI: [10.13334/j.0258-8013.pcsee.242015](https://doi.org/10.13334/j.0258-8013.pcsee.242015).

[7] PEEL D, MCLACHLAN G J. Robust mixture modelling using the t distribution[J]. Statistics and Computing, 2000, 10(4): 339–348. DOI: [10.1023/A:1008981510081](https://doi.org/10.1023/A:1008981510081).

[8] HINTZ E, HOFERT M, LEMIEUX C. Grouped Normal Variance Mixtures[J]. Risks, 2020, 8(4): 103. DOI: [10.3390/risks8040103](https://doi.org/10.3390/risks8040103).

[9] MARDIA K V. Measures of multivariate skewness and kurtosis with applications[J]. Biometrika, 1970, 57(3): 519–530. DOI: [10.1093/biomet/57.3.519](https://doi.org/10.1093/biomet/57.3.519).

[10] GIL-PELAEZ J. Note on the inversion theorem[J]. Biometrika, 1951, 38(3–4): 481–482. DOI: [10.1093/biomet/38.3-4.481](https://doi.org/10.1093/biomet/38.3-4.481).
