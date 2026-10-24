# 推导档案

## 2026-10-24 当前方法

正式方法以论文正文格式写在[02_PAPER/manuscript/method_zh.md](../02_PAPER/manuscript/method_zh.md)，其式(1)—(23)及附录A—C为当前定义。论文只保留三种节点电压联合分布模型：

1. AQGMM（前作GMM），期望最大化估计。
2. 标准TMM：每个分量一个波动强度作用于全部电压方向；在同K的AQGMM分量归属上按矩确定参数（method_zh.md式(9)、(10)）。
3. 分组t混合：天气驱动电压方向按method_zh.md第4.2节4个步骤确定，一阶方向每个天气过程一组、潮流二次响应方向各自一组；各组强度由同一均匀随机数同步决定并按出力限值截断；其余方向为高斯回归；按矩确定参数（式(20)、(21)）。

相对下方来源推导的变化：

- 分块TMM（第A—F节）是分组t只有一组且不截断的特例；第G节多块TMM的各块独立强度改为同步强度，使电气量计算只需一维强度积分（method_zh.md式(23)）。
- 参数由最大似然期望条件最大化（式(B12)—(B15)）改为按矩确定，保持各分量电压均值和协方差，避免最大似然低估光伏驱动方向的方差。
- 第I节共用残差协方差不进入当前方法，保留为归档路线。
- 第J节分组规则补入强度上限（method_zh.md式(16)），二次响应方向单独成组。
- 新增稀释关系：$r$个方向的多元峰度超出量为$b_u-1$、其余$d-r$个方向近似高斯且独立时，全网电压$b-1=r(r+2)(b_u-1)/[d(d+2)]$（method_zh.md式(12)）。33节点$d=64$、$r=8$时系数0.019，$b_u=2$对应$\nu\approx109$。【已由理论推导支持】
- 拟合用节点电压由快速映射（下方“1. 功率到电压”）计算，MC方法只用于检验。【已由数值实验支持】

待完成：分组t的式(23)解析计算【代码尚未实现】；式(21)中$\mathrm E[\sqrt{S_gS_h}]$当前以强度抽样计算【代码待核查】；分组用的平均注入处潮流与快速映射锚点尚未统一为同一次潮流【尚未实现】。

## 来源推导（自原项目derivation.md移入，公式编号保持）

## 2026-10-21 分块TMM中共同波动强度作用范围的确定

本节对应[固定预测时刻方案](plans/pv_fixed_forecast_common_scale_plan_20261021.md)第12节及证据E-PV-BLOCK-TMM-20261021。模型仍沿用TMM的基本结构：每个混合分量一个自由度和一个随机波动强度，给定强度后电压服从高斯分布。新增内容是确定该强度作用于哪些电压变化方向。状态：模型与确定方法已由代码实现支持，IEEE 33节点单一光伏场景已由数值实验支持，其他场景尚未验证。

### A. 分块电压模型

记去除平衡节点后的直角坐标电压为

$$
\mathbf x=\left[\Re(\tilde{\mathbf V}_{\mathcal N})^\top,\ \Im(\tilde{\mathbf V}_{\mathcal N})^\top\right]^\top\in\mathbb R^{d},\qquad d=2(n-1),
\tag{B1}
$$

其中$\tilde{\mathbf V}_{\mathcal N}$为非平衡节点电压相量，$n$为节点数。拟合在训练样本白化坐标中进行：

$$
\boldsymbol\xi=\mathbf W(\mathbf x-\bar{\mathbf x}),\qquad \mathbf W=\mathbf D^{-1}\mathbf E^\top,
\tag{B2}
$$

$\bar{\mathbf x}$为训练电压均值，$\mathbf E$与$\mathbf D=\mathrm{diag}(\sigma_1,\ldots,\sigma_d)$分别为训练电压协方差的特征向量矩阵和沿各特征方向的标准差。设$\mathbf Q=[\mathbf Q_u,\ \mathbf Q_w]$为$d\times d$正交矩阵，$\mathbf Q_u\in\mathbb R^{d\times r}$张成天气驱动的电压变化方向，$\mathbf Q_w$为其正交补。定义

$$
\mathbf u=\mathbf Q_u^\top\boldsymbol\xi\in\mathbb R^{r},\qquad
\mathbf w=\mathbf Q_w^\top\boldsymbol\xi\in\mathbb R^{d-r}.
\tag{B3}
$$

第$k$个混合分量写为

$$
\mathbf u\mid C=k,S_k=s\sim\mathcal N(\mathbf m_k,\ s\boldsymbol\Sigma_k),\qquad
S_k\sim\mathrm{Inv\text{-}Gamma}\!\left(\tfrac{\nu_k}{2},\tfrac{\nu_k}{2}\right),
\tag{B4}
$$

$$
\mathbf w\mid \mathbf u,C=k\sim\mathcal N(\mathbf a_k+\mathbf G_k\mathbf u,\ \boldsymbol\Omega_k).
\tag{B5}
$$

其中$C$为分量标签，$\pi_k=\Pr(C=k)$；$S_k$为该分量的共同波动强度，$\nu_k$为其自由度；$\mathbf m_k$、$\boldsymbol\Sigma_k$为天气驱动块的位置与散布；$\mathbf a_k$、$\mathbf G_k$、$\boldsymbol\Omega_k$为其余块的截距、对天气驱动块的线性回归系数和残差协方差。

式(B4)使同一次天气事件按同一强度$\sqrt{S_k}$放大全部天气驱动方向，保留多个光伏节点电压同时出现较大偏差的机理。式(B5)中，与天气驱动方向相关的那部分电压变化通过$\mathbf G_k\mathbf u$随强度放大，其余由负荷独立小波动引起的变化保持高斯。当$r=d$时，式(B4)—(B5)退化为标准多元TMM；当$\nu_k\to\infty$时，$S_k\equiv1$，退化为GMM。分量内$\mathbf u$的边缘分布为多元t分布$t_{\nu_k}(\mathbf m_k,\boldsymbol\Sigma_k)$，整体向量$(\mathbf u,\mathbf w)$不再是标准多元t分布。

每个分量的独立参数个数为

$$
p_k=r+\tfrac{r(r+1)}{2}+1+(d-r)+(d-r)r+\tfrac{(d-r)(d-r+1)}{2},
\tag{B6}
$$

模型总参数为$K-1+\sum_k p_k$，与同$K$标准多元TMM相同，均比同$K$的GMM多$K$个。

### B. 由潮流灵敏度确定作用范围

共同波动强度的作用范围取决于两类信息：哪些注入功率由同一天气过程驱动，以及这些注入的变化如何改变节点电压。前者来自源荷建模，后者来自网络在运行点附近的灵敏度。

记非平衡节点注入功率向量为$\mathbf p=[\mathbf P_{\mathcal N}^\top,\mathbf Q_{\mathcal N}^\top]^\top$，天气驱动注入集合为$\mathcal W$，本算例中为8个光伏站的有功注入，$n_w=|\mathcal W|$。在运行点$\mathbf x^{(0)}$附近，

$$
\Delta\mathbf x\approx\mathbf J_0^{-1}\Delta\mathbf p,\qquad
\mathbf J_0=\left.\frac{\partial\mathbf p}{\partial\mathbf x}\right|_{\mathbf x^{(0)}},
\tag{B7}
$$

$\mathbf J_0$为直角坐标交流潮流雅可比矩阵。天气驱动注入的变化为$\Delta\mathbf p=\mathbf L_{\mathcal W}\Delta\mathbf p_{\mathcal W}$，$\mathbf L_{\mathcal W}\in\{0,1\}^{d\times n_w}$为对应选择矩阵。天气驱动的电压变化方向张成

$$
\mathbf H=\mathbf J_0^{-1}\mathbf L_{\mathcal W}\in\mathbb R^{d\times n_w},
\tag{B8}
$$

即潮流雅可比逆矩阵中各光伏节点有功对应的列。映射到拟合坐标后作奇异值分解：

$$
\tilde{\mathbf H}=\mathbf W\mathbf H=\mathbf U_H\boldsymbol\Lambda_H\mathbf R_H^\top,\qquad
r=\#\{\lambda_{H,j}>\epsilon_H\lambda_{H,1}\},\qquad
\mathbf Q_u=\mathbf U_H(:,1{:}r),
\tag{B9}
$$

$\lambda_{H,j}$为奇异值，$\epsilon_H$为秩判别相对阈值，本算例取$10^{-10}$；$\mathbf Q_w$取$\mathbf I-\mathbf Q_u\mathbf Q_u^\top$的前$d-r$个左奇异向量。由于$\mathrm{rank}(\mathbf H)\le n_w$，共同强度作用的电压方向数不超过同一天气驱动的注入个数。

运行点$\mathbf x^{(0)}$取PtoV映射的第一个运行点，即离源荷样本平均注入最近的样本（`map_samples_with_atlas.m`），也可直接取预测出力对应的运行点。PtoV映射在该点已求解一次交流潮流并分解$\mathbf J_0$（`build_anchor.m`），式(B8)只取已有逆矩阵中的$n_w$列。额外计算仅为式(B9)的一次$d\times n_w$奇异值分解，复杂度为$O(dn_w^2)$，不需要逐样本潮流或额外交流潮流计算。

### C. 运行点灵敏度不能由额定电压近似替代

额定电压平启动下，$\Delta\mathbf I\approx\Delta\mathbf P-\mathrm j\Delta\mathbf Q$，灵敏度方向可近似取去除平衡节点后节点阻抗矩阵$\mathbf Z=\mathbf Y_{\mathcal N\mathcal N}^{-1}$的对应列，记为$\mathbf H^{Z}$。设$\mathbf H^{Z}=\mathbf H+\boldsymbol\Delta$，则在拟合坐标中

$$
\|\mathbf W\boldsymbol\Delta\|_2\le\frac{\|\boldsymbol\Delta\|_2}{\sigma_d},
\tag{B10}
$$

$\sigma_d$为训练电压最弱特征方向的标准差。配电网中负荷小波动方向的$\sigma_d$可比光伏驱动方向小5—6个数量级，原始坐标中百分之几的方向偏差会在拟合坐标中被放大到主导地位，使天气驱动的变化泄漏到式(B5)的高斯块。因此作用范围须在实际运行点附近确定，不能用额定电压下的网络近似代替。

### D. 训练数据核对

$r$由式(B9)在拟合前确定，训练数据只用于核对，不用于改变$r$。若$\mathbf u$确实受共同强度放大，其各坐标应呈正超额峰度，且不同坐标幅值平方相互正相关：

$$
\kappa_j=\frac{\mathrm E[u_j^4]}{(\mathrm E[u_j^2])^2}-3>0,\qquad
\rho_{ij}=\mathrm{corr}(u_i^2,u_j^2)>0 .
\tag{B11}
$$

实际系统中事件强度不可观测，$\rho_{ij}$只需电压样本即可计算；合成算例中还可直接计算$u_j^2$与已知强度平方的相关系数。$\mathbf w$的坐标应近似满足$\kappa_j\approx0$。

### E. 参数估计

采用期望条件最大化。第$m$次迭代中，样本$i$对分量$k$的后验概率和天气驱动块的条件强度权重为

$$
\gamma_{ik}=\frac{\pi_k f_t(\mathbf u_i;\mathbf m_k,\boldsymbol\Sigma_k,\nu_k)\,\phi(\mathbf w_i;\mathbf a_k+\mathbf G_k\mathbf u_i,\boldsymbol\Omega_k)}
{\sum_l\pi_l f_t(\mathbf u_i;\mathbf m_l,\boldsymbol\Sigma_l,\nu_l)\,\phi(\mathbf w_i;\mathbf a_l+\mathbf G_l\mathbf u_i,\boldsymbol\Omega_l)},
\qquad
\tau_{ik}=\frac{\nu_k+r}{\nu_k+\delta_{ik}},
\tag{B12}
$$

$f_t$为$r$维多元t密度，$\phi$为高斯密度，$\delta_{ik}=(\mathbf u_i-\mathbf m_k)^\top\boldsymbol\Sigma_k^{-1}(\mathbf u_i-\mathbf m_k)$。记$N_k=\sum_i\gamma_{ik}$，条件最大化步为

$$
\pi_k=\frac{N_k}{N},\quad
\mathbf m_k=\frac{\sum_i\gamma_{ik}\tau_{ik}\mathbf u_i}{\sum_i\gamma_{ik}\tau_{ik}},\quad
\boldsymbol\Sigma_k=\frac{1}{N_k}\sum_i\gamma_{ik}\tau_{ik}(\mathbf u_i-\mathbf m_k)(\mathbf u_i-\mathbf m_k)^\top,
\tag{B13}
$$

$$
[\mathbf a_k,\ \mathbf G_k]=\arg\min\sum_i\gamma_{ik}\left\|\mathbf w_i-\mathbf a_k-\mathbf G_k\mathbf u_i\right\|_2^2,\qquad
\boldsymbol\Omega_k=\frac{1}{N_k}\sum_i\gamma_{ik}\mathbf e_{ik}\mathbf e_{ik}^\top,
\tag{B14}
$$

$\mathbf e_{ik}=\mathbf w_i-\mathbf a_k-\mathbf G_k\mathbf u_i$。自由度由下式求根：

$$
\ln\frac{\nu_k}{2}-\psi\!\left(\frac{\nu_k}{2}\right)+1+\frac{1}{N_k}\sum_i\gamma_{ik}\left[\psi\!\left(\frac{\nu_k+r}{2}\right)-\ln\frac{\nu_k+\delta_{ik}}{2}-\tau_{ik}\right]=0,
\tag{B15}
$$

$\psi$为双伽马函数，根限定在$[4.1,10^5]$。各更新均为闭式或一维求根；散布与残差协方差特征值下限为$10^{-7}$，与标准TMM设置一致。

### F. 由电压到电气量

给定$C=k$、$S_k=s$，由式(B4)—(B5)，$\boldsymbol\xi$服从高斯分布，协方差为

$$
\mathrm{Cov}(\boldsymbol\xi\mid C=k,S_k=s)=\mathbf Q\begin{bmatrix}s\boldsymbol\Sigma_k & s\boldsymbol\Sigma_k\mathbf G_k^\top\\ s\mathbf G_k\boldsymbol\Sigma_k & s\mathbf G_k\boldsymbol\Sigma_k\mathbf G_k^\top+\boldsymbol\Omega_k\end{bmatrix}\mathbf Q^\top .
\tag{B16}
$$

节点电压幅值平方、线路有功和无功及网损均为$\mathbf x$的二次型，因此给定强度后可沿用高斯电压二次型的条件特征函数，再对$S_k$作一维Gauss–Laguerre求积。与标准TMM不同，式(B16)的协方差不与$s$成比例，条件二次型核矩阵的特征分解须在每个求积强度节点分别计算。当前数值比较采用分量分层随机化Sobol抽样，按式(B4)—(B5)生成电压后由原交流功率关系计算电气量。

### G. 多块TMM：多个独立天气过程

当光伏站分属$G$个相互独立的天气过程（如相距较远的云区）时，同一分量内应有$G$个独立的波动强度。记第$g$个天气过程驱动的注入集合为$\mathcal W_g$，灵敏度为$\mathbf H_g=\mathbf J_0^{-1}\mathbf L_{\mathcal W_g}$。依次正交化得到各组方向：

$$
\mathbf Q_1=\mathrm{orth}(\mathbf W\mathbf H_1),\qquad
\mathbf Q_g=\mathrm{orth}\!\Big(\big(\mathbf I-\textstyle\sum_{l<g}\mathbf Q_l\mathbf Q_l^\top\big)\mathbf W\mathbf H_g\Big),\quad g\ge2,
\tag{B17}
$$

$\mathbf Q_w$为其余方向。记$\mathbf u_g=\mathbf Q_g^\top\boldsymbol\xi$，$r_g$为其维数。以$G=2$为例，第$k$个分量写为

$$
\mathbf u_1\sim t_{\nu_{1k}}(\mathbf m_{1k},\boldsymbol\Sigma_{1k}),\qquad
\mathbf u_2\mid\mathbf u_1\sim t_{\nu_{2k}}(\mathbf m_{2k}+\mathbf F_k\mathbf u_1,\ \boldsymbol\Sigma_{2k}),\qquad
\mathbf w\mid\mathbf u_1,\mathbf u_2\sim\mathcal N\!\big(\mathbf a_k+\mathbf G_k[\mathbf u_1^\top,\mathbf u_2^\top]^\top,\ \boldsymbol\Omega_k\big),
\tag{B18}
$$

其中$\mathbf u_2\mid\mathbf u_1$的多元t分布按其自身的强度$S_{2k}\sim\mathrm{Inv\text{-}Gamma}(\nu_{2k}/2,\nu_{2k}/2)$生成，与$\mathbf u_1$的强度$S_{1k}$相互独立；$\mathbf F_k$表示两组电压变化之间由网络和负荷形成的线性联系。$G=1$时退化为式(B4)—(B5)。

参数估计中，$\mathbf u_1$按式(B13)、(B15)更新；$\mathbf u_2$为带强度权重$\tau^{(2)}_{ik}=(\nu_{2k}+r_2)/(\nu_{2k}+\delta^{(2)}_{ik})$的加权最小二乘，$\delta^{(2)}_{ik}$为其残差马氏距离平方，自由度按式(B15)将$r$、$\delta$换为$r_2$、$\delta^{(2)}$求根；$\mathbf w$按式(B14)对$[\mathbf u_1,\mathbf u_2]$回归。给定$S_{1k}=s_1$、$S_{2k}=s_2$，$(\mathbf u_1,\mathbf u_2,\mathbf w)$为高斯分布，

$$
\mathrm{Cov}(\mathbf u_1)=s_1\boldsymbol\Sigma_{1k},\quad
\mathrm{Cov}(\mathbf u_2)=s_1\mathbf F_k\boldsymbol\Sigma_{1k}\mathbf F_k^\top+s_2\boldsymbol\Sigma_{2k},\quad
\mathrm{Cov}(\mathbf u_2,\mathbf u_1)=s_1\mathbf F_k\boldsymbol\Sigma_{1k},
\tag{B19}
$$

$\mathbf w$的协方差为$\mathbf G_k\mathrm{Cov}([\mathbf u_1;\mathbf u_2])\mathbf G_k^\top+\boldsymbol\Omega_k$。电气量的条件特征函数沿用高斯二次型闭式，对$(s_1,s_2)$作二维广义Gauss–Laguerre张量积求积。状态：推导完成，代码与算例见方案第15节C。

### H. 本算例数值

IEEE 33节点，$d=64$，$n_w=8$，由式(B9)得$r=8$。$\mathbf u$方向承担训练电压98.8%的方差，各坐标超额峰度1.4—2.0，幅值平方与事件强度平方的相关系数约0.4；$\mathbf w$方向超额峰度中位数0.09，相关系数中位数0.075。所选模型自由度4.9—9.7，标准TMM为30—200。

按式(B10)核对额定电压近似：原始坐标中$\mathbf H^{Z}$与$\mathbf H$的主夹角余弦为0.985—0.99999，列方向相对偏差0.6%—8.5%；训练电压最强与最弱方向标准差之比约$7\times10^{5}$，拟合坐标中主夹角余弦降至0.07—0.61，$\mathbf H^{Z}$方向只承担1.8%的方差。该结果已由数值实验支持，说明本算例中必须使用运行点雅可比。

### I. 各分量共用其余方向的残差协方差

式(B5)中每个分量各有一个$(d-r)\times(d-r)$残差协方差$\boldsymbol\Omega_k$，$d=64$、$r=3$时为1891个参数，占每分量参数的约88%。在8760小时量级的固定样本下，低权重分量只分到数百个电压样本，$\boldsymbol\Omega_k$估计误差大，而尾部恰由这些分量决定（风电最小测试第7节）。

物理依据：给定天气驱动块$\mathbf u$后，其余方向的剩余变化来自负荷的小幅独立波动，与天气处于哪个分量无关；分量之间的差别由截距$\mathbf a_k$和回归系数$\mathbf G_k$表示，二者也承担交流潮流二次关系的分段线性近似。因此令

$$
\mathbf w\mid \mathbf u,C=k\sim\mathcal N(\mathbf a_k+\mathbf G_k\mathbf u,\ \boldsymbol\Omega),
\tag{B20}
$$

$\boldsymbol\Omega$为各分量共用的残差协方差，其余与式(B4)相同。条件最大化步中，$\pi_k$、$\mathbf m_k$、$\boldsymbol\Sigma_k$、$\nu_k$及$[\mathbf a_k,\mathbf G_k]$仍按式(B13)—(B15)更新。由于各输出坐标共用同一设计矩阵$[1,\mathbf u^\top]$，共用$\boldsymbol\Omega$不改变$[\mathbf a_k,\mathbf G_k]$的加权最小二乘解；残差协方差改为

$$
\boldsymbol\Omega=\frac{1}{N}\sum_k\sum_i\gamma_{ik}\mathbf e_{ik}\mathbf e_{ik}^\top .
\tag{B21}
$$

总参数为

$$
K-1+K\Big[r+\tfrac{r(r+1)}{2}+1+(d-r)+(d-r)r\Big]+\tfrac{(d-r)(d-r+1)}{2},
\tag{B22}
$$

本算例每增加一个分量增加254个参数，标准GMM为2144个。令$\nu_k\to\infty$得到结构相同的高斯对照，用于区分共用结构与t分量各自的作用。式(B16)中$\boldsymbol\Omega_k$换为$\boldsymbol\Omega$即可沿用由电压到电气量的计算。状态：推导完成，代码与算例见风电最小测试方案第7节。

### J. 天气驱动注入与分块的确定规则（2026-10-22，工程规则）

第B节的式(B8)要求已知天气驱动注入集合$\mathcal W$。$\mathcal W$按节点注入功率定义，不按设备类型定义：光伏、风电或受气温影响的负荷，只要其预测偏差被同一天气过程共同放大，就属于$\mathcal W$。确定规则只用注入功率场景、功率到电压计算得到的电压和预测运行点的一次潮流，不需要额外潮流计算。

1. 共同放大判别。对任意两个注入$i,j$的偏差，计算
$$
c_{ij}=\mathrm{corr}(e_i^2,e_j^2)-\mathrm{corr}(e_i,e_j)^2 .
\tag{B23}
$$
两者联合正态时$c_{ij}\approx0$；被同一强度共同放大时，大偏差同时出现，$c_{ij}>0$。阈值取相同协方差、相同场景数的正态场景中$c_{ij}$的最大值。$c_{ij}$超过阈值的注入相连，连通的注入构成一个天气过程的$\mathcal W_g$，每组至少两个注入。判别使用全部可得的注入场景；输入分布已知时可多取场景，不增加潮流计算。
2. 电压方向。各组按式(B8)—(B9)取$\mathbf H_g=\mathbf J_0^{-1}\mathbf L_{\mathcal W_g}$，多组按第G节依次正交化。
3. 补漏。偏差较大时，潮流二次响应等使一部分天气放大的变化落在一阶方向之外。对其余电压方向作四阶矩检验，同时满足以下三条的方向并入被放大的一块：四阶统计量超出相同场景数正态情形的范围；承担至少$10^{-3}$的电压方差；其平方与被放大一块的平方和相关，相关系数超出正态情形的范围。第三条排除出力限值等与天气强度无关的非正态。
4. 退化情形。无任何组时退化为GMM；全部注入同属一组时退化为标准多元TMM。

检验（`tests/python/block_split_rule_check.py`，结果`result/case33bw/block_split_rule_check_20261022/`）：用全部45056个注入场景，6个已有33节点场景的分组均与输入设定一致，包括两片云区的6+2分组和天气强度恒定情形的无分组；只用8192个场景时，两片云区漏判一个光伏站，天气强度恒定情形误判一对注入，均因信号接近阈值。另构造8个负荷节点由同一天气强度放大、光伏只有2%独立误差的场景，判出的$\mathcal W$恰为这8个节点的有功和无功。单片云区光伏场景补漏后，其余方向超额峰度最大值由5—25降至0.7—2.8，中位数0.02—0.06；NREL输入中最大泄漏方向95.7%位于由同一锚点潮流计算的潮流二次响应方向张成的空间内。状态：分块规则已由数值实验支持；按新分块拟合后的电气量精度尚未验证。

补充检验（`tests/python/block_split_rule_scenarios.py`，结果`split_rule_scenarios.json`）：负荷受天气驱动（8个负荷节点有功、无功共16个注入判为一组）、光伏与负荷属同一天气过程（24个注入一组）、光伏与负荷属两个独立天气过程（负荷16个、光伏8个两组）、光伏与负荷波动都大但无共同放大（无分组）四类构造场景均判对。AQGMM前作33节点冻结输入判为无分组：32个节点负荷近似同比例变化（节点各自波动只占方差2.7%），共同负荷倍数由常规模式（85%，超额峰度−0.54）和高负荷模式（15%，−0.88）组成，各模式内不厚尾，整体正峰度来自两模式分离，属多峰而非强度放大，退化为GMM与其输入特征一致。

局限：注入近似同比例变化时，(B23)只反映一个共同变量，不能判别该共同变量本身是否厚尾。补充做法为对近似同比例变化的共同方向单独检查各运行模式内是否厚尾；误把不厚尾方向放入被放大一块时自由度估计趋大而自动接近高斯，误漏厚尾方向则损失尾部精度，因此该检查宜从宽。【尚未实现】

## 1. 功率到电压

固定拓扑、一个平衡母线、其余 PQ 母线，指定高电压解支，满足
\(s=V\odot\overline{YV}\)。输入净注入为发电减负荷，单位 p.u.。
直角坐标为 \(x=[\Re V_{PQ},\Im V_{PQ}]\)。沿用 PFIGMM 的锚点与二次残差修正：

\[
h_1=J_a^{-1}(s-s_a),\quad
h_2=-J_a^{-1}[F(x_a+h_1)-s],\quad
\hat x=x_a+h_1+h_2+J_a^{-1}[s-F(x_a+h_1+h_2)].
\]

复用代码：`+BTMM_core/+inverse/`、`+BTMM_core/+powerflow/build_anchor.m`。
局部逆初值计算结果作为固定雅可比修正 Newton 迭代的初值：

\[
x^{(k+1)}=x^{(k)}+\alpha_kJ_a^{-1}[s-F(x^{(k)})],\qquad
\eta_k=\|J_a^{-1}\|_2\|s-F(x^{(k)})\|_2.
\]

交流功率关于直角坐标电压的二次关系精确，初值计算方法的有限次逆修正保留截断误差。迭代导数为 \(I-J_a^{-1}J(x)\)，在锚点处为零；收敛适用于锚点周围的合适局部邻域。每个锚点复用已有逆矩阵，按列批量处理尚未达标样本。默认最多追加 10 轮；验收保持 \(\eta\le10^{-7}\)，并增加逐节点功率残差无穷范数不超过 \(10^{-8}\) p.u. 及原电压范围检查。步长初值为 1，残差增长或试探电压越界时逐列减半，最多回溯 8 次。步长回溯后仍无残差下降的样本停止使用当前锚点，转入后续锚点。

锚点仍按未达标样本的最大残差自适应选取，上限 12。锚点样本也计算实际功率残差。所有可用锚点的追加修正仍未通过时，才逐样本调用精确 AC 潮流。记录实际修正量 \(\|J_a^{-1}[s-F(x)]\|_2\)，验收仍采用原残差换算量；二者均不单独作为严格的非线性电压误差界。AC 参照电压只用于离线误差和指定高电压解支核查，不参与运行中的停止判断。当前不支持非平衡 PV 母线或运行模式切换。

## 4. 电压到线路功率

采用 MATPOWER 的完整 `Yf`、`Yt`，包含变压器、线路充电和投运状态：

\[
S_f=S_b V_f\odot\overline{Y_f V},\qquad
S_t=S_b V_t\odot\overline{Y_t V}.
\]

两端均以流入线路为正，反向功率包含损耗，不能直接取负。总线路有功损耗为 \(\sum_l\Re(S_{f,l}+S_{t,l})\)。模型抽样经此精确公式传播；本阶段尚未实现 T 混合二次型 CDF 的确定性积分反演。

## 6. 非抽样的矩核查

`physical_moments.m` 将电压均值、协方差和网损二次型直接按混合分量计算。对于分量协方差 C、自由度 ν>4，以及二次型 \(q=z^TAz+b^Tz+c\)：

\[
E[q]=\mu^TA\mu+b^T\mu+c+\operatorname{tr}(AC),
\]
\[
\operatorname{Var}(q)=(2A\mu+b)^TC(2A\mu+b)
+\frac{2(\nu-2)}{\nu-4}\operatorname{tr}[(AC)^2]
+\frac{2}{\nu-4}[\operatorname{tr}(AC)]^2.
\]

Gaussian 极限令后两项系数为 2、0。混合方差包含分量间均值差。ν≤4 时一般网损方差标为 Inf。该核用于区别模型自身的矩失配与 Monte Carlo 波动。
