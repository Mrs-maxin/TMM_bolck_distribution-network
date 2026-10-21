# 分块TMM推导

## 2026-10-21 分块TMM中共同波动强度作用范围的确定

本节对应[固定预测时刻方案](pv_fixed_forecast_common_scale_plan_20261021.md)第12节及证据E-PV-BLOCK-TMM-20261021。模型仍沿用TMM的基本结构：每个混合分量一个自由度和一个随机波动强度，给定强度后电压服从高斯分布。新增内容是确定该强度作用于哪些电压变化方向。状态：模型与确定方法已由代码实现支持，IEEE 33节点单一光伏场景已由数值实验支持，其他场景尚未验证。

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
