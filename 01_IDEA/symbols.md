# Symbols

论文符号以`02_PAPER/manuscript/method_zh.md`附录C为准。本表补充代码变量对应关系；新增或修改符号时同步更新`method_zh.md`、`derivation.md`、`BTMM_note.md`和`BTMM_changelog.md`。

| 符号 | 含义 | 维度/单位 | 代码变量（`tests/python/`） |
| --- | --- | --- | --- |
| $\mathbf p$ | 非平衡节点净注入有功、无功 | $d$，p.u. | `target`（$P_2\ldots P_{33}$，$Q_2\ldots Q_{33}$） |
| $\mathbf x$ | 非平衡节点电压实部、虚部 | $d=2(n-1)$，p.u. | `states`；快速映射输出`state`（全节点，`fast_voltage_map.fast_map`返回复电压） |
| $\bar{\mathbf x}$, $\mathbf E$, $\mathbf D$ | 电压均值、协方差特征向量、各方向标准差 | | `center`, `basis`, `scale` |
| $\boldsymbol\xi$ | 按各方向标准差归一化的电压坐标 | $d$ | `x`（估计参数用）、`xv`（选择分量数用） |
| $\mathbf J_a$, $\mathbf J_0$ | 锚点、平均注入处潮流雅可比 | $d\times d$ | MATLAB `anchor.jacobianPolar`；Python `rect_jacobian(v0, Y)` |
| $q$ | 节点电压幅值、支路功率、网损 | p.u.、MW、MVAr | `base.outputs(...)`，对象表`base.OBJECTS` |
| $K$, $\pi_k$, $\gamma_{ik}$ | 分量数、权重、AQGMM分量归属概率 | | `k`, `weight`, `resp` |
| $\nu_k$, $b_k$ | 标准TMM自由度、多元峰度比 | | `df`；`nu_from_kurtosis` |
| $c_{ij}$, $c^*$ | 注入共同放大量及阈值 | | `coamplification`；`null` |
| $\mathcal W_g$ | 第$g$个天气过程放大的注入 | | `split(...)['groups']` |
| $\mathbf Q_u$, $\mathbf Q_w$ | 天气驱动方向、其余方向 | $d\times r$，$d\times(d-r)$ | `Q1`, `Q2` |
| $r_g$, $g(j)$ | 组维数、方向所属组 | | `group_cols`；`gid` |
| $s_{\max,g}$ | 第$g$组强度上限 | | `strength_cap(...)`，`s_max` |
| $U$, $S_g$, $\nu_{g,k}$, $b_{g,k}$ | 天气形势强弱、组强度、组自由度、组峰度比 | | `uniform`；`TruncatedStrength.draw`；`nu`；`m` |
| $\tilde{\boldsymbol\Sigma}_k$ | 散布矩阵（按$\mathrm E[\sqrt{S_gS_h}]$折算） | $r\times r$ | `scatter`；`mean_pair` |
| $\mathbf a_k$, $\mathbf G_k$, $\boldsymbol\Omega_k$ | 其余方向截距、回归系数、残差协方差 | | `intercept`, `coefficient`, `residual` |
