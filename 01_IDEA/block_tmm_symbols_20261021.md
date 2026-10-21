# 分块TMM符号

## 2026-10-21 分块TMM符号（derivation.md式(B1)—(B16)）

| 符号 | 含义 | 单位/维数 | 代码字段 |
|---|---|---|---|
| x、ξ | 非平衡节点直角坐标电压、白化后的拟合坐标 | p.u.；d=2(n−1)=64 | voltage实虚部；fitting_inputs.npz的x |
| W、E、D、σ_j | 白化矩阵、训练电压协方差特征向量、各方向标准差 | 无量纲/p.u. | basis、scale |
| Q_u、Q_w、r | 天气驱动方向、其正交补、驱动方向数 | d×r、d×(d−r)；本算例r=8 | block_subspace.npz的Q1、Q2 |
| u、w | 天气驱动块坐标、其余块坐标 | r维、d−r维 | x@Q1、x@Q2 |
| S_k、ν_k | 第k分量共同波动强度及其自由度 | 无量纲；ν_k∈[4.1,10^5] | df |
| m_k、Σ_k | 天气驱动块位置与散布 | r维、r×r | means、scatter |
| a_k、G_k、Ω_k | 其余块截距、对u的回归系数、残差协方差 | (d−r)维、(d−r)×r、(d−r)×(d−r) | intercept、coefficient、residual |
| p、𝒲、n_w、L_𝒲 | 非平衡节点注入、天气驱动注入集合、其个数、选择矩阵 | p.u.；本算例n_w=8为光伏有功 | SITES |
| J_0、H、H^Z | 运行点直角坐标潮流雅可比、天气驱动灵敏度、额定电压阻抗近似 | d×d、d×n_w | rect_jacobian、sensitivity |
| γ_ik、τ_ik、δ_ik | 分量后验概率、强度权重、天气驱动块马氏距离平方 | 无量纲 | fit_block内部量 |
| κ_j、ρ_ij | 坐标超额峰度、幅值平方相关系数 | 无量纲 | block_subspace_diagnostics.json |

分块模型族在结果表中记为BTMM。G_k与方案中“回归系数”对应，避免与分量标签C混用。
