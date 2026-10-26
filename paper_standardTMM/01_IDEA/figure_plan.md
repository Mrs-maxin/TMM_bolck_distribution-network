# Figure plan（标准TMM论文）

## 重尾分布注入算例（claimId S1）

在IEEE 33节点、8个电源有功服从重尾或长尾分布的条件下，AQGMM的高斯分量使节点电压、支路功率和网损的尾段概率偏低；标准TMM以t分量替换高斯分量，与AQGMM在同一K下对照MC方法，显示0.1%至5%尾段累积概率误差和极端分位误差的差别。

| 序号 | 读者问题 | 工程对象 | 图型 | 数据 | 状态 |
|---|---|---|---|---|---|
| 1 | 重尾注入下哪些电气量的尾段明显延伸 | 节点电压、支路功率、网损 | MC分布与尾部放大 | `result/heavytail_injection/case33bw_20261026/tail_screening/` | 【待补充】 |
| 2 | 同一K下两种模型的尾段CDF与MC的差别 | 典型节点电压、支路功率 | 尾段CDF对数坐标叠加 | `results/`中逐阈值明细（大文件在`shared/`） | 【待补充】 |
| 3 | 不同K下标准TMM较优的项数和误差分布 | 373项尾部延伸电气量 | 分组柱或箱线图 | `tail_screening/tail_cdf_rmse_method_compare.csv` | 【待补充】 |

现有诊断图（`shared/result/heavytail_injection/case33bw_20261026/tail_distribution/figures/`）为Python图并含分组t，本文正式图需用两方法数据经MATLAB `save_figure`重新导出。

## 实际出力注入算例（claimId S1、S2）

【待补充】计算完成后补充。
