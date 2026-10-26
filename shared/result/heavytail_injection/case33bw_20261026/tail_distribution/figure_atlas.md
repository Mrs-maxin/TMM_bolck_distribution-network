# PDF图册

PDF指概率密度函数。下表同时提供PNG预览和可下载的PDF图文件。每个方法比较图包含K=1、2、4、6；尾部图不重新归一化。全部图和数据保持一一对应。

## Pareto

| 电气量 | 尾侧 | 当前MC尾段判断 | 全分布PDF | 尾部PDF放大 |
|---|---|---|---|---|
| 节点22电压 | 上尾 | 当前尾段重尾特征较强 | [PNG](figures/pareto/V22_full_pdf.png) / [PDF](figures/pareto/V22_full_pdf.pdf) | [PNG](figures/pareto/V22_tail_pdf_zoom.png) / [PDF](figures/pareto/V22_tail_pdf_zoom.pdf) |
| 节点21电压 | 上尾 | 当前尾段重尾特征较强 | [PNG](figures/pareto/V21_full_pdf.png) / [PDF](figures/pareto/V21_full_pdf.pdf) | [PNG](figures/pareto/V21_tail_pdf_zoom.png) / [PDF](figures/pareto/V21_tail_pdf_zoom.pdf) |
| 节点20电压 | 上尾 | 当前尾段重尾特征较强 | [PNG](figures/pareto/V20_full_pdf.png) / [PDF](figures/pareto/V20_full_pdf.pdf) | [PNG](figures/pareto/V20_tail_pdf_zoom.png) / [PDF](figures/pareto/V20_tail_pdf_zoom.pdf) |
| 支路32有功（32→33） | 下尾 | 当前尾段重尾特征较强 | [PNG](figures/pareto/P32_full_pdf.png) / [PDF](figures/pareto/P32_full_pdf.pdf) | [PNG](figures/pareto/P32_tail_pdf_zoom.png) / [PDF](figures/pareto/P32_tail_pdf_zoom.pdf) |
| 支路17有功（17→18） | 下尾 | 当前尾段重尾特征较强 | [PNG](figures/pareto/P17_full_pdf.png) / [PDF](figures/pareto/P17_full_pdf.pdf) | [PNG](figures/pareto/P17_tail_pdf_zoom.png) / [PDF](figures/pareto/P17_tail_pdf_zoom.pdf) |
| 支路31有功（31→32） | 下尾 | 当前尾段重尾特征较强 | [PNG](figures/pareto/P31_full_pdf.png) / [PDF](figures/pareto/P31_full_pdf.pdf) | [PNG](figures/pareto/P31_tail_pdf_zoom.png) / [PDF](figures/pareto/P31_tail_pdf_zoom.pdf) |
| 支路18无功（2→19） | 上尾 | 当前尾段重尾特征较强 | [PNG](figures/pareto/Q18_full_pdf.png) / [PDF](figures/pareto/Q18_full_pdf.pdf) | [PNG](figures/pareto/Q18_tail_pdf_zoom.png) / [PDF](figures/pareto/Q18_tail_pdf_zoom.pdf) |
| 支路30无功（30→31） | 上尾 | 当前尾段重尾特征较强 | [PNG](figures/pareto/Q30_full_pdf.png) / [PDF](figures/pareto/Q30_full_pdf.pdf) | [PNG](figures/pareto/Q30_tail_pdf_zoom.png) / [PDF](figures/pareto/Q30_tail_pdf_zoom.pdf) |
| 支路19无功（19→20） | 上尾 | 当前尾段重尾特征较强 | [PNG](figures/pareto/Q19_full_pdf.png) / [PDF](figures/pareto/Q19_full_pdf.pdf) | [PNG](figures/pareto/Q19_tail_pdf_zoom.png) / [PDF](figures/pareto/Q19_tail_pdf_zoom.pdf) |

全部节点/线路与MC形态比较：

- 节点电压：[全部32项尾部特征](figures/pareto/V_all_quantities_tail_compare.png)；[标准化PDF](figures/pareto/V_mc_standardized_pdf_compare.png)；[标准化尾部PDF](figures/pareto/V_mc_standardized_tail_pdf_compare.png)。
- 支路有功：[全部32项尾部特征](figures/pareto/P_all_quantities_tail_compare.png)；[标准化PDF](figures/pareto/P_mc_standardized_pdf_compare.png)；[标准化尾部PDF](figures/pareto/P_mc_standardized_tail_pdf_compare.png)。
- 支路无功：[全部32项尾部特征](figures/pareto/Q_all_quantities_tail_compare.png)；[标准化PDF](figures/pareto/Q_mc_standardized_pdf_compare.png)；[标准化尾部PDF](figures/pareto/Q_mc_standardized_tail_pdf_compare.png)。

## 对数正态

| 电气量 | 尾侧 | 当前MC尾段判断 | 全分布PDF | 尾部PDF放大 |
|---|---|---|---|---|
| 节点22电压 | 上尾 | 尾部延伸明显，重尾形状证据不足 | [PNG](figures/lognormal/V22_full_pdf.png) / [PDF](figures/lognormal/V22_full_pdf.pdf) | [PNG](figures/lognormal/V22_tail_pdf_zoom.png) / [PDF](figures/lognormal/V22_tail_pdf_zoom.pdf) |
| 节点21电压 | 上尾 | 尾部延伸明显，重尾形状证据不足 | [PNG](figures/lognormal/V21_full_pdf.png) / [PDF](figures/lognormal/V21_full_pdf.pdf) | [PNG](figures/lognormal/V21_tail_pdf_zoom.png) / [PDF](figures/lognormal/V21_tail_pdf_zoom.pdf) |
| 节点20电压 | 上尾 | 尾部延伸明显，重尾形状证据不足 | [PNG](figures/lognormal/V20_full_pdf.png) / [PDF](figures/lognormal/V20_full_pdf.pdf) | [PNG](figures/lognormal/V20_tail_pdf_zoom.png) / [PDF](figures/lognormal/V20_tail_pdf_zoom.pdf) |
| 支路32有功（32→33） | 下尾 | 尾部延伸明显，重尾形状证据不足 | [PNG](figures/lognormal/P32_full_pdf.png) / [PDF](figures/lognormal/P32_full_pdf.pdf) | [PNG](figures/lognormal/P32_tail_pdf_zoom.png) / [PDF](figures/lognormal/P32_tail_pdf_zoom.pdf) |
| 支路31有功（31→32） | 下尾 | 尾部延伸明显，重尾形状证据不足 | [PNG](figures/lognormal/P31_full_pdf.png) / [PDF](figures/lognormal/P31_full_pdf.pdf) | [PNG](figures/lognormal/P31_tail_pdf_zoom.png) / [PDF](figures/lognormal/P31_tail_pdf_zoom.pdf) |
| 支路21有功（21→22） | 下尾 | 尾部延伸明显，重尾形状证据不足 | [PNG](figures/lognormal/P21_full_pdf.png) / [PDF](figures/lognormal/P21_full_pdf.pdf) | [PNG](figures/lognormal/P21_tail_pdf_zoom.png) / [PDF](figures/lognormal/P21_tail_pdf_zoom.pdf) |
| 支路18无功（2→19） | 上尾 | 当前尾段重尾特征较强 | [PNG](figures/lognormal/Q18_full_pdf.png) / [PDF](figures/lognormal/Q18_full_pdf.pdf) | [PNG](figures/lognormal/Q18_tail_pdf_zoom.png) / [PDF](figures/lognormal/Q18_tail_pdf_zoom.pdf) |
| 支路19无功（19→20） | 上尾 | 当前尾段重尾特征较强 | [PNG](figures/lognormal/Q19_full_pdf.png) / [PDF](figures/lognormal/Q19_full_pdf.pdf) | [PNG](figures/lognormal/Q19_tail_pdf_zoom.png) / [PDF](figures/lognormal/Q19_tail_pdf_zoom.pdf) |
| 支路31无功（31→32） | 上尾 | 当前尾段重尾特征较强 | [PNG](figures/lognormal/Q31_full_pdf.png) / [PDF](figures/lognormal/Q31_full_pdf.pdf) | [PNG](figures/lognormal/Q31_tail_pdf_zoom.png) / [PDF](figures/lognormal/Q31_tail_pdf_zoom.pdf) |

全部节点/线路与MC形态比较：

- 节点电压：[全部32项尾部特征](figures/lognormal/V_all_quantities_tail_compare.png)；[标准化PDF](figures/lognormal/V_mc_standardized_pdf_compare.png)；[标准化尾部PDF](figures/lognormal/V_mc_standardized_tail_pdf_compare.png)。
- 支路有功：[全部32项尾部特征](figures/lognormal/P_all_quantities_tail_compare.png)；[标准化PDF](figures/lognormal/P_mc_standardized_pdf_compare.png)；[标准化尾部PDF](figures/lognormal/P_mc_standardized_tail_pdf_compare.png)。
- 支路无功：[全部32项尾部特征](figures/lognormal/Q_all_quantities_tail_compare.png)；[标准化PDF](figures/lognormal/Q_mc_standardized_pdf_compare.png)；[标准化尾部PDF](figures/lognormal/Q_mc_standardized_tail_pdf_compare.png)。

## 对数逻辑斯蒂

| 电气量 | 尾侧 | 当前MC尾段判断 | 全分布PDF | 尾部PDF放大 |
|---|---|---|---|---|
| 节点22电压 | 上尾 | 当前尾段重尾特征较强 | [PNG](figures/loglogistic/V22_full_pdf.png) / [PDF](figures/loglogistic/V22_full_pdf.pdf) | [PNG](figures/loglogistic/V22_tail_pdf_zoom.png) / [PDF](figures/loglogistic/V22_tail_pdf_zoom.pdf) |
| 节点21电压 | 上尾 | 当前尾段重尾特征较强 | [PNG](figures/loglogistic/V21_full_pdf.png) / [PDF](figures/loglogistic/V21_full_pdf.pdf) | [PNG](figures/loglogistic/V21_tail_pdf_zoom.png) / [PDF](figures/loglogistic/V21_tail_pdf_zoom.pdf) |
| 节点20电压 | 上尾 | 当前尾段重尾特征较强 | [PNG](figures/loglogistic/V20_full_pdf.png) / [PDF](figures/loglogistic/V20_full_pdf.pdf) | [PNG](figures/loglogistic/V20_tail_pdf_zoom.png) / [PDF](figures/loglogistic/V20_tail_pdf_zoom.pdf) |
| 支路32有功（32→33） | 下尾 | 当前尾段重尾特征较强 | [PNG](figures/loglogistic/P32_full_pdf.png) / [PDF](figures/loglogistic/P32_full_pdf.pdf) | [PNG](figures/loglogistic/P32_tail_pdf_zoom.png) / [PDF](figures/loglogistic/P32_tail_pdf_zoom.pdf) |
| 支路31有功（31→32） | 下尾 | 当前尾段重尾特征较强 | [PNG](figures/loglogistic/P31_full_pdf.png) / [PDF](figures/loglogistic/P31_full_pdf.pdf) | [PNG](figures/loglogistic/P31_tail_pdf_zoom.png) / [PDF](figures/loglogistic/P31_tail_pdf_zoom.pdf) |
| 支路30有功（30→31） | 下尾 | 当前尾段重尾特征较强 | [PNG](figures/loglogistic/P30_full_pdf.png) / [PDF](figures/loglogistic/P30_full_pdf.pdf) | [PNG](figures/loglogistic/P30_tail_pdf_zoom.png) / [PDF](figures/loglogistic/P30_tail_pdf_zoom.pdf) |
| 支路18无功（2→19） | 上尾 | 当前尾段重尾特征较强 | [PNG](figures/loglogistic/Q18_full_pdf.png) / [PDF](figures/loglogistic/Q18_full_pdf.pdf) | [PNG](figures/loglogistic/Q18_tail_pdf_zoom.png) / [PDF](figures/loglogistic/Q18_tail_pdf_zoom.pdf) |
| 支路19无功（19→20） | 上尾 | 当前尾段重尾特征较强 | [PNG](figures/loglogistic/Q19_full_pdf.png) / [PDF](figures/loglogistic/Q19_full_pdf.pdf) | [PNG](figures/loglogistic/Q19_tail_pdf_zoom.png) / [PDF](figures/loglogistic/Q19_tail_pdf_zoom.pdf) |
| 支路31无功（31→32） | 上尾 | 当前尾段重尾特征较强 | [PNG](figures/loglogistic/Q31_full_pdf.png) / [PDF](figures/loglogistic/Q31_full_pdf.pdf) | [PNG](figures/loglogistic/Q31_tail_pdf_zoom.png) / [PDF](figures/loglogistic/Q31_tail_pdf_zoom.pdf) |

全部节点/线路与MC形态比较：

- 节点电压：[全部32项尾部特征](figures/loglogistic/V_all_quantities_tail_compare.png)；[标准化PDF](figures/loglogistic/V_mc_standardized_pdf_compare.png)；[标准化尾部PDF](figures/loglogistic/V_mc_standardized_tail_pdf_compare.png)。
- 支路有功：[全部32项尾部特征](figures/loglogistic/P_all_quantities_tail_compare.png)；[标准化PDF](figures/loglogistic/P_mc_standardized_pdf_compare.png)；[标准化尾部PDF](figures/loglogistic/P_mc_standardized_tail_pdf_compare.png)。
- 支路无功：[全部32项尾部特征](figures/loglogistic/Q_all_quantities_tail_compare.png)；[标准化PDF](figures/loglogistic/Q_mc_standardized_pdf_compare.png)；[标准化尾部PDF](figures/loglogistic/Q_mc_standardized_tail_pdf_compare.png)。

## 混合

| 电气量 | 尾侧 | 当前MC尾段判断 | 全分布PDF | 尾部PDF放大 |
|---|---|---|---|---|
| 节点33电压 | 上尾 | 当前尾段重尾特征较强 | [PNG](figures/mixed/V33_full_pdf.png) / [PDF](figures/mixed/V33_full_pdf.pdf) | [PNG](figures/mixed/V33_tail_pdf_zoom.png) / [PDF](figures/mixed/V33_tail_pdf_zoom.pdf) |
| 节点32电压 | 上尾 | 当前尾段重尾特征较强 | [PNG](figures/mixed/V32_full_pdf.png) / [PDF](figures/mixed/V32_full_pdf.pdf) | [PNG](figures/mixed/V32_tail_pdf_zoom.png) / [PDF](figures/mixed/V32_tail_pdf_zoom.pdf) |
| 节点31电压 | 上尾 | 当前尾段重尾特征较强 | [PNG](figures/mixed/V31_full_pdf.png) / [PDF](figures/mixed/V31_full_pdf.pdf) | [PNG](figures/mixed/V31_tail_pdf_zoom.png) / [PDF](figures/mixed/V31_tail_pdf_zoom.pdf) |
| 支路17有功（17→18） | 下尾 | 当前尾段重尾特征较强 | [PNG](figures/mixed/P17_full_pdf.png) / [PDF](figures/mixed/P17_full_pdf.pdf) | [PNG](figures/mixed/P17_tail_pdf_zoom.png) / [PDF](figures/mixed/P17_tail_pdf_zoom.pdf) |
| 支路32有功（32→33） | 下尾 | 当前尾段重尾特征较强 | [PNG](figures/mixed/P32_full_pdf.png) / [PDF](figures/mixed/P32_full_pdf.pdf) | [PNG](figures/mixed/P32_tail_pdf_zoom.png) / [PDF](figures/mixed/P32_tail_pdf_zoom.pdf) |
| 支路31有功（31→32） | 下尾 | 当前尾段重尾特征较强 | [PNG](figures/mixed/P31_full_pdf.png) / [PDF](figures/mixed/P31_full_pdf.pdf) | [PNG](figures/mixed/P31_tail_pdf_zoom.png) / [PDF](figures/mixed/P31_tail_pdf_zoom.pdf) |
| 支路30无功（30→31） | 上尾 | 当前尾段重尾特征较强 | [PNG](figures/mixed/Q30_full_pdf.png) / [PDF](figures/mixed/Q30_full_pdf.pdf) | [PNG](figures/mixed/Q30_tail_pdf_zoom.png) / [PDF](figures/mixed/Q30_tail_pdf_zoom.pdf) |
| 支路31无功（31→32） | 上尾 | 当前尾段重尾特征较强 | [PNG](figures/mixed/Q31_full_pdf.png) / [PDF](figures/mixed/Q31_full_pdf.pdf) | [PNG](figures/mixed/Q31_tail_pdf_zoom.png) / [PDF](figures/mixed/Q31_tail_pdf_zoom.pdf) |
| 支路29无功（29→30） | 上尾 | 当前尾段重尾特征较强 | [PNG](figures/mixed/Q29_full_pdf.png) / [PDF](figures/mixed/Q29_full_pdf.pdf) | [PNG](figures/mixed/Q29_tail_pdf_zoom.png) / [PDF](figures/mixed/Q29_tail_pdf_zoom.pdf) |

全部节点/线路与MC形态比较：

- 节点电压：[全部32项尾部特征](figures/mixed/V_all_quantities_tail_compare.png)；[标准化PDF](figures/mixed/V_mc_standardized_pdf_compare.png)；[标准化尾部PDF](figures/mixed/V_mc_standardized_tail_pdf_compare.png)。
- 支路有功：[全部32项尾部特征](figures/mixed/P_all_quantities_tail_compare.png)；[标准化PDF](figures/mixed/P_mc_standardized_pdf_compare.png)；[标准化尾部PDF](figures/mixed/P_mc_standardized_tail_pdf_compare.png)。
- 支路无功：[全部32项尾部特征](figures/mixed/Q_all_quantities_tail_compare.png)；[标准化PDF](figures/mixed/Q_mc_standardized_pdf_compare.png)；[标准化尾部PDF](figures/mixed/Q_mc_standardized_tail_pdf_compare.png)。
