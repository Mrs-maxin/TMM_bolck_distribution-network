# IEEE33重尾测试：结果阅读入口

主要误差现包括尾部CDF-RMSE、极端分位数误差和尾部分位区间W1，另报告全分布均值方差。当前没有全分布CDF-RMSE表。

原“结果/CDF.csv”现名为“结果/tail_cdf_rmse_all_quantities.csv”。它每行汇总某一尾侧50个阈值的概率误差，不是CDF曲线，也不是全局CDF误差。原“尾段.csv”现名为“tail_cdf_per_threshold.csv”，才是逐阈值的CDF及尾概率明细。

## 本轮新增：全部节点与线路的尾部形状、PDF、W1和全分布矩

- [尾部分析完整说明](tail_distribution/README.md)：先按全部MC结果分析位置，再比较方法，不只看节点18。
- [PDF图册](tail_distribution/figure_atlas.md)：36个重点电气量的全分布及尾部放大图，全部K保留；另有全部32项比较图。
- [重点电气量三项误差及均值方差](tail_distribution/focus_quantities_three_metrics_mean_variance.csv)：432行。
- [尾部分位区间W1比较](results/tail_quantile_w1_method_compare.csv)；[全分布均值方差比较](results/full_mean_variance_method_compare.csv)。

本轮MC形状分析用于补充重点出图，原373项尾部延伸标记保留不变。新分析不改变注入、负荷、相关系数和任何已保存模型。

## 先看这三个文件

- [筛选电气量的尾部CDF-RMSE三方法比较](tail_screening/tail_cdf_rmse_method_compare.csv)：1492行，每行直接比较同一情形、同一K、电气量和尾侧，误差单位为百分点。
- [筛选电气量的极端分位误差三方法比较](tail_screening/extreme_quantile_error_method_compare.csv)：4476行，每个分位水平分别比较，单位见表中单位列。
- [可筛选结果表](results_browser.html)：包含全部电气量，可切换“MC已标记尾部延伸”或“全部电气量”，不预先固定某个K或情形。

筛选依据为既有MC的尾部延伸比超过1.15，共373个“情形×电气量×尾侧”。这是有限概率范围的筛选，不等于373项都已证明为严格渐近重尾。

## 分组t的结果应如何表述

**现有结果不支持“分组t整体较差”的概括。**它在部分节点电压和支路功率的尾部误差较小，在部分电气量上较大；两类误差也可能给出不同次序。此前过多强调K=2的个别无功结果，不能据此概括其他K、节点和线路。

分组t的参数一致性未通过，与它在某些电气量上误差更小可以同时成立。当前数值改善如实保留；参数问题另见[模型参数核查](model_parameter_checks/README.md)，不能把参数未通过解释成每个电气量都较差，也不能把某些误差更小解释成参数关系已经满足。

以下仅统计MC已标记的373项，方便定位结果；这是逐项比较数量，不是全网平均误差或理论有效性的总评分。

| K | 比较项数 | 分组t误差小于AQGMM项数 | 分组t误差小于标准TMM项数 | 分组t误差三者最小项数 |
|---|---|---|---|---|
| 1 | 373 | 158 | 75 | 73 |
| 2 | 373 | 205 | 180 | 125 |
| 4 | 373 | 282 | 260 | 237 |
| 6 | 373 | 262 | 238 | 209 |

下列两项同时保留，CDF-RMSE单位均为百分点：

| 情形 | K | 电气量 | 尾侧 | AQGMM | 标准TMM | 分组t |
|---|---|---|---|---|---|---|
| Pareto | 2 | Q17 | 上尾 | 15.4229 | 15.5322 | 22.3757 |
| Pareto | 4 | V18 | 上尾 | 0.471067 | 0.281909 | 0.154382 |

极端分位误差必须另看。例如Pareto、K=2的节点18电压上尾，分组t的尾部CDF-RMSE较大，但99.9%分位值绝对误差较小。不能由一个指标推断另一个指标的次序。

## 文件用途

| 位置 | 内容 | 是否用于当前精度比较 |
|---|---|---|
| [结果](results/README.md) | 最新两项误差、逐阈值CDF、MC分位值、电气量编号 | 是，逐电气量、逐K、逐尾侧比较 |
| [尾部筛选](tail_screening/README.md) | 按既有MC标记筛选的同一批结果 | 主要阅读入口 |
| [模型参数核查](model_parameter_checks/README.md) | 均值、协方差、随机强度矩、自由度等关系 | 是参数检查，不是CDF误差排名 |
| [诊断记录](diagnostics/README.md) | 改变某一物理计算条件或自由度以追查原因 | 与原模型精度表分开阅读 |
| [历史结果](legacy_metrics/README.md) | 原尾部平均绝对概率误差MAE | 不能当作当前RMSE |
| 计算记录 | 数值求解、积分加密、文件改名与哈希核验 | 辅助核查 |
| 程序 | 重算误差、整理表格及诊断程序 | 不混放数值结果 |
| 存档 | 冻结原始程序、注入、模型、百万场景MC和早期记录 | 保持原样 |

所有当前文件的用途见[file_index.csv](file_index.csv)，逐列含义见[field_dictionary.csv](field_dictionary.csv)，旧名与新名见[文件改名对照](computation_records/file_rename_map_20261026_cn.csv)。

本次整理逐项复算465600个尾部阈值对应的RMSE，核对分位误差和筛选表。主误差数值未更改；未生成注入、求新潮流或拟合模型。441个冻结存档文件的SHA256全部一致。

算例设置见[case_settings.csv](case_settings.csv)，注入PDF见[injection_power_pdf.png](injection_power_pdf.png)。原始MC单位仍为p.u.、MW、Mvar；用于汇报的分位误差为p.u.、kW、kvar。

