# 当前误差表：文件、指标与单位

## 当前指标

| 文件 | 每行含义 | 指标与单位 | 行数 |
|---|---|---|---:|
| [tail_cdf_rmse_all_quantities.csv](tail_cdf_rmse_all_quantities.csv) | 情形×方法×K×电气量×尾侧 | 50档尾部CDF-RMSE；概率值与百分点均保留 | 9312 |
| [tail_cdf_rmse_method_compare.csv](tail_cdf_rmse_method_compare.csv) | 情形×K×电气量×尾侧 | 三种方法的同一指标，列名明确为百分点 | 3104 |
| [extreme_quantile_error_all_quantities.csv](extreme_quantile_error_all_quantities.csv) | 情形×方法×K×电气量×分位水平 | MC分位值、方法分位值、带符号与绝对误差；p.u./kW/kvar | 27936 |
| [extreme_quantile_error_method_compare.csv](extreme_quantile_error_method_compare.csv) | 情形×K×电气量×分位水平 | 三种方法的分位值及误差，同一行单位相同 | 9312 |
| [tail_cdf_per_threshold.csv](tail_cdf_per_threshold.csv) | 情形×方法×K×电气量×尾侧×阈值 | CDF和尾概率为0—1；概率差另以百分点列出 | 465600 |
| [tail_two_metrics_merged.csv](tail_two_metrics_merged.csv) | 分位误差每行附相应尾侧CDF-RMSE | CDF-RMSE在同侧三个极端分位水平重复，不应重复汇总 | 27936 |
| [mc_extreme_quantiles_ci95.csv](mc_extreme_quantiles_ci95.csv) | 情形×电气量×分位水平 | 仅为MC基准分位值及逐点区间，不是方法误差 | 2328 |
| [mc_quantity_distribution_features.csv](mc_quantity_distribution_features.csv) | 情形×电气量 | MC均值、标准差、偏度、超额峰度和尾部延伸比；原单位见字段说明 | 388 |
| [quantity_ids_units.csv](quantity_ids_units.csv) | 一个电气量 | 节点号、支路方向及汇报单位 | 97 |

尾部CDF-RMSE：下尾取MC的0.1%—5%分位阈值，上尾取95%—99.9%分位阈值，每侧50档。在相同阈值比较方法CDF与MC经验CDF，概率差平方、平均、开平方。乘100后为百分点，不是相对百分比。

极端分位水平：0.001、0.005、0.01、0.99、0.995、0.999。带符号误差＝方法分位值−MC分位值；比较精度时看绝对误差。两项指标不合并为一个分数。

## 当前没有的指标

本目录没有“全分布CDF-RMSE”。历史MAE文件中的full_CDF_max_pp为251个既有MC分位阈值上的CDF最大差，范围0.1%—99.9%；它既不是全分布RMSE，也不是严格KS距离。全分布曲线原始网格保存在冻结存档的CDF_方法_K编号.npz中。

## 比较范围

分组t、标准TMM、AQGMM的误差按同一行条件比较。不同K、不同节点/线路、不同尾侧可能次序不同。参数状态列只说明模型关系检查，不承担误差排序。重点结论见[尾部筛选](../tail_screening/README.md)，完整定义见[误差标准](../error_metrics.md)。


## 新增结果文件

- [tail_quantile_w1_all_quantities.csv](tail_quantile_w1_all_quantities.csv)：9312个上下尾距离；区间[0.001,0.05]或[0.95,0.999]，不是完整W1。
- [tail_quantile_w1_method_compare.csv](tail_quantile_w1_method_compare.csv)：3104行，同一行的单位相同。
- [full_mean_variance_all_quantities.csv](full_mean_variance_all_quantities.csv)：4656行，含MC及方法均值、方差、误差和标准差。
- [full_mean_variance_method_compare.csv](full_mean_variance_method_compare.csv)：1552行，均值和方差另列单位。

电压统计量为幅值，非幅值平方。方法方差/MC方差＝1表示一致。完整定义及PDF图见[本轮尾部分析](../tail_distribution/README.md)。
