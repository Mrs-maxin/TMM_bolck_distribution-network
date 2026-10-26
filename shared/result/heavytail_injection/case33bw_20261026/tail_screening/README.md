# MC已标记尾部延伸的电气量

本目录使用此前保存的MC筛选结果，未改变筛选阈值，也未按方法误差挑选电气量。共373个“注入情形×电气量×尾侧”，对应4个K。

| 文件 | 内容 |
|---|---|
| [tail_extended_quantity_list.csv](tail_extended_quantity_list.csv) | 373项，附MC偏度、超额峰度和该侧尾部延伸比 |
| [tail_cdf_rmse_method_compare.csv](tail_cdf_rmse_method_compare.csv) | 1492行，同一电气量三方法并列，单位为百分点 |
| [extreme_quantile_error_method_compare.csv](extreme_quantile_error_method_compare.csv) | 4476行，逐分位水平比较，单位见表中单位列 |
| [tail_cdf_rmse_screened_quantities.csv](tail_cdf_rmse_screened_quantities.csv) | 4476行，逐方法的尾部CDF-RMSE长表 |
| [extreme_quantile_error_screened_quantities.csv](extreme_quantile_error_screened_quantities.csv) | 13428行，逐方法的极端分位误差长表 |
| [tail_two_metrics_screened_merged.csv](tail_two_metrics_screened_merged.csv) | 13428行，分位误差附相应尾侧CDF-RMSE；后者在同侧三行重复 |
| [tail_cdf_rmse_win_counts_by_K.csv](tail_cdf_rmse_win_counts_by_K.csv) | 逐K记录误差较小的比较项数，不能替代逐电气量结果 |
| [extreme_quantile_win_counts_by_K_level.csv](extreme_quantile_win_counts_by_K_level.csv) | 不同分位水平分开记录，不平均不同物理单位 |

筛选依据：MC的0.5%或99.5%分位到中位数的距离，与相同标准差高斯分布对应距离相比超过1.15。这只能说明该概率范围的尾部延伸，不是严格渐近重尾证明。

分组t的数值改善和变差均保留。参数一致性问题与误差大小分别记录，不以参数检查未通过推断误差一定较大。

| K | 比较项数 | 分组t误差小于AQGMM项数 | 分组t误差小于标准TMM项数 | 分组t误差三者最小项数 |
|---|---|---|---|---|
| 1 | 373 | 158 | 75 | 73 |
| 2 | 373 | 205 | 180 | 125 |
| 4 | 373 | 282 | 260 | 237 |
| 6 | 373 | 262 | 238 | 209 |
