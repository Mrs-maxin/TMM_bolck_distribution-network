# 历史误差：尾部CDF平均绝对误差MAE

本目录保留首轮指标，不能作为当前尾部CDF-RMSE。

- tail_cdf_mae_all_quantities.csv：全部电气量；lower_tail_pp和upper_tail_pp是50档阈值处的平均绝对概率误差，单位为百分点。
- tail_cdf_mae_screened_quantities.csv：相同旧指标在MC筛选电气量上的结果。

full_CDF_max_pp是251个既有MC分位阈值上的CDF最大绝对差，单位为百分点；不是全分布CDF-RMSE，也不是精确KS距离。CF_end_abs是积分检查量，不是精度指标。原列名保留，逐列说明见根目录field_dictionary.csv。
