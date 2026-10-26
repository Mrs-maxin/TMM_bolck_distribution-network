# 本文结果（AQGMM与标准TMM）

| 目录 | 算例组 | 内容 |
|---|---|---|
| `heavytail_injection/case33bw_20261026/` | 重尾分布注入，IEEE 33节点 | 由`shared/result/heavytail_injection/case33bw_20261026/`拆出，只保留AQGMM和标准TMM；宽表的排序、差值、胜出列及胜出计数按两种方法重算。来源SHA256与处理方式见`split_manifest.json`，字段含义见`shared/`中同名目录的`field_dictionary.csv`和`error_metrics.md` |
| `actual_output/case33bw_pv8_2006/` | 实际出力注入，IEEE 33节点，NREL 2006年8个光伏电站 | 尚无结果【待补充】 |

`results/tail_cdf_per_threshold.csv`超过60 MB，不进入Git，可由`python shared/tests/python/split_results_by_paper.py`从公共结果重新生成。
