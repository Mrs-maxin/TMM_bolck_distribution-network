# shared/result

两篇论文共用的计算结果，按计算场景算例分组。每组只保存一份三方法（AQGMM、标准TMM、分组t）原始结果；各论文所用表格由`shared/tests/python/split_results_by_paper.py`拆分到`paper_standardTMM/result/`和`paper_groupedT/result/`。

| 算例组 | 目录 | 内容 |
|---|---|---|
| 重尾分布注入 | `heavytail_injection/case33bw_20261026/` | `run_outputs/`为计算入口的原始输出（注入、MC电气量、三方法模型、CDF网格，大数组Git忽略）；`results/`、`tail_screening/`、`tail_distribution/`、`model_parameter_checks/`、`diagnostics/`为后处理结果；`audit_theory/`为参数一致性复核；`records/`为首轮交付记录 |
| 实际出力注入 | `actual_output/case33bw_pv8_2006/` | `cloud_package_20261026/`为待执行的云端计算包；计算结果将写入`python_cloud_20261026/` |

新增算例时按“算例组/网络_输入版本”建立目录，并在`shared/protocols/`中先登记协议。
