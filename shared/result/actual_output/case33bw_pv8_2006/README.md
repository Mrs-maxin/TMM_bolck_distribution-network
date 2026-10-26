# 实际出力注入：IEEE 33节点，NREL 2006年8站同步出力

协议见`shared/protocols/actual_output_pv8_2006_selection_20261026.md`（选站）和`shared/protocols/actual_output_pv8_2006_python_cloud_20261026.md`（计算与诊断）。输入数据为`shared/data/nrel_sind/pv8_direct_2006/`。

| 目录 | 内容 | 状态 |
|---|---|---|
| `cloud_package_20261026/` | `shared/tests/python/build_direct_pv8_cloud_bundle.py`生成的云端计算包及回执 | 包内为拆分前目录结构，入口输出目录为`result/case33bw/direct_pv8_python_cloud_20261026`；打包脚本中随包附带的说明文件（`AGENTS.md`、`01_IDEA/plans/*`、`02_PAPER/manuscript/method_zh.md`等）仍按拆分前的根目录列出，需改为`shared/protocols/`与相应论文目录后重新打包，再提交云端【待补充】 |
| `python_cloud_20261026/` | `shared/tests/python/run_direct_pv8_python_cloud.py`的计算结果 | 尚未计算 |

计算返回后运行`python3 shared/tests/python/split_results_by_paper.py actual_output/case33bw_pv8_2006`生成两篇论文的表格（结果表需与重尾算例采用相同列名）。
