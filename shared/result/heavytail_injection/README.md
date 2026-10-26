# 重尾分布注入算例组

电源有功出力直接服从给定的重尾或长尾分布，不采用预测加误差，负荷恒定。协议见`shared/protocols/heavytail_injection_case33bw_20261026_scope.md`。

| 目录 | 网络与情形 | 阅读入口 |
|---|---|---|
| `case33bw_20261026/` | IEEE 33节点；Pareto、对数正态、对数逻辑斯蒂、不同节点采用不同分布4种情形；K=1、2、4、6 | `case33bw_20261026/README.md`（结果阅读入口，原`IEEE33重尾/说明.md`）；`error_metrics.md`（误差定义）；`case_settings.csv`（逐节点注入设置） |

`case33bw_20261026/`由`/Users/maxin/Documents/Codex/IEEE33重尾`于2026-10-26导入，文件名已改为英文，新旧名对照见`archive/reorganization_20261026/step2_import_map.csv`；表格列名和内容保持中文原样。计算入口为`shared/tests/python/run_direct_heavytail_case33.py`，后处理程序在`shared/tests/python/heavytail_postprocess/`。
