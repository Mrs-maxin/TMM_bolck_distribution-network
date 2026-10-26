# 标准TMM论文

研究对象：配电网解析交流概率潮流中，用标准t混合模型（TMM）表示节点电压联合分布，计算节点电压、支路功率和网损的尾部分布。比较方法为AQGMM（课题组前作的GMM电压联合分布）。

| 文件或目录 | 内容 |
|---|---|
| `BTMM_note.md` | 本文研究记录：研究对象、模型、算例、证据状态和待决问题 |
| `BTMM_changelog.md` | 本文研究内容变更记录 |
| `01_IDEA/` | 主线、论点、推导、符号、证据登记、图表计划、研究轨迹 |
| `02_PAPER/` | 文稿：`manuscript/method_zh.md`为本文方法稿（第1至3.3节、计算步骤、附录A）；摘要和引言为拆分前草稿 |
| `result/heavytail_injection/case33bw_20261026/` | 重尾分布注入算例：只含AQGMM和标准TMM的结果表，`split_manifest.json`记录来源与处理方式 |
| `result/actual_output/case33bw_pv8_2006/` | 实际出力注入算例：尚无结果 |

公共代码、数据和三方法原始结果在`shared/`；拆分前的完整记录在`archive/combined_paper_20261026/`。
