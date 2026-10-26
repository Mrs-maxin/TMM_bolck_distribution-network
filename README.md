# 配电网交流概率潮流的TMM电压联合分布：两篇论文的工作目录

2026-10-26起，本仓库按两篇论文组织：标准TMM论文比较AQGMM与标准TMM，分组t论文比较AQGMM、标准TMM与分组t混合。两篇论文共用同一套潮流、快速映射、电压二次型和解析计算程序，以及同一批源荷场景和MC方法参照结果。

## 目录

| 目录 | 内容 |
|---|---|
| `shared/` | 共用代码根目录：`+BTMM_core/`、`+BTMM_utils/`、`+BTMM_sota/`、`tests/`、`data/`、`cache/`、`result/`、`protocols/`、`03_REFERENCE/`、MATLAB入口`BTMM*.m` |
| `paper_standardTMM/` | 标准TMM论文：`01_IDEA/`、`02_PAPER/`、研究记录`BTMM_note.md`与`BTMM_changelog.md`、只含AQGMM和标准TMM的结果表`result/` |
| `paper_groupedT/` | 分组t论文：同上，结果表含AQGMM、标准TMM和分组t |
| `archive/` | 拆分前的合并版论文材料、按预测误差或参数规律构造输入的早期算例、旧代码、旧缓存和整理记录 |

`shared/`内的脚本按自身位置确定根目录（`Path(__file__).parents[2]`为`shared/`），MATLAB入口用`fileparts(mfilename('fullpath'))`，因此`data/`、`cache/`、`result/`的相对路径与拆分前一致。

## 计算场景算例

两篇论文使用相同的两组算例，结果按算例组分开保存。

| 算例组 | 输入 | 共用结果 | 论文表格 | 状态 |
|---|---|---|---|---|
| 重尾分布注入 | IEEE 33节点，8个电源节点（18、17、16、15、14、13、33、22），有功出力直接服从Pareto、对数正态、对数逻辑斯蒂或三类混用，均值0.25 MW、标准差0.0645 MW，Pearson相关系数0.3，负荷恒定；每种情形100万个场景逐场景交流潮流 | `shared/result/heavytail_injection/case33bw_20261026/` | `paper_*/result/heavytail_injection/case33bw_20261026/` | 已完成首轮计算；分组t参数一致性未通过，见共用结果`README.md` |
| 实际出力注入 | IEEE 33节点，NREL 2006年8站同步实际出力（10:00至14:00），每站0.60 MW，不加扰动、不用预测误差 | `shared/result/actual_output/case33bw_pv8_2006/` | `paper_*/result/actual_output/case33bw_pv8_2006/` | 数据与云端计算包已准备，尚无计算结果 |

算例协议见`shared/protocols/`。拆分前按NREL预测误差记录构造的S1至S3场景、参数规律构造的光伏和风电场景及141节点算例均已归档，不进入两篇论文，见`archive/README.md`。

## 运行

```bash
cd shared/tests/python
python3 run_direct_heavytail_case33.py          # 重尾分布注入：生成注入、MC交流潮流、三方法电压分布与电气量CDF
python3 heavytail_postprocess/recompute.py      # 由已保存模型和MC结果重算尾部CDF-RMSE与极端分位误差
python3 run_direct_pv8_python_cloud.py          # 实际出力注入（云端执行）
python3 split_results_by_paper.py heavytail_injection/case33bw_20261026   # 按论文拆分三方法比较表
```

快速映射需要MATLAB与MATPOWER（`shared/tests/python/fast_voltage_map.py`调用`shared/tests/map_injections_fast_case33.m`）。大于Git限制的MC数组、模型文件和缓存保留在本地并由`.gitignore`排除，SHA256见各结果目录的清单。
