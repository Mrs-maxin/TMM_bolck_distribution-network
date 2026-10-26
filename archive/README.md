# archive

2026-10-26拆分为两篇论文时移入的材料。内容保持原样，只改变所在位置；逐文件对照见`reorganization_20261026/`。

| 目录 | 内容 | 说明 |
|---|---|---|
| `combined_paper_20261026/` | 拆分前的`01_IDEA/`、`02_PAPER/`、`BTMM_note.md`、`BTMM_changelog.md`、`BTMM_blog.md`和根目录`README.md` | 一篇论文包含AQGMM、标准TMM和分组t三种模型时的完整研究记录；两篇论文的历史判断以此为准 |
| `result/case33bw/`、`result/case141/` | 2026-10-21至10-26按参数规律、NREL预测误差记录（S1、S2E1、S2E2、S3）和共同云层构造输入的算例结果 | 不进入两篇论文；其中T5至T14的输入与TMM结构相同，合并版已降为【结果待复核】 |
| `result/tests_python_result/` | 早期脚本在`tests/python/result/`下误写的云端审计目录 | 保留原样 |
| `cache/record_scenarios_*` | S1至S3的百万场景MC输出（约4.2 GB，Git忽略） | 仅本地保存 |
| `data/experiments/ieee33_20261004/` | AQGMM前作的33节点冻结输入 | 只被归档脚本`block_split_rule_scenarios.py`使用 |
| `code/tests/python/` | 只服务于上述归档算例的33个Python脚本 | 依赖`shared/tests/python/`中的公共模块；需要重算时复制回该目录运行 |
| `heavytail_source_snapshot_20261026/` | 重尾分布注入首轮计算时的完整源程序快照（原`IEEE33重尾/存档`的程序部分） | 冻结；`shared/tests/python/heavytail_postprocess/recompute.py`从这里读取原计算函数 |
| `misc/` | Codex云端任务工具日志 | Git忽略 |
| `reorganization_20261026/` | 本次整理的程序与对照表 | 见下 |

## 2026-10-26整理记录

- `step1_move_repository_content.py`、`step1_move_map.csv`：仓库内文件移入`shared/`和`archive/`，同一卷内改名，文件内容不变。
- `step2_import_ieee33_heavytail.py`、`step2_import_map.csv`：导入`/Users/maxin/Documents/Codex/IEEE33重尾`并改为英文文件名。986行中838个文件原样复制；9个重尾计算程序另复制到`shared/tests/python/`，其中`block_split_rule_check.py`（电压维数由64改为实际维数）和`fast_voltage_map.py`（增加MATLAB日志文件）替换了旧版本，旧版本SHA256记在状态列；138个说明、索引、图清单和后处理程序只改写了文件名字符串；冻结清单增加原路径列。441个冻结文件在新位置全部核对一致。
- `step2b_fix_prose_paths.py`：正文中紧跟汉字的文件名和目录名。
- `step3_seed_paper_folders.py`：由合并版生成两篇论文的`02_PAPER/`和推导、符号文件，标准TMM论文的方法稿只保留第1至3.3节、计算步骤和附录A。
- `previous_cn_reorganization/reorganize_results_cn.py`：`IEEE33重尾`内部上一次改名所用程序，原样保留，不应再运行。
