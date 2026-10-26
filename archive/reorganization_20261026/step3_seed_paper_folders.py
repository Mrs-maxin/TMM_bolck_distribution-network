"""Step 3: seed paper_standardTMM/ and paper_groupedT/ from the frozen combined paper material.

02_PAPER is copied to both folders. derivation.md, symbols.md and research_trace.yaml are copied with a scope
banner. The standard-TMM method text keeps sections 1, 2, 3.1-3.3, appendix A and the standard-TMM symbols;
section 3.4, section 4, table 2, appendix B and the grouped-t references stay in the grouped-t paper.
"""
import re
import shutil
from pathlib import Path

TOP = Path(__file__).resolve().parents[2]
SRC = TOP/'archive/combined_paper_20261026'
BANNER = {
    'paper_standardTMM': '> 2026-10-26 由合并版复制到标准TMM论文。分块TMM、分组t、天气驱动方向等内容属于分组t论文，本文只保留标准TMM、AQGMM与公共的快速映射和电压二次型部分；逐段清理【待补充】。合并版原件见`archive/combined_paper_20261026/`。\n\n',
    'paper_groupedT': '> 2026-10-26 由合并版复制到分组t论文。本文保留AQGMM、标准TMM作为比较方法；合并版原件见`archive/combined_paper_20261026/`。\n\n',
}
DRAFT_BANNER = {
    'paper_standardTMM': '> 2026-10-26 拆分说明：本稿写于两篇论文拆分前，归入标准TMM论文作为起点；其中涉及分组t或分块TMM的表述需删去，并按`shared/result/heavytail_injection/`与`shared/result/actual_output/`两组算例重写【待补充】。\n\n',
    'paper_groupedT': '> 2026-10-26 拆分说明：本稿写于两篇论文拆分前，归入分组t论文作为起点；需按分组t混合的立意和两组算例重写【待补充】。\n\n',
}


def sections(text):
    parts = re.split(r'(?m)^(?=## )', text)
    return {p.splitlines()[0]: p for p in parts if p.startswith('## ')}, parts[0]


def standard_method(text):
    secs, head = sections(text)
    keep = ['## 1 交流概率潮流计算框架', '## 2 基于GMM的电压联合分布', '## 3 标准TMM',
            '## 附录A 高斯电压下电气量二次型的特征函数', '## 附录C 符号', '## 参考文献（正文新增，编号接引言）']
    sec3 = secs['## 3 标准TMM'].split('### 3.4 配电网中的局限')[0]
    head = head.replace('正文涉及三种电压联合分布模型：作为比较基础的GMM（AQGMM）、标准TMM，以及针对配电网的改进TMM（分组t混合）。',
                        '正文涉及两种电压联合分布模型：作为比较基础的GMM（AQGMM）和标准TMM。2026-10-26由合并稿拆出，分组t混合（原第4节）移至分组t论文。')
    head = head.replace('第1—5节为正文，附录A—C为正文附录，第6节为内部的实现与证据对应',
                        '第1—4节为正文，附录A、B为正文附录，第5节为内部的实现与证据对应')
    sec1 = secs['## 1 交流概率潮流计算框架'].replace(
        '第2—4节依次给出GMM、标准TMM和分组t混合三种模型，三者共用第1.1节的快速映射和式(5)的二次型。',
        '第2、3节依次给出GMM和标准TMM两种模型，两者共用第1.1节的快速映射和式(5)的二次型。')
    steps = '''## 4 计算步骤

1. **源荷场景与节点电压。** 按源荷功率规律生成注入场景，由第1.1节快速映射计算节点电压，并另生成一组场景用于选择分量数。
2. **估计AQGMM。** 对给定$K$由期望最大化得到分量归属$\\gamma_{ik}$；$K$按第二组场景上的电压似然选择。
3. **确定标准TMM参数。** 按式(9)、(10)。
4. **构造电气量二次型。** 按式(5)由网络参数得到节点电压幅值平方、支路功率和网损的系数。
5. **计算电气量分布。** 按式(11)计算特征函数，经Gil-Pelaez反演（附录A式(A2)）得到概率分布函数、概率密度和限值概率。

步骤1中锚点处的交流潮流是全部计算中唯一的完整交流潮流求解，电气量计算只用电压二次型。MC方法（逐场景求解交流潮流）只用于检验计算精度。

---

'''
    appA = secs['## 附录A 高斯电压下电气量二次型的特征函数'].replace(
        '式(22)的协方差随$U$变化，每个强度积分点需重新计算$\\mathbf L$和$\\boldsymbol\\Lambda$。',
        '标准TMM给定强度$s$时协方差为$s\\mathbf C_k$，$\\mathbf L$按$\\sqrt s$缩放，$\\boldsymbol\\Lambda$按$s$缩放，各强度积分点不需重新分解。')
    symbols = secs['## 附录C 符号'].replace('## 附录C 符号', '## 附录B 符号')
    drop = ['$\\mathcal W_g$', '$\\mathbf Q_g$', '$U$ |', '$s_{\\max,g}$', '$\\mathbf a_k$', '$c_{ij}$']
    symbols = '\n'.join(l for l in symbols.splitlines() if not any(l.startswith('| ' + d) for d in drop))
    symbols = symbols.replace('| $\\mathbf J_a$, $\\mathbf J_0$ | 快速映射锚点处的潮流雅可比矩阵（$\\mathbf J_0=\\mathbf J_a$） |',
                              '| $\\mathbf J_a$ | 快速映射锚点处的潮流雅可比矩阵 |')
    symbols = symbols.replace('| $S$, $S_g$ | 波动强度（均值为1） |', '| $S$ | 波动强度（均值为1） |')
    symbols = symbols.replace('| $\\nu_k$, $\\nu_{g,k}$ | 自由度 |', '| $\\nu_k$ | 自由度 |')
    refs = secs['## 参考文献（正文新增，编号接引言）']
    refs = '\n\n'.join(b for b in refs.split('\n\n') if not re.match(r'\[(18|19|20|21)\]', b.strip()))
    internal = '''
## 5 实现与证据对应（内部，不进入论文）

| 内容 | 实现 | 证据状态 |
|---|---|---|
| 快速映射（式(3)、(4)） | `shared/+BTMM_core/+inverse/map_samples_with_atlas.m`；Python调用`shared/tests/python/fast_voltage_map.py` | 已由数值实验支持（合并版E-FASTMAP-20261024） |
| 二次型（式(5)） | 前作[14]；`shared/tests/python/run_pv_fixed_forecast_case33.py`中`evaluate_outputs` | 已由代码实现支持 |
| 标准TMM按矩（式(9)、(10)） | `shared/tests/python/moment_matched_tmm_case33.py`中`moment_models` | 已由代码实现支持 |
| 解析计算（式(11)、附录A） | `shared/tests/python/analytic_grouped_t_case33.py`（三种模型共用；标准TMM 40点广义Gauss–Laguerre，Gil-Pelaez反演） | 已由数值实验支持（合并版E-ANALYTIC-20261024） |
| 重尾分布注入算例 | `shared/tests/python/run_direct_heavytail_case33.py`；本文表格`paper_standardTMM/result/heavytail_injection/case33bw_20261026/` | 见`01_IDEA/evidence_map.md` E-STD-HEAVYTAIL-20261026 |
'''
    return head + sec1 + secs['## 2 基于GMM的电压联合分布'] + sec3 + steps + appA + symbols + '\n\n' + refs.rstrip() + '\n\n---\n' + internal


def grouped_method(text):
    head, internal = text.split('## 6 实现与证据对应（内部，不进入论文）')
    internal = internal.replace('`tests/python/', '`shared/tests/python/').replace('`+BTMM_core/', '`shared/+BTMM_core/')
    internal = internal.replace('`shared/tests/python/mainline_analytic_case33.py`', '`archive/code/tests/python/mainline_analytic_case33.py`')
    internal += '| 重尾分布注入算例 | `shared/tests/python/run_direct_heavytail_case33.py`；本文表格`paper_groupedT/result/heavytail_injection/case33bw_20261026/` | 见`01_IDEA/evidence_map.md` E-GT-HEAVYTAIL-20261026 |\n'
    head = head.replace('日期：2026-10-24。', '日期：2026-10-24；2026-10-26拆分为分组t论文的方法稿。')
    return head + '## 6 实现与证据对应（内部，不进入论文）' + internal


def main():
    method = (SRC/'02_PAPER/manuscript/method_zh.md').read_text()
    for paper in ['paper_standardTMM', 'paper_groupedT']:
        out = TOP/paper
        assert not (out/'02_PAPER').exists() and not (out/'01_IDEA').exists(), paper
        shutil.copytree(SRC/'02_PAPER', out/'02_PAPER')
        for name in ['abstract_zh.md', 'introduction_zh.md', 'introduction_method_tail_zh.md']:
            p = out/'02_PAPER/manuscript'/name
            p.write_text(DRAFT_BANNER[paper] + p.read_text())
        (out/'02_PAPER/manuscript/method_zh.md').write_text(
            standard_method(method) if paper == 'paper_standardTMM' else grouped_method(method))
        (out/'01_IDEA').mkdir()
        for name in ['derivation.md', 'symbols.md']:
            text = (SRC/'01_IDEA'/name).read_text()
            title, rest = text.split('\n', 1)
            (out/'01_IDEA'/name).write_text(title + '\n\n' + BANNER[paper] + rest.lstrip('\n'))
        shutil.copyfile(SRC/'01_IDEA/research_trace.yaml', out/'01_IDEA/research_trace.yaml')
        print('seeded', paper)


if __name__ == '__main__':
    main()
