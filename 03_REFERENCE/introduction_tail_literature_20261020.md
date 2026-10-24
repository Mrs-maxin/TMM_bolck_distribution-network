# 中文引言的尾部研究文献依据

本文件保留前两版11项文献及原编号的核查历史。当前引言已改为15项文献，编号与本表不同；应以[尾部计算专项检索记录](tail_computation_search_20261020.md)及 `02_PAPER/introduction_zh.md` 为准。

日期：2026-10-20。当前正文为 `02_PAPER/introduction_zh.md`，原始引言工作区保留在 `TMM_note.md`。这里只记录文献与陈述的对应关系，不记录新的科学计算结果。

## 文献与使用范围

| 正文编号 | 文献及原始来源 | 用于支撑的陈述 | 已读取范围与限制 |
|---|---|---|---|
| 1 | Wellby SJ, Engerer NA. Categorizing the Meteorological Origins of Critical Ramp Events in Collective Photovoltaic Array Output. JAMC, 2016,55(6):1323–1344. DOI:10.1175/JAMC-D-15-0107.1。[作者机构库](https://researchportalplus.anu.edu.au/en/publications/categorizing-the-meteorological-origins-of-critical-ramp-events-i/) | 云层变化可引起区域光伏出力集中爬坡 | 作者机构库摘要、期刊检索摘要及Crossref书目信息。不能据此把所有多云天气称为极端事件，也不证明联合电压必然厚尾 |
| 2 | 葛晓琳等. 基于降尺度台风模拟的海上风电有序避险时空优化调度策略. 中国电机工程学报,2025,45(19):7575–7588. DOI:10.13334/j.0258-8013.pcsee.240911。[期刊](https://epjournal.csee.org.cn/zh/article/doi/10.13334/j.0258-8013.pcsee.240911/) | 台风影响下风机切出和风电功率爬坡 | PowerLit引言片段及期刊摘要、书目信息。本文不引入该文避险优化模型 |
| 3 | Xu Y, Korkali M, Mili L, et al. Risk Assessment of Rare Events in Probabilistic Power Flow via Hybrid Multi-Surrogate Method. TSG,2020,11(2):1593–1603. DOI:10.1109/TSG.2019.2940928。[作者成果公开存档](https://www.osti.gov/servlets/purl/1668487) | 全局代理模型的尾部精度问题；对低概率尾部样本使用高精度模型重新计算 | PowerLit正文开头包含摘要、引言；Crossref核对最终卷期页。公开PDF网页接口超时，未伪称本次完整读取PDF |
| 4 | Wang C, Zhou Y, Peng Y, et al. Efficient surrogate-assisted importance sampling for rare event assessment in probabilistic power flow. Journal of Renewable and Sustainable Energy,2024,16(1):015501. DOI:10.1063/5.0177383。[出版方登记元数据与摘要](https://api.crossref.org/works/10.1063/5.0177383) | 代理模型辅助重要性抽样，面向电压与线路越限小概率事件 | 2026-10-20直接读取出版方提交的Crossref摘要与完整元数据；只据摘要概括方法，不推断未读取的全文结果。替换原稀有事件敏感性文献，使核心引用直接对应分布尾部计算 |
| 5 | Usaola J. Probabilistic load flow with wind production uncertainty using cumulants and Cornish–Fisher expansion. IJEPES,2009,31(9):474–481. DOI:10.1016/j.ijepes.2009.02.003。[期刊](https://www.sciencedirect.com/science/article/pii/S0142061509000258) | 非高斯风电出力下的累积量与分布展开 | 期刊摘要、作者机构库检索正文及Crossref。只说明方法能力，不把该文归结为尾部失效 |
| 6 | Tourandaz Kenari M, Sepasian MS, Setayesh Nazar M, et al. Combined cumulants and Laplace transform method for probabilistic load flow analysis. IET GTD,2017,11(14):3548–3556. DOI:10.1049/iet-gtd.2017.0097。[期刊全文](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-gtd.2017.0097) | 通过拉普拉斯变换改进累积量重构；展开方法存在尾部适用条件 | 期刊全文引言、方法及结果段落。文中交流潮流使用线性化，不能将其限制套用于本项目完整交流GMM基线 |
| 7 | Singh V, Moger T, Jena D. A modified point estimate-based probabilistic load flow approach for improving tail accuracy in wind-integrated power systems. EPSR,2025,245:111606. DOI:10.1016/j.epsr.2025.111606。[作者机构库](https://idr.nitk.ac.in/items/0ce0c735-2b91-450d-8704-cbf46e71b2ae) | 通过增加尾部代表点改进点估计法 | 作者机构库摘要、Crossref。没有读取全文，不推断其所有指标或运行条件 |
| 8 | Nijhuis M, Gibescu M, Cobben S. Gaussian Mixture Based Probabilistic Load Flow For LV-Network Planning. TPWRS,2017,32(4):2878–2886. DOI:10.1109/TPWRS.2016.2628959。[作者机构库](https://research.tue.nl/en/publications/gaussian-mixture-based-probabilistic-load-flow-for-lv-network-pla/) | GMM用于复杂负荷分布，并结合低压网特点进行解析计算 | PowerLit引言片段、作者机构库摘要与最终书目信息 |
| 9 | Gao Y, Xu X, Yan Z, et al. Analytical Probabilistic Power Flow and Global Sensitivity Analysis of Distribution Systems Based on Gaussian Mixture Model of Input-Output Variables. TPWRS,2024,39(3):5283–5296. DOI:10.1109/TPWRS.2023.3329197。[作者主页](https://wanghan126.github.io/publication/2023-11-01-journal-en-Analytical-Probabilistic-Power-Flow-and-Global-Sensitivity-Analysis) | GMM表示非高斯分布与相关性，输入输出联合建模，分段线性解析概率潮流 | PowerLit摘要、引言前半及作者主页；2026-10-20直接读取Crossref，补齐39(3):5283–5296。 |
| 10 | 吴涵. 基于随机变量二次型的配电网交流概率潮流建模和解析计算方法(一)：从电量到功率量的推导. 中国电机工程学报,2026,46(3):928–941. DOI:10.13334/j.0258-8013.pcsee.242015。[期刊](https://csee.publish.founderss.cn/zh/article/doi/10.13334/j.0258-8013.pcsee.242015/) | 交流功率的电压二次型关系以及高斯电压下的分布解析计算 | PowerLit引言及首节片段、期刊页面。不能把该文直接写成TMM或完整GMM方法 |
| 11 | Peel D, McLachlan GJ. Robust mixture modelling using the t distribution. Statistics and Computing,2000,10(4):339–348. DOI:10.1023/A:1008981510081。[作者公开稿](https://people.smp.uq.edu.au/GeoffMcLachlan/pm_sc00.pdf) | 标准多元t混合的分量及较长尾部表示 | 作者公开稿摘要、引言及Crossref。不将聚类稳健性直接等同电力系统真实尾部计算更准 |

## 解析性与现有推导的对应

当前项目 `01_IDEA/derivation.md` 给出标准多元t混合、完整交流电压到功率关系、二次型解析矩，以及2026-10-08条件概率积分。`+TMM_core/+metrics/quadratic_t_cdf.m` 将一个坐标的二次不等式概率解析求出，再对剩余变量积分。引言据此表述为解析积分表达式与数值积分，不声称全部PDF/CDF均为闭式。

历史相邻项目 `2-tmm_PFF/02_PAPER/drafts/07-methodology_zh.md` 还保留条件高斯二次型特征函数、尺度积分和分布反演推导。本次只读，不将该历史实现直接认作当前天气算例的执行路径。当前天气误差与曲线采用已保存模型的加权Sobol传播，与解析公式的理论身份分别记录。本文方法主体仍需清楚对应所采用的解析公式；当前文字没有新增“解析反演已在天气批次完成”的数值主张。

## 整体行文复核

修改稿顺序为天气源荷与馈线受电或反送、尾部分布与运行限值、稀有事件模拟、统计量近似、GMM及高斯二次型解析研究、TMM依据和本文解析计算方法。GMM作为最后一个概率潮流文献类别进入方法，不称为所有类别中发表时间最晚的方法。首稿不预告算例，不列节点线路编号，不报告本轮改善数值，不引入节点识别方法。分布较长尾部与严格渐近重尾不混用。

## 审查后的引用取舍

原初稿文献[4] Li等2025年稀有事件敏感性研究（DOI:10.1109/TPWRS.2025.3527251）保持其原学术身份，本版正文改引直接研究概率潮流小概率越限估计的Wang等2024年论文。TMM必要性来自概率集中区域之外的较长尾部及分量规模关系，多峰本身不是充分依据。改稿明确所述解析输出为累积分布函数的积分表达式，未新增全部PDF闭式解或天气批次已完成解析计算验证的主张。
