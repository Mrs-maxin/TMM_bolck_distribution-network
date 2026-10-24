# 概率潮流尾部计算文献检索与核验

## 2026-10-20 补查：标准t性质与二次型原创性范围

本轮针对理论归属补查原始统计文献及t混合概率潮流近邻，不替代下方既有尾部研究综述。引言现有16项参考文献，新增[16]为Peng Ding，On the Conditional Distribution of the Multivariate t Distribution，The American Statistician，2016，70(3):293–295，DOI：[10.1080/00031305.2016.1164756](https://doi.org/10.1080/00031305.2016.1164756)。出版方信息及作者[公开全文](https://arxiv.org/html/1604.00561v1)已核对。该文给出的条件自由度和尺度变化是已有性质，承担引言第七段的依据。

另读取José A. Díaz-García的[2012年作者公开稿](https://arxiv.org/html/1210.5283v1)，题为Distribution theory of quadratic forms with matrix argument。其摘要与引言明确包含椭圆分布及t分布二次型的密度、特征函数，并不要求二次型矩阵正定。未把它与本项目带一次项的标量功率条件积分视为同一算法，未据此补写尚未核实的期刊卷期。该来源用于限制首次t二次型理论的表述，暂不为扩充正文篇幅加入引言参考文献。

补查关键词包括multivariate t conditional distribution、Student t quadratic form distribution、probabilistic power flow t-mixture及对应中文词。未取得可确认与本项目完全相同的概率潮流条件积分文献，不能由搜索未命中推定首创。当前可确认项目具体推导与标准性质的归属，文献原创性仍待近邻逐式比较。完整判断见[理论增量审查](../01_IDEA/theoretical_increment_audit_20261020.md)。

## 2026-10-20 复审后的引用范围说明

本轮重读缓存原文与既有出版方核验记录，保持15项参考文献身份。重要性抽样的目标事件定位与完整分布计算是不同计算用途，不据此判定前者缺乏完整分布能力。混合代理和尾部重算文献用于说明限值附近的潮流预测偏差，输出重构文献用于说明有限统计信息与密度形式，极值接入容量文献用于说明输入出力尾部及其解耦线性化潮流假设。GMM已有非高斯、多峰和相关分布能力保留，TMM的必要性限定于较长联合电压尾部下的概率表示与解析计算扩展。

此次修订没有重新开展全库检索，也没有提高原文读取深度的标记。对应的原句问题、代码范围与实际Sobol计算路径见[复审记录](../02_PAPER/revision/introduction_review_20261020.md)。下方检索范围及逐篇核验记录继续作为来源依据。

检索日期：2026-10-20。研究对象为概率潮流中的电压、线路功率尾部分布、小概率越限事件及相应风险概率的计算。本文引言不再复述一般概率潮流方法分类。本记录对应 `02_PAPER/introduction_zh.md` 与 `02_PAPER/abstract_zh.md`。

## 检索范围与方法

检索结合PowerLit的IEEE TPWRS、IEEE TSG、中国电机工程学报及IJEPES缓存索引、互联网原始期刊页面、作者机构库、公开论文、Crossref出版方元数据和Europe PMC全文。年份不限，重点补查2024—2026年；以2026-10-20为检索时点。文献既按题名检索，也按摘要、引言和可得全文中的尾部、稀有事件与越限术语检索，并追查核心论文参考文献。

核心检索式包括：

- `probabilistic power flow rare event importance sampling tail`
- `probabilistic load flow tail extreme value`
- `probabilistic power flow tail distribution 2025 2026`
- `probabilistic load flow tail accuracy`
- `probabilistic power flow extreme value theory`
- `probabilistic power flow tail Gaussian mixture`
- `voltage violation extreme value theory`
- `load flow generalized Pareto`
- `probabilistic power flow large deviations`
- `probabilistic load flow tail saddle point`
- `probabilistic power flow tail conditional value at risk`
- `概率潮流 尾部 小概率`
- `电压越限 极值理论`

随后逐项检索已发现论文的完整题名、DOI、作者公开稿及机构库。Nature Communications 2026论文的参考文献进一步指向2019年交叉熵风险计算、稀有事件并集重要性抽样以及2025年可扩展稀有事件评估。2020年混合代理模型全文又指向最大熵与Gram–Charlier分布重构比较研究。

纳入核心综述的条件是，原文明确针对尾部概率、运行越限小概率、尾部样本或密度重构中的尾部误差采取了具体处理。仅在背景提到风险，或仅使用常规概率潮流给出均值、方差的论文，不作为尾部计算进展的主要依据。调度优化中的CVaR目标、充裕性失负荷事件和动态连锁故障另作邻近研究，不直接代替电气量分布计算现状。

这里的完整检索指多组关键词、主流期刊、近期研究及引文追溯均已覆盖并记录，不声称穷尽所有已发表文献。可得全文、可得关键段落、仅可得摘要及仅核对书目信息分别标识。搜索结果中的二手摘要不单独承担正文方法论据。

## 直接用于正文的尾部计算文献

| 文献 | 具体计算对象及尾部处理 | 核验依据与读取范围 | 本文引用方式 |
|---|---|---|---|
| Leite da Silva、de Castro，TPWRS 2019，DOI:10.1109/TPWRS.2018.2869769 | 直接估计潮流输出目标区域的概率，以交叉熵调整重要性抽样分布，提高稀有越限事件采样效率 | PowerLit摘要与引言关键段落，Crossref核对34(2):1193–1202 | 说明小概率事件定向抽样已是成熟研究方向，不以普通蒙特卡洛慢作为本文独有动机 |
| Christianen等，WSC 2024，DOI:10.1109/WSC63780.2024.10838841 | 随机负荷使配电网电压超限，结合大偏差或交叉熵设计抽样，并判断负荷对应的潮流可行性 | [会议原始全文](https://informs-sim.org/wsc24papers/inv132.pdf)，下载12页，读取摘要、模型、重要性抽样与可行性判别段落 | 面向配电网电压越限的小概率计算，不将其简化为常规全分布模拟 |
| Wang等，JRSE 2024，DOI:10.1063/5.0177383 | 以径向基函数代理潮流辅助重要性抽样，估计小概率电压与线路越限 | 出版方提交的Crossref完整摘要与元数据；未声称读过全文 | 说明抽样效率与非线性潮流代理可以联合处理 |
| Xu等，TSG 2020，DOI:10.1109/TSG.2019.2940928 | 对低概率风险相关区域采用高精度模型重新计算，降低直接使用代理模型导致的尾部偏差 | [作者公开全文](https://par.nsf.gov/servlets/purl/10157204)，下载11页，读取问题定义、混合计算与结论 | 强调越限区域误差修正这一直接针对尾部的技术进展 |
| Tan、Zhao、Chen，TPWRS 2025，DOI:10.1109/TPWRS.2024.3435490 | 神经网络与向量高斯过程代理潮流，以分类器识别需要重算的低概率样本，兼顾拓扑变化和尾部精度 | [作者公开全文](https://www.osti.gov/servlets/purl/2526233)，下载15页，读取摘要、分类重采样、模型与结论；[PNNL摘要](https://www.pnnl.gov/publications/scalable-risk-assessment-rare-events-power-systems-uncertain-wind-generation-and-loads)；Crossref核对最终卷期40(2):1374–1388 | 按最终卷期记2025年，不能因DOI含2024写成最终2024年论文 |
| Tan等，Nature Communications 2026，DOI:10.1038/s41467-025-68241-y | 面向输电过载风险，借助交流潮流求解器重算过载阈值附近样本，修正代理预测偏差 | [出版方全文](https://www.nature.com/articles/s41467-025-68241-y)及[Europe PMC原文XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12891488/fullTextXML)，读取摘要、引言、相关方法与讨论；正式发表日期2026-01-05，17:1521 | 直接承认近期研究已专门改善过载尾部计算，不再写成现有研究普遍忽略尾部 |
| Huang等，Elektronika ir Elektrotechnika 2018，DOI:10.5755/j01.eie.24.3.20980 | 用C型Gram–Charlier重构改进电压与线路功率密度的尾部负值和偏差；输入采用GMM | [期刊原始全文](https://eejournal.ktu.lt/index.php/elt/article/view/20980/9442)，下载7页，读取输出分布重构及算例尾部段落；Crossref核对24(3):61–67 | 归入输出尾部密度重构，不能与本文联合电压GMM的作用混为一谈 |
| Singh等，EPSR 2025，DOI:10.1016/j.epsr.2025.111528 | 用最大熵输出密度重构改善Gram–Charlier尾部负密度问题，计及输入及线路状态不确定性 | [出版方摘要与引言片段](https://www.sciencedirect.com/science/article/pii/S0378779625001208)，Crossref核对244:111528；未声称完整读过全文 | 只用其直接处理尾部密度合法性的内容，不概述一般累积量框架 |
| Singh等，EPSR 2025，DOI:10.1016/j.epsr.2025.111606 | 在常规三点估计外增加尾部代表点，改善风电接入系统尾部精度 | [作者预印本摘要](https://ssrn.com/abstract=4935783)、作者机构公开记录及Crossref，核对245:111606；预印本2024与期刊2025分开 | 只归纳尾部代表点设置，不再单列一般点估计法发展史 |
| Ma等，IJEPES 2025，DOI:10.1016/j.ijepes.2025.110963 | 用GEV表示分布式电源出力尾部，结合解耦线性化潮流及电压、热约束的越限概率要求，解析估计接入容量 | [出版方摘要及要点](https://www.sciencedirect.com/science/article/pii/S0142061525005113)，Crossref核对171:110963；未声称读过全文 | 说明专门拟合极端尾部再计算电压和潮流风险的解析研究已经存在，区别在实际采用的输出对象和潮流关系 |

## 邻近研究、候选及取舍

| 文献 | 核验状态 | 取舍及原因 |
|---|---|---|
| Li等，TPWRS 2025，DOI:10.1109/TPWRS.2025.3527251，稀有事件子集模拟敏感性 | PowerLit摘要、问题定义及引言关键段落；Crossref卷期页已核 | 其对稀有事件的处理直接相关，但重点是输入分布参数对事件概率的敏感性。纳入检索记录，正文不为覆盖方法名称而增加敏感性分支 |
| Jin等，Computing Rare Probabilities of Voltage Collapse，arXiv:2604.03807 | [2026-04-04公开预印本及HTML全文](https://arxiv.org/html/2604.03807v1)，摘要及问题定义可读；Crossref最终期刊卷期为EPSR 264:113793，2027年3月 | 以大偏差理论及最可能失稳源荷组合估计崩溃概率，属于相邻的稳定边界计算。本次检索按2026预印本身份记录，不把2027卷期当作2026已出版期刊文献 |
| Sun等，SPIES 2022，DOI:10.1109/SPIES55999.2022.10082652 | Crossref核对作者、题名、日期及64–68页；原摘要与全文未取得 | 题名直接涉及极值理论电压越限，但仅凭题名不能归纳其尾部模型。正文改用方法摘要已由出版方核实的Ma等2025年论文 |
| Choi等，TSTE 2026，DOI:10.1109/TSTE.2026.3714770 | Crossref题名作者及2026-07-17登记已核；[作者机构2026-07-20录用消息](https://devcms.yonsei.ac.kr/ee_en/community/news.do?article.offset=0&articleLimit=10&articleNo=475334&mode=view)确认身份；原始方法摘要及全文未取得 | 相关题名涉及分模态KDE与高斯混合Copula。保留为需进一步读取的近邻，不将二手网站中的越限概率/CVaR改善数字写进正文，也不据此声称输入尾部依赖研究不存在 |
| Owen等，Electronic Journal of Statistics 2019，稀有事件并集重要性抽样 | [作者预印本](https://arxiv.org/abs/1710.06965)与[Stanford机构记录](https://statistics.stanford.edu/technical-reports/importance-sampling-union-rare-events-application-power-systems-analysis) | 针对多约束稀有事件并集，示例为高斯相角差越限。作为抽样路线的补充，正文已用更贴近交流潮流及配电网电压对象的文献 |
| Chan等，Adaptive Monte Carlo Methods for Estimating Rare Events in Power Grids，2025，DOI:10.1061/AJRUA6.RUENG-1404 | [原始期刊摘要](https://ascelibrary.org/doi/10.1061/AJRUA6.RUENG-1404)，Crossref核对11(1):04024082 | 自适应稀有事件方法的横向比较纳入搜索覆盖，未完整读取其电网失效定义，不据其摘要推断完整交流PDF精度 |
| Mohammadi等，TSG 2018，DOI:10.1109/TSG.2017.2671740，鞍点近似 | PowerLit摘要、引言与[作者机构摘要](https://repository.lsu.edu/eecs_pubs/594/) | 主要提出非参数分布求解；本次未取得足够的直接尾部改进证据，不再因方法名称而列入主综述 |
| Williams、Crawford，TPWRS 2013，最大熵与Gram–Charlier重构比较 | 通过Xu等2020全文参考文献查到书目信息；本次未完整读取原文 | 保留追溯线索，不将后续论文对它的归纳替代原文核验。正文用已读输出尾部段落或出版方摘要的2018、2025年研究 |
| Maximum entropy based probabilistic load flow calculation for power system integrated with wind power generation，MPCE 2018，DOI:10.1007/s40565-018-0384-6 | [期刊全文](https://link.springer.com/article/10.1007/s40565-018-0384-6)的摘要与尾部段落可得 | 直接讨论尾部负密度问题，属于密度重构的先例。为压缩正文，与2018年CGC和2025年最大熵文献合并考虑，不逐篇罗列 |
| Huang等，EPSR 2024，DOI:10.1016/j.epsr.2024.110995，改进CGC与静态安全 | 原始期刊摘要和题录已定位 | 对尾部密度合法性研究形成补充，正文保留方法角色更清楚且已经读取关键段落的文献 |
| Guerra等，Energies 2026，DOI:10.3390/en19122777，相关输入与PCE概率潮流 | 原始期刊摘要及涉及尾部分位的段落已定位 | 计及尾部分位作为验证指标，不再把仅报告尾部指标与专门改变尾部计算机制等同。本版不增加常规PCE研究史 |
| Nijhuis等，TPWRS 2017，DOI:10.1109/TPWRS.2016.2628959 | PowerLit引言与作者机构摘要已核 | 原稿曾引用，属于GMM概率潮流基础；本版不再重述低压网负荷GMM发展史 |

Gao等2024年GMM输入输出解析概率潮流、吴涵2026年高斯电压二次型计算及Peel、McLachlan 2000年标准t混合，分别作为本文最近计算基础和统计模型依据，不包装为专门的小概率事件方法。天气背景文献仅支持光伏区域爬坡及台风切出等物理过程，不承担TMM尾部优势证明。

## 检索后可以成立的归纳

现有研究已针对越限区域定向抽样、尾部代理误差修正、极值尾部表示和尾部密度合法性采取具体改进。本文不能写成现有概率潮流只关心均值方差，或尚无人准确计算小概率风险。

越限事件概率、完整密度的尾部形状、分位误差和越限后果是不同计算对象。稀有事件也不等于重尾分布；高斯输入和近高斯输出同样可出现极小概率事件。本文已有比较直接支持电气量分布误差，不能借稀有事件文献扩写成已经验证极小概率高风险事件估计。

TMM的切入应落在联合电压模型的分量表示与交流功率分布计算。GMM已有非高斯、多峰和相关分布表示能力；当概率集中区域之外仍存在较长尾部时，不同分量形式对主体和尾部的近似及所需分量数可能不同。t分量提供可比较的长尾表示，但较长尾部不自动意味着更准确的真实尾部。

本文解析性对应联合电压模型给定后的电气量分布表达式，保留交流功率二次关系，解析计算条件概率并采用数值积分求其余部分。现有天气结果采用加权Sobol传播，保持原证据身份。摘要结果仅表述整体精度相当和部分对象、尾侧及指标的已有收益，不报告未验证的超低概率风险精度、总加速或调度优化收益。

## 原始资料位置

公开论文的文本提取仅用于文献阅读，缓存位于 `cache/tail_literature_20261020/`，包含Xu2020、Tan2025、Huang2018、Christianen2024的原PDF与提取文本，以及部分Crossref元数据。Nature 2026由Europe PMC原文XML核验。缓存可重新取得，不作为科学结果或论文正文。所有源荷、潮流、拟合、传播与误差计算均未在本轮启动。
