> 2026-10-26 拆分说明：本稿写于两篇论文拆分前，归入标准TMM论文作为起点；其中涉及分组t或分块TMM的表述需删去，并按`shared/result/heavytail_injection/`与`shared/result/actual_output/`两组算例重写【待补充】。

> 2026-10-20草稿，按当时的标准TMM思路撰写，待按[method_zh.md](method_zh.md)的三种模型和两个核心论点重写；参考文献[1]—[16]的编号被method_zh.md沿用。

# 0 引言

极端天气会改变新能源出力与负荷需求的组合，进而影响配电网受电、反送和沿线电压。大范围云团遮蔽和移出可引起区域光伏减发与恢复[1]，台风强风下的风机切出则可能使本地供电能力大幅下降[2]。常见源荷波动与出现频率较低的大幅变化并存时，电压、线路功率及网损可能呈现非高斯尾部分布。交流概率潮流需要在计算分布主体的同时，准确描述这些大偏差及其发生概率，为运行越限和高损耗水平分析提供依据。

非高斯尾部的计算同时受到概率分布与网络电气关系的影响。同一源荷变化会影响多个节点电压，线路有功和无功取决于相关电压的幅值与相角组合。交流功率包含电压变量的乘积，总有功损耗又与各线路电流平方有关，因此相关电压同时出现较大偏差的概率会影响功率尾部。从联合电压出发进行解析计算，需要在描述非高斯尾部的同时，保留节点电压相关性及功率关于电压的二次关系。

已有基于样本的概率潮流研究，从尾部有效样本和电气量计算偏差两方面改进尾部概率估计。重要性抽样及交叉熵方法调整源荷抽样分布，增加电压或线路功率处于指定尾部区域的有效样本，再通过概率权重估计相应概率[3–4]。代理潮流模型能够减少逐样本交流潮流的求解次数[5]，其尾部预测偏差则通过混合代理和自适应重采样加以修正，对需要重算的样本调用原始潮流模型[6–8]。这些进展改善了尾部区域的数值估计，解析分布计算则进一步关注如何由概率模型和交流功率关系求得相应分布。

面向概率分布的解析近似，已有尾部研究重点改进统计信息和密度形式。C型Gram–Charlier展开与最大熵方法改善了常规分布展开的尾部负密度问题[9–10]，增加尾部代表点补充了较大源荷偏差对应的潮流信息[11]。极值分布也被用于表示电源出力尾部，并结合解耦线性化潮流分析电压和线路约束下的接入容量[12]。有限阶统计量相近的分布仍可具有不同尾部；采用线性化潮流时，电压偏差乘积项被省略，可能影响较大电压偏差下的功率及网损尾部。因此，尾部分布的解析计算需要同时明确概率表示及其对应的交流功率关系。

为同时描述分布形态和变量相关性，高斯混合模型（Gaussian mixture model，GMM）采用不同均值和协方差的分量表示多峰、偏斜及相关分布。已有输入输出联合建模研究结合分段线性潮流，给出了相关非高斯输入下的分布解析解[13]。高斯联合电压下的二次型研究则保留原始交流电压到功率关系，建立了电压与功率分布之间的解析联系[14]。将GMM用于联合电压表示，可以在各高斯分量内保留交流功率二次关系，再按分量权重合成功率分布，为兼顾相关性与功率非线性的分布计算提供基础。

当联合电压分布主体较集中、尾部延伸较远时，GMM可通过不同尺度的分量逼近分布，有限分量下需要兼顾集中区域与较远尾部的计算精度。多元t分量具有共同随机尺度，其自由度影响分布的尾部衰减。在均值和协方差存在且保持相同的条件下，多元t分量仍可表示不同的集中程度和外侧概率，自由度增大时趋近高斯分布[15]。这些性质使t混合模型（t mixture model，TMM）能够在分量内表示常见电压波动与较大联合偏差，为改善电气量尾部分布计算及减少相应精度所需的分量提供了模型依据。

线路电流由复电压线性计算，多元t的线性变换性质使全网模型能够给出线路所需的电压、电流联合分布。已有条件t分布理论给出条件自由度及尺度随其余变量取值的变化关系[16]，可用于计算电压幅值和线路功率的阈值概率。由此，TMM的联合尾部表示能够与交流功率二次关系结合，为电气量分布的解析计算提供依据。

本文针对极端天气下交流概率潮流的非高斯尾部分布，提出基于TMM的配电网交流概率潮流计算方法。建立全网电压实部、虚部的联合TMM，利用t分量对集中部分及较远尾部的表示，计算相关电压偏差经原交流关系形成的电气量分布。推导电压幅值及单线路有功、无功功率的解析条件概率，再通过剩余积分和分量加权求得累积分布。总有功损耗由同一全网联合电压计算，保留线路损耗之间的相关性。模型分布的数值比较采用分量加权Sobol积分，通过原交流关系计算各积分点的电压幅值、线路功率及总有功损耗。

## 参考文献

[1] WELLBY S J, ENGERER N A. Categorizing the Meteorological Origins of Critical Ramp Events in Collective Photovoltaic Array Output[J]. Journal of Applied Meteorology and Climatology, 2016, 55(6): 1323–1344. DOI: [10.1175/JAMC-D-15-0107.1](https://doi.org/10.1175/JAMC-D-15-0107.1).

[2] 葛晓琳, 严鑫, 符杨, 等. 基于降尺度台风模拟的海上风电有序避险时空优化调度策略[J]. 中国电机工程学报, 2025, 45(19): 7575–7588. DOI: [10.13334/j.0258-8013.pcsee.240911](https://doi.org/10.13334/j.0258-8013.pcsee.240911).

[3] LEITE DA SILVA A M, DE CASTRO A M. Risk Assessment in Probabilistic Load Flow via Monte Carlo Simulation and Cross-Entropy Method[J]. IEEE Transactions on Power Systems, 2019, 34(2): 1193–1202. DOI: [10.1109/TPWRS.2018.2869769](https://doi.org/10.1109/TPWRS.2018.2869769).

[4] CHRISTIANEN M, LAM H, VLASIOU M, et al. Importance Sampling of Rare Events for Distribution Networks with Stochastic Loads[C]//2024 Winter Simulation Conference. 2024: 3590–3601. DOI: [10.1109/WSC63780.2024.10838841](https://doi.org/10.1109/WSC63780.2024.10838841).

[5] WANG C, ZHOU Y, PENG Y, et al. Efficient surrogate-assisted importance sampling for rare event assessment in probabilistic power flow[J]. Journal of Renewable and Sustainable Energy, 2024, 16(1): 015501. DOI: [10.1063/5.0177383](https://doi.org/10.1063/5.0177383).

[6] XU Y, KORKALI M, MILI L, et al. Risk Assessment of Rare Events in Probabilistic Power Flow via Hybrid Multi-Surrogate Method[J]. IEEE Transactions on Smart Grid, 2020, 11(2): 1593–1603. DOI: [10.1109/TSG.2019.2940928](https://doi.org/10.1109/TSG.2019.2940928).

[7] TAN B, ZHAO J, CHEN Y. Scalable Risk Assessment of Rare Events in Power Systems With Uncertain Wind Generation and Loads[J]. IEEE Transactions on Power Systems, 2025, 40(2): 1374–1388. DOI: [10.1109/TPWRS.2024.3435490](https://doi.org/10.1109/TPWRS.2024.3435490).

[8] TAN B, YE K, ZHAO J, et al. Computationally efficient tail distribution-aware large-scale power system overloading risk assessment[J]. Nature Communications, 2026, 17: 1521. DOI: [10.1038/s41467-025-68241-y](https://doi.org/10.1038/s41467-025-68241-y).

[9] HUANG Y, XU Q, JIANG X, et al. An Analytic Approach to Probabilistic Load Flow Incorporating Correlation Between Non-Gaussian Random Variables[J]. Elektronika ir Elektrotechnika, 2018, 24(3): 61–67. DOI: [10.5755/j01.eie.24.3.20980](https://doi.org/10.5755/j01.eie.24.3.20980).

[10] SINGH V, MOGER T, JENA D. Maximum entropy based probabilistic load flow for assessing input uncertainties and line outages in wind-integrated power systems[J]. Electric Power Systems Research, 2025, 244: 111528. DOI: [10.1016/j.epsr.2025.111528](https://doi.org/10.1016/j.epsr.2025.111528).

[11] SINGH V, MOGER T, JENA D. A modified point estimate-based probabilistic load flow approach for improving tail accuracy in wind-integrated power systems[J]. Electric Power Systems Research, 2025, 245: 111606. DOI: [10.1016/j.epsr.2025.111606](https://doi.org/10.1016/j.epsr.2025.111606).

[12] MA X, WU H, YUAN Y, et al. Analytical first-screening of distributed generation hosting capacity via extreme-value theory[J]. International Journal of Electrical Power & Energy Systems, 2025, 171: 110963. DOI: [10.1016/j.ijepes.2025.110963](https://doi.org/10.1016/j.ijepes.2025.110963).

[13] GAO Y, XU X, YAN Z, et al. Analytical Probabilistic Power Flow and Global Sensitivity Analysis of Distribution Systems Based on Gaussian Mixture Model of Input-Output Variables[J]. IEEE Transactions on Power Systems, 2024, 39(3): 5283–5296. DOI: [10.1109/TPWRS.2023.3329197](https://doi.org/10.1109/TPWRS.2023.3329197).

[14] 吴涵. 基于随机变量二次型的配电网交流概率潮流建模和解析计算方法(一)：从电量到功率量的推导[J]. 中国电机工程学报, 2026, 46(3): 928–941. DOI: [10.13334/j.0258-8013.pcsee.242015](https://doi.org/10.13334/j.0258-8013.pcsee.242015).

[15] PEEL D, MCLACHLAN G J. Robust mixture modelling using the t distribution[J]. Statistics and Computing, 2000, 10(4): 339–348. DOI: [10.1023/A:1008981510081](https://doi.org/10.1023/A:1008981510081).

[16] DING P. On the Conditional Distribution of the Multivariate t Distribution[J]. The American Statistician, 2016, 70(3): 293–295. DOI: [10.1080/00031305.2016.1164756](https://doi.org/10.1080/00031305.2016.1164756).
