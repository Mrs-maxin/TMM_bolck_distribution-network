"""Summarize the extreme-weather tests at equal K (case_study_plan_20261022.md section 5.2)."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]/'result/case33bw/extreme_weather_case33_20261022'
LABEL = {'GMM': 'AQGMM', 'TMM': '标准TMM', 'BTMM': '分块TMM', 'SBTMM': '分块TMM共用残差'}


def main():
    rows = []
    for d in sorted(ROOT.glob('E*_p*')):
        if not (d/'summary_same_K.csv').exists():
            continue
        s = pd.read_csv(d/'summary_same_K.csv'); info = json.loads((d/'run_info.json').read_text())
        targets = json.loads((d/'targets_locked.json').read_text())['targets']
        for _, r in s.iterrows():
            rows.append(dict(场景=d.name, K=int(r.K), 模型=LABEL[r.family], 似然=round(r.validation_ll, 3),
                             目标尾段误差=r.target_tail_err_median, 带符号=r.target_tail_signed_median,
                             深尾概率=r.target_beyond_01_median, 电压KS=r.V_ks_median, 支路有功KS=r.P_ks_median))
        print(d.name, '| groups', info['split'].get('groups'), '| MC voltage kurtosis (node 2, 33)', [round(x, 1) for x in info['MC_voltage_excess_kurtosis']],
              '| targets', [f"{t['output']}{t['node'] or t['branch']}-{t['side']}" for t in targets])
    t = pd.DataFrame(rows); t.to_csv(ROOT/'summary_all.csv', index=False)
    pd.set_option('display.width', 220)
    for col in ['目标尾段误差', '电压KS', '深尾概率']:
        print(f'\n=== {col}（目标对象中位数；深尾概率应为0.001）')
        print(t.pivot_table(index=['场景', 'K'], columns='模型', values=col).round(4).to_string())


if __name__ == '__main__':
    main()
