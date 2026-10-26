"""Check the analytic distribution computation against model sampling on one operating condition.

For AQGMM, standard TMM and the grouped t mixture with mean shift at several K: cumulative probabilities of all 97
electrical quantities at the MC-method tail quantiles (0.1%-5% both sides) and at 401 quantiles over the whole
range, computed analytically and from 262144 model voltages; reports the largest difference and the time.
Usage: python check_analytic_grouped_t_case33.py S1 [K ...]
"""
import os, sys, json, time
if len(sys.argv) >= 2:
    os.environ['MAIN_SCENE'] = sys.argv[1]
KS = [int(k) for k in sys.argv[2:]] or [1, 2, 4]
os.environ['MAIN_SHIFT'] = '1'
sys.argv = sys.argv[:1]
import numpy as np
import pandas as pd
import mainline_comparison_case33 as mc
import analytic_grouped_t_case33 as an

OUT = mc.REPO/'result/case33bw/analytic_check_20261024'/mc.SCENE


def main():
    OUT.mkdir(parents=True, exist_ok=True); data = mc.load(); rng = np.random.default_rng(2026102450); ref = mc.reference()
    split, Q1, Q2 = mc.chk.split(data['train'], data['x'], data['basis'], data['scale'], rng, mc.SCENE, group_target=data['every'], return_q=True)
    caps = mc.ts.strength_cap(data['every'], split['groups']); merged = [np.array(g) for g in split['group_cols']]
    t0 = time.perf_counter(); forms = an.quadratic_forms(data); t_forms = time.perf_counter()-t0
    grid = np.linspace(0, 1, 401); rows = []
    for k in KS:
        gmm, _ = mc.fit_aqgmm(data['x'], data['xv'], k, data['scale'])
        std, _, _ = mc.mm.moment_models(data['x'], gmm, Q1, Q2)
        prop = mc.sh.shift_grouped_model(data['x'], gmm, Q1, Q2, merged, caps)
        for name, kind, m in [('AQGMM', 'GMM', gmm), ('标准TMM', 'TMM', std), ('分组t（均值移动）', 'SHIFT', prop)]:
            y, w = mc.outputs(m, data, 2026102460+k); y = y.copy(); y[:, :32] = y[:, :32]**2
            t1 = time.perf_counter(); tm = an.terms(m, kind, forms); diffs, cf_tail = [], 0.
            for j, c in enumerate(ref):
                r = c['r']; n = len(r)
                pts = np.r_[r[(mc.LEVELS*n).astype(int)], r[((1-mc.LEVELS)*n).astype(int)], r[np.minimum((grid*(n-1)).astype(int), n-1)]]
                if j < 32:
                    pts = pts**2
                Fa, tail = an.cdf_at(tm[j], pts); cf_tail = max(cf_tail, tail)
                o = np.argsort(y[:, j]); Fs = np.interp(pts, y[o, j], np.cumsum(w[o])/w.sum(), left=0., right=1.)
                diffs.append(np.abs(Fa-Fs).max())
            rows.append(dict(model=name, K=k, max_abs_difference=float(max(diffs)), median_object_max_difference=float(np.median(diffs)),
                             analytic_seconds=time.perf_counter()-t1, cf_at_end=cf_tail))
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    t = pd.DataFrame(rows); t.to_csv(OUT/'analytic_vs_sampling.csv', index=False)
    (OUT/'run_info.json').write_text(json.dumps(dict(quadratic_reconstruction_max_error=forms['check'], quadratic_forms_seconds=t_forms,
        u_pieces=an.U_PIECES, u_nodes_per_piece=an.U_NODES, laguerre_nodes=an.STRENGTH_NODES, draws=mc.gs.DRAWS), indent=1))
    print(t.round(5).to_string(index=False)); print('quadratic reconstruction', forms['check'], 'forms seconds', round(t_forms, 1))


if __name__ == '__main__':
    main()
