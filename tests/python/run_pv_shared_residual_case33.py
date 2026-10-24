"""IEEE 33-bus fixed-forecast PV: block mixtures with one shared residual covariance (derivation B20-B22).

Protocol: 01_IDEA/plans/pv_shared_residual_protocol_20261021.md. The PV fixed-forecast runner is imported
unchanged; inputs regenerate from its seeds and the ac_only phase checks them against the original PV
run byte for byte. Phases: ac_only, fit, reference, block, shared, limits, compare.
"""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '1')
from pathlib import Path
import sys, json, time, argparse, traceback
sys.dont_write_bytecode = True
import numpy as np
import run_pv_fixed_forecast_case33 as base
import shared_residual_mixture as shared

REPO = base.REPO
PLAN = REPO/'01_IDEA/plans/pv_shared_residual_protocol_20261021.md'
PV_MAIN = REPO/'result/case33bw/pv_fixed_forecast_case33_20261021'
SCENE = 'common_strength'


def V(node):
    return node-2


LIMIT_EVENTS = {
    'V29 < 0.960': lambda y: y[:, V(29)] < .960,
    'V31 < 0.955': lambda y: y[:, V(31)] < .955,
    'V31 < 0.950': lambda y: y[:, V(31)] < .950,
    'any node < 0.950': lambda y: np.any(y[:, :32] < .950, axis=1),
    'V18 > 1.070': lambda y: y[:, V(18)] > 1.070,
    'V18 > 1.080': lambda y: y[:, V(18)] > 1.080,
    'V18 > 1.090': lambda y: y[:, V(18)] > 1.090,
    'nodes 13-18 all > 1.070': lambda y: np.all(y[:, [V(b) for b in range(13, 19)]] > 1.070, axis=1),
}
shared.install(base, base.pv_voltage_subspace, LIMIT_EVENTS)


def check_inputs_match_main():
    rows = {}
    for split in base.SPLITS:
        new = json.loads((base.O/SCENE/f'{split}_input_manifest.json').read_text())['SHA256']
        old = json.loads((PV_MAIN/SCENE/f'{split}_input_manifest.json').read_text())['SHA256']
        rows[split] = dict(SHA256=new, main_SHA256=old, identical=new == old)
    base.write_json(base.O/'inputs_match_main.json', rows)
    assert all(r['identical'] for r in rows.values()), 'regenerated PV inputs differ from the original PV run'


def save_shared_protocol():
    path = base.O/'shared_protocol.json'
    record = dict(plan=str(PLAN.relative_to(REPO)), plan_SHA256=base.sha(PLAN), script_SHA256=base.sha(Path(__file__)),
                  shared_module_SHA256=base.sha(Path(shared.__file__)), runner_SHA256=base.sha(Path(base.__file__)),
                  K_order=base.CONFIG['K_order'], limit_events=list(LIMIT_EVENTS), fixed_targets=str(PV_MAIN/SCENE/'targets.csv'))
    if path.exists():
        old = json.loads(path.read_text())
        assert old['runner_SHA256'] == record['runner_SHA256'], 'imported PV runner changed during the run'
        return
    base.write_json(path, record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['ac_only', 'fit', 'reference', 'block', 'shared', 'limits', 'compare'])
    parser.add_argument('--out', default=str(REPO/'result/case33bw/pv_shared_residual_case33_20261021'))
    parser.add_argument('--smoke', action='store_true', help='tiny counts to test the code path only')
    parser.add_argument('--K', type=int, help='fit one K; finish with a plain fit run to collect all candidates')
    parser.add_argument('--K-order', default='1,2,3,4,6,8,12')
    args = parser.parse_args()
    base.CONFIG['K_order'] = [int(k) for k in args.K_order.split(',')]
    base.O = Path(args.out); base.O.mkdir(parents=True, exist_ok=True)
    base.FIXED_TARGETS = PV_MAIN/SCENE/'targets.csv'
    base.apply_mode(args.smoke)
    try:
        save_shared_protocol()
        if args.phase == 'ac_only':
            base.phase_ac_only(SCENE, args.smoke)
            if not args.smoke:
                check_inputs_match_main()
            return
        {'fit': lambda: base.phase_fit(SCENE, args.K),
         'reference': lambda: base.phase_reference(SCENE),
         'block': lambda: base.phase_block(SCENE),
         'shared': lambda: shared.phase_shared(SCENE),
         'limits': lambda: shared.phase_limits(SCENE),
         'compare': lambda: shared.phase_compare(SCENE, 'pv_shared_residual_protocol section 4, fixed before the shared-residual fits')}[args.phase]()
    except Exception:
        base.write_json(base.O/f'failure_{args.phase}_{os.getpid()}.json',
                        dict(error=traceback.format_exc(), UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
        raise


if __name__ == '__main__':
    main()
