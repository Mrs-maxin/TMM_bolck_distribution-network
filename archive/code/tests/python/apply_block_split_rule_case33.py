"""Write the block split of derivation.md section J into a case directory as block_subspace.npz.

Weather groups are judged on all injection scenarios of the case (training, validation, reference);
the voltage directions and the leak check use the training voltages only. The file replaces the
hand-set split read by base.pv_voltage_subspace, so the block and shared phases fit on this split.
Usage: python apply_block_split_rule_case33.py <case output dir> <scene>
"""
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np
import block_split_rule_check as chk


def main():
    out, scene = Path(sys.argv[1]), sys.argv[2]; d = out/scene
    assert not (d/'block_subspace.npz').exists(), 'split already present; remove it explicitly to recompute'
    f = np.load(d/'fitting_inputs.npz')
    train = np.load(d/'training_inputs.npz')['target']
    every = np.r_[train, np.load(d/'validation_inputs.npz')['target'], np.load(d/'reference_inputs.npz')['target']]
    row, Q1, Q2 = chk.split(train, f['x'], f['basis'], f['scale'], np.random.default_rng(2026102261), scene,
                            group_target=every, return_q=True)
    assert np.allclose(np.c_[Q1, Q2].T@np.c_[Q1, Q2], np.eye(64), atol=1e-9)
    chk.base.save_npz(d/'block_subspace.npz', Q1=Q1, Q2=Q2, sensitivity=np.zeros((64, 0)))
    row.update(rule='derivation.md section J', injection_scenarios_for_groups=len(every), script=Path(__file__).name)
    chk.base.write_json(d/'block_subspace_diagnostics.json', row)
    print(json.dumps(row, ensure_ascii=False))


if __name__ == '__main__':
    main()
