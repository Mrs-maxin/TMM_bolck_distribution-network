"""Audit fixed bus-18 reactive demand in the fitted joint voltage distributions."""
import pickle
import numpy as np
import pandas as pd
import run_direct_heavytail_case33 as r

def main():
    rows=[]
    for scene in r.SCENES:
        d=r.OUT/scene;a=np.load(d/'estimation.npz');v=a['voltage']
        power=v*np.conj(v@r.base.Y.T)*r.base.BASE
        target=-r.base.BUS[17,3]
        features=pd.read_csv(d/'MC_electrical_features.csv')
        qsd=features[features.object=='Q17'].sd.iloc[0]
        rows.append(dict(scene=scene,model='fast_mapping',node=18,Q_injection_target_Mvar=target,
            Q_injection_RMS_error_Mvar=float(np.sqrt(np.mean((power[:,17].imag-target)**2))),MC_Q17_sd_Mvar=qsd))
        with (d/'models_K2.pkl').open('rb') as f:models=pickle.load(f)
        c=dict(np.load(d/'voltage_coordinates.npz'))
        for name,model in models.items():
            if name=='grouped_t_shift':z,w=r.sh.sample_shift(model,65536,np.random.default_rng(2026102679))
            else:z,w=r.base.propagate_any(model,65536,2026102679)
            volt=r.base.voltage((z*c['scale'])@c['basis'].T+c['center'])
            power=volt*np.conj(volt@r.base.Y.T)*r.base.BASE;error=power[:,17].imag-target
            rows.append(dict(scene=scene,model=name,node=18,Q_injection_target_Mvar=target,
                Q_injection_RMS_error_Mvar=float(np.sqrt(w@(error**2)/w.sum())),MC_Q17_sd_Mvar=qsd))
    pd.DataFrame(rows).to_csv(r.OUT/'fixed_reactive_injection_check_K2.csv',index=False)

if __name__=='__main__':main()
