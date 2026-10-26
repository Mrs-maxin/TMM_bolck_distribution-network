"""Read existing arrays only. Never import generation, AC, fit or propagation code."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
WIND = ROOT / 'restored/typhoon_v2/typhoon_comparison_K136_20261019_v2'
PV = ROOT / 'restored/pv_tables'
SCENES = ['weak_local', 'regional_passage', 'W0', 'W1', 'W2', 'W3']
Q = np.array([.005, .01, .025, .05, .25, .5, .75, .95, .975, .99, .995])
QN = ['q005','q01','q025','q05','q25','q50','q75','q95','q975','q99','q995']
OBJECTS = [('V', n, 0, 'p.u.') for n in range(2,34)]
OBJECTS += [(o, 0, b, 'MW' if o == 'P' else 'MVAr') for o in ['P','Q'] for b in range(1,33)]
OBJECTS += [('loss', 0, 0, 'MW')]
RULE = '''探索性AC-only规则v1；在读取方法误差前保存。
所有97输出均扫描；上下尾分别处理；网损独立，不决定节点和线路。
经验分位使用逆ECDF（inverted_cdf），不作IID置信区间。
近确定：std不超过V的1e-8或P/Q/损耗的1e-7及相对尺度1e-10的较大者；不参与排序。
IQR小于0.05std时，IQR标准化指标记缺失，禁止极小IQR放大排序。
形态线索：该尾(q50-q005或q995-q50)/std超过正态2.575829的1.15倍；或朝该尾偏态绝对值>0.5；或有零点质量>=1%；或24箱平滑直方图峰形线索>=2。
峰形线索只作描述：24箱、sigma=1、峰突出度0.15最大箱高、峰距3；不作为严格多峰或厚尾证明。
每场景每类V/P/Q每尾优先取形态分数最高的1个；同类候选间|r|>=0.995且线性残差std<=0.05原std视为近比例重复，继续向后取不重复者。
形态分数=max(该尾标准差跨度/(2.575829*1.15),朝该尾偏态/0.5,峰形数/2,零质量/0.01)，只有满足形态线索者入选。
同分按物理尾跨度、编号依次排序；不预设旧节点。通常最多6个节点/线路，加独立网损一项。
paired参考为regional_passage对weak_local，以及W1/W2/W3对W0；W0和weak_local为基线，不虚构无天气反事实。
同期源荷、开停和潮流方向用于解释固定候选；不以方法胜负改选。尾部形态不等同数学渐近厚尾。
风电365点留出参考决定候选与方法比较；2044点AC只作补充形态，不改写参考误差。极端0.5%分位仅约2个顺序观测。
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_arrays(scene):
    if scene.startswith('W'):
        inp = dict(np.load(WIND/'source_reuse'/scene/'inputs.npz', allow_pickle=False))
        ac = dict(np.load(WIND/'source_reuse'/scene/'standard_AC.npz', allow_pickle=False))
        ref = dict(np.load(WIND/scene/'reference_outputs.npz', allow_pickle=False))
        rows = np.flatnonzero(inp['split'] == 'reference')
        assert np.array_equal(rows, ref['source_rows'])
        assert len(rows) == 365
    else:
        mapping = json.loads((OUT/'pv_mapping.json').read_text())
        inp = dict(np.load(mapping[scene+'/reference_inputs.npz'], allow_pickle=False))
        ac = dict(np.load(mapping[scene+'/reference_AC.npz'], allow_pickle=False))
        ref = dict(np.load(mapping[scene+'/reference_outputs.npz'], allow_pickle=False))
        rows = np.arange(6272)
        assert np.unique(inp['process_id']).size == 128
        assert np.all(np.unique(inp['process_id'], return_counts=True)[1] == 49)
        assert inp['seed'].item() == 2026101927 + 209759
        assert (ac['iterations'] >= 0).all()
    full = np.c_[abs(ac['voltage'][:,1:]), ac['outputs']]
    assert np.isfinite(full).all()
    assert ref['values'].shape == (len(rows),97)
    assert np.max(abs(full[rows]-ref['values'])) < 1e-10
    return full[rows], full, rows, inp, ac


def feature(x):
    quant = np.quantile(x, Q, method='inverted_cdf')
    sd = x.std()
    iqr = quant[6]-quant[4]
    out = dict(zip(QN,quant))
    out.update(n=len(x),mean=x.mean(),std=sd,iqr=iqr,minimum=x.min(),maximum=x.max(),
               zero_mass=np.mean(x == 0),max_atom_mass=np.unique(x,return_counts=True)[1].max()/len(x),
               unique_values=len(np.unique(x)),skewness=skew(x) if sd > 0 else np.nan,
               excess_kurtosis=kurtosis(x) if sd > 0 else np.nan)
    counts, _ = np.histogram(x,bins=24)
    smooth = gaussian_filter1d(counts.astype(float),1)
    peaks = find_peaks(np.r_[0,smooth,0],prominence=.15*smooth.max(),distance=3)[0]
    out['histogram_peak_hints'] = len(peaks)
    for side,span in [('lower',quant[5]-quant[0]),('upper',quant[-1]-quant[5])]:
        out[side+'_span'] = span
        out[side+'_std_extension'] = span/sd if sd > 0 else np.nan
        out[side+'_iqr_extension'] = span/iqr if iqr > max(.05*sd,0) else np.nan
    return out


def main():
    (OUT/'selection_rules.txt').write_text(RULE,encoding='utf-8')
    rule_hash = sha(OUT/'selection_rules.txt')
    branches = pd.read_csv(OUT/'branch_map.csv').set_index('branch')
    data = {}
    audits = []
    features = []
    full_features = []
    for scene in SCENES:
        ref,full,rows,inp,ac = load_arrays(scene)
        data[scene] = (ref,full,rows,inp,ac)
        processes = inp['process_id'][rows]
        audits.append(dict(scene=scene,reference_rows=len(ref),all_rows=len(full),reference_processes=len(np.unique(processes)),
                           reference_values_match_AC=True,split_proof='NPZ split + source_rows' if scene.startswith('W') else 'reference seed +128x49; training/validation arrays not restored; disjointness protocol only'))
        for j,(output,node,branch,unit) in enumerate(OBJECTS):
            meta=dict(scene=scene,object_index=j,output=output,node=node,branch=branch,unit=unit,
                      from_bus=int(branches.loc[branch,'from_bus']) if branch else 0,
                      to_bus=int(branches.loc[branch,'to_bus']) if branch else 0)
            f=feature(ref[:,j]);f.update(meta)
            f['near_deterministic'] = f['std'] <= max(1e-8 if output=='V' else 1e-7,1e-10*max(1,abs(f['mean'])))
            f['negative_fraction']=float(np.mean(ref[:,j]<0))
            features.append(f)
            if scene.startswith('W'):
                full_features.append(dict(**meta,**feature(full[:,j])))
        print(f'AC-only扫描 {scene}: {len(ref)}参考点，97对象',flush=True)
    table=pd.DataFrame(features)
    paired=[]
    for scene,base in [('regional_passage','weak_local'),('W1','W0'),('W2','W0'),('W3','W0')]:
        a=table[table.scene==scene].set_index('object_index')
        b=table[table.scene==base].set_index('object_index')
        for j in a.index:
            paired.append(dict(scene=scene,paired_base=base,object_index=j,output=a.loc[j,'output'],node=a.loc[j,'node'],branch=a.loc[j,'branch'],
                               **{col+'_delta':a.loc[j,col]-b.loc[j,col] for col in ['mean','q005','q995','lower_span','upper_span','skewness','excess_kurtosis','negative_fraction']}))
    table.to_csv(OUT/'AC_features_all_reference.csv',index=False)
    pd.DataFrame(full_features).to_csv(OUT/'AC_features_wind_all2044_supplement.csv',index=False)
    pd.DataFrame(paired).to_csv(OUT/'AC_paired_changes.csv',index=False)
    pd.DataFrame(audits).to_csv(OUT/'sample_audit.csv',index=False)
    all_selected=[]
    for scene in SCENES:
        selected=[]
        a=table[(table.scene==scene)&~table.near_deterministic].copy()
        for output in ['V','P','Q']:
            for side,direction in [('lower',-1),('upper',1)]:
                pool=a[a.output==output].copy()
                pool['shape_score']=np.maximum.reduce([pool[side+'_std_extension'].to_numpy()/(2.575829*1.15),
                    direction*pool.skewness.to_numpy()/.5,pool.histogram_peak_hints.to_numpy()/2,pool.zero_mass.to_numpy()/.01])
                pool=pool[pool.shape_score>1].sort_values(['shape_score',side+'_span','object_index'],ascending=[False,False,True])
                for _, row in pool.iterrows():
                    j=int(row.object_index)
                    duplicate=False
                    for other in selected:
                        if other['output']!=output:
                            continue
                        k=other['object_index']
                        if j==k:
                            other['tail_sides']='both' if other['tail_sides']!=side else side
                            duplicate=True
                            break
                        r=np.corrcoef(data[scene][0][:,j],data[scene][0][:,k])[0,1]
                        if abs(r)>=.995 and np.sqrt(max(0,1-r*r))<=.05:
                            duplicate=True
                            break
                    if duplicate:
                        continue
                    item=row.to_dict();item['tail_sides']=side;item['selection']='AC_shape_rule';selected.append(item)
                    break
        loss=table[(table.scene==scene)&(table.output=='loss')].iloc[0].to_dict()
        loss.update(tail_sides='both',selection='independent_loss',shape_score=np.nan)
        selected.append(loss)
        frame=pd.DataFrame(selected)
        path=OUT/f'candidates_{scene}.csv';frame.to_csv(path,index=False)
        print('候选已固定',scene,frame[['output','node','branch','tail_sides']].to_dict('records'),'SHA256',sha(path),flush=True)
        all_selected.extend(selected)
    candidates=pd.DataFrame(all_selected);candidates.to_csv(OUT/'candidates_fixed.csv',index=False)
    conditions=[]
    for row in all_selected:
        scene=row['scene'];j=int(row['object_index']);ref,full,rows,inp,ac=data[scene]
        src={k:inp[k].sum(1)[rows] for k in ['load_P_MW','load_Q_MVAr']}
        if scene.startswith('W'):
            for k in ['wind_group_P_MW','wind_group_Q_MVAr']:
                src[k]=inp[k].sum(1)[rows]
            src['connected_fraction']=inp['connected'].reshape(2044,6).mean(1)[rows]
            src['weather_load_multiplier']=inp['weather_load_multiplier'][rows]
        else:
            src['PV_P_MW']=inp['PV_P_MW'].sum(1)
            src['shading_fraction']=inp['shading_fraction'][:,:6].mean(1)
        src['loss_MW']=ref[:,-1]
        src['minimum_voltage_pu']=ref[:,:32].min(1)
        src['object_value']=ref[:,j]
        if row['branch']:
            branch=int(row['branch']);f=int(row['from_bus'])-1;t=int(row['to_bus'])-1
            src['branch_P_MW']=ref[:,32+branch-1]
            src['branch_Q_MVAr']=ref[:,64+branch-1]
            src['from_angle_minus_to_rad']=np.angle(ac['voltage'][rows,f]/ac['voltage'][rows,t])
        for tail,mask in [('all',np.ones(len(ref),bool)),('lower',ref[:,j]<=row['q05']),('upper',ref[:,j]>=row['q95'])]:
            conditions.append(dict(scene=scene,output=row['output'],node=row['node'],branch=row['branch'],tail=tail,n=int(mask.sum()),processes=len(np.unique(inp['process_id'][rows][mask])),
                                   **{k+'_mean':float(v[mask].mean()) for k,v in src.items()},
                                   branch_negative_P_fraction=float(np.mean(src['branch_P_MW'][mask]<0)) if 'branch_P_MW' in src else np.nan))
    pd.DataFrame(conditions).to_csv(OUT/'candidate_tail_source_conditions.csv',index=False)
    paths=[OUT/'selection_rules.txt',OUT/'AC_features_all_reference.csv',OUT/'candidates_fixed.csv']+[OUT/f'candidates_{s}.csv' for s in SCENES]
    lock={'stage':'AC_only_complete_before_method_error_read','selection_rule_sha256':rule_hash,'sha256':{p.name:sha(p) for p in paths},'new_source_samples':0,'new_AC_solves':0,'new_fits':0,'new_propagations':0}
    (OUT/'AC_ONLY_LOCK.json').write_text(json.dumps(lock,indent=2))
    print('AC_ONLY_LOCK',sha(OUT/'AC_ONLY_LOCK.json'),flush=True)


if __name__ == '__main__':
    main()
