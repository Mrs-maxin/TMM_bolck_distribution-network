"""Build the viewer from the explicitly named per-object comparison tables."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]/'result/heavytail_injection/case33bw_20261026'

def main():
    c=pd.read_csv(ROOT/'results/tail_cdf_rmse_method_compare.csv')
    q=pd.read_csv(ROOT/'results/extreme_quantile_error_method_compare.csv')
    assert len(c)==3104 and len(q)==9312
    w=pd.read_csv(ROOT/'results/tail_quantile_w1_method_compare.csv')
    m=pd.read_csv(ROOT/'results/full_mean_variance_method_compare.csv')
    keys=['情形','K','电气量','类型','尾侧']
    w=w.merge(c[keys+['MC尾部延伸','电气量说明']],on=keys,validate='1:1')
    flags=c.groupby(['情形','K','电气量','类型'],as_index=False).agg(MC尾部延伸=('MC尾部延伸','any'))
    m=m.merge(flags,on=['情形','K','电气量','类型'],validate='1:1')
    desc=pd.read_csv(ROOT/'results/quantity_ids_units.csv')[['电气量','说明']].rename(columns={'说明':'电气量说明'})
    m=m.merge(desc,on='电气量',validate='m:1');m['尾侧']='全分布'
    s=pd.read_csv(ROOT/'tail_distribution/focus_quantities_mc_selection.csv');sel=set(zip(s.情形,s.电气量,s.尾侧));global_sel=set(zip(s.情形,s.电气量))
    for table in [c,q,w]:table['本轮MC重点']=[(x.情形,x.电气量,x.尾侧) in sel for x in table.itertuples()]
    m['本轮MC重点']=[(x.情形,x.电气量) in global_sel for x in m.itertuples()]
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>IEEE33逐电气量尾部误差</title>
<style>body{font:15px/1.7 system-ui,"PingFang SC",sans-serif;max-width:1550px;margin:24px auto;padding:0 18px;color:#263238}h1{font-size:24px}.note{background:#f2f6f9;border-left:4px solid #859eb1;padding:12px}.controls{display:flex;flex-wrap:wrap;gap:12px;margin:20px 0}label{display:flex;flex-direction:column}select,input,button{font:inherit;padding:6px;border:1px solid #bbc5cb;background:#fff;border-radius:4px}.scroll{overflow:auto}table{border-collapse:collapse;min-width:1100px;width:100%;font-variant-numeric:tabular-nums}th,td{padding:9px 12px;border-bottom:1px solid #e3e7ea;white-space:nowrap;text-align:right}thead th{background:#e8eff6}tbody tr:nth-child(even){background:#f7f9fb}a{color:#31638c}.pager{display:flex;gap:18px;margin:16px 0}</style>
<h1>IEEE33：尾部精度与全分布均值方差</h1>
<p>每行比较同一注入情形、同一分量数K、同一电气量和同一尾侧。误差越小，这一项越准确。本页没有全分布CDF-RMSE。</p>
<p class="note">分组t在部分电气量上更准、部分更差。参数一致性未通过与某项误差更小可以同时出现。这里的分组t是含均值移动和截断的已有实现；标准TMM在Pareto、混合K=1时触及自由度下限。</p>
<p>尾部CDF-RMSE使用每侧0.1%—5%的50档MC阈值，单位为百分点。尾部分位区间W1使用[0.001,0.05]或[0.95,0.999]，单位为p.u.、kW或kvar。均值方差针对全分布，电压为幅值。各指标分别比较。</p>
<p><a href="tail_distribution/README.md">本轮全部节点与线路尾部分析</a> · <a href="tail_distribution/figure_atlas.md">全分布PDF与尾部放大图册</a> · <a href="results/tail_quantile_w1_method_compare.csv">W1比较表</a> · <a href="results/full_mean_variance_method_compare.csv">均值方差比较表</a></p>
<p><a href="README.md">阅读说明</a> · <a href="results/tail_cdf_rmse_method_compare.csv">尾部CDF-RMSE比较</a> · <a href="results/extreme_quantile_error_method_compare.csv">极端分位误差比较</a> · <a href="results/tail_cdf_per_threshold.csv">逐阈值CDF及概率误差</a> · <a href="error_metrics.md">误差标准</a></p>
<div class="controls"><label>指标<select id="mode"><option value="cdf">尾部CDF-RMSE</option><option value="q">极端分位数误差</option><option value="w1">尾部分位区间W1</option><option value="mean">全分布均值</option><option value="variance">全分布方差</option></select></label>
<label>电气量范围<select id="scope"><option value="new">本轮MC形状分析重点</option><option value="focus">原MC尾部延伸标记</option><option value="all">全部电气量</option></select></label>
<label>注入情形<select id="scene"><option>全部</option><option>Pareto</option><option>对数正态</option><option>对数逻辑斯蒂</option><option>混合</option></select></label>
<label>分量数K<select id="k"><option>全部</option><option>1</option><option>2</option><option>4</option><option>6</option></select></label>
<label>电气量<select id="kind"><option value="全部">全部</option><option value="V">节点电压</option><option value="P">支路有功</option><option value="Q">支路无功</option><option value="loss">有功网损</option></select></label>
<label>尾侧<select id="side"><option>全部</option><option>下尾</option><option>上尾</option></select></label>
<label>尾概率<select id="p"><option value="0.001">0.1%</option><option value="0.005">0.5%</option><option value="0.01">1%</option></select></label>
<label>节点或支路<input id="object" placeholder="V18、P17或Q17"></label></div>
<p id="caption"></p><div class="scroll"><table><thead id="head"></thead><tbody id="body"></tbody></table></div><div class="pager"><button id="prev">上一页</button><span id="count"></span><button id="next">下一页</button></div>
<p>“分组t减AQGMM”是误差之差：负值表示分组t误差较小，正值表示较大。排序表示当前数值大小，不表示差异具有统计显著性。尾部延伸标记来自既有MC，不能单凭此标记证明严格渐近重尾。</p>
<script>const data=__DATA__;let page=0,filtered=[],size=50;const el=id=>document.getElementById(id);const fmt=v=>typeof v==='number'?Number(v).toPrecision(7):String(v??'');
function select(){const mode=el('mode').value,scene=el('scene').value,k=el('k').value,kind=el('kind').value,side=el('side').value,obj=el('object').value.trim().toUpperCase(),moment=['mean','variance'].includes(mode),scope=el('scope').value;el('p').disabled=mode!=='q';el('side').disabled=moment;filtered=data[mode].filter(r=>(scope==='all'||(scope==='new'?r['本轮MC重点']:r['MC尾部延伸']))&&(scene==='全部'||r['情形']===scene)&&(k==='全部'||r.K===Number(k))&&(kind==='全部'||r['类型']===kind)&&(moment||side==='全部'||r['尾侧']===side)&&(!obj||r['电气量'].toUpperCase().includes(obj))&&(mode!=='q'||Math.abs(r['尾概率']-Number(el('p').value))<1e-10));filtered.sort((a,b)=>a['情形'].localeCompare(b['情形'],'zh')||a.K-b.K||a['类型'].localeCompare(b['类型'])||(parseInt(a['电气量'].slice(1))||0)-(parseInt(b['电气量'].slice(1))||0)||a['尾侧'].localeCompare(b['尾侧']));page=0;render();}
function render(){const mode=el('mode').value;let cols=[['情形','情形'],['K','分量数K'],['电气量说明','电气量'],['尾侧','尾侧']];if(mode==='cdf'){for(const m of ['AQGMM','标准TMM','分组t'])cols.push([m+'_尾部CDF_RMSE_百分点',m+'尾部CDF-RMSE']);cols.push(['分组t减AQGMM_百分点','分组t减AQGMM'],['分组t减标准TMM_百分点','分组t减标准TMM'],['误差数值由小到大','误差从小到大']);el('caption').textContent='尾部CDF-RMSE，单位：百分点。这里是50档尾部概率误差，不是全分布CDF误差。';}else if(mode==='w1'){for(const m of ['AQGMM','标准TMM','分组t'])cols.push([m,m+'尾部分位区间W1']);cols.push(['单位','单位']);el('caption').textContent='同一尾部分位区间的W1，越小越准确。不是包含概率0或1端点的完整尾部W1。';}else if(mode==='mean'){cols.push(['MC均值','MC均值']);for(const m of ['AQGMM','标准TMM','分组t'])cols.push(['方法均值_'+m,m+'均值'],['均值误差_'+m,m+'均值误差']);cols.push(['单位','单位']);el('caption').textContent='全分布均值；电压为幅值，支路功率保留真实方向。';}else if(mode==='variance'){cols.push(['MC方差','MC方差']);for(const m of ['AQGMM','标准TMM','分组t'])cols.push(['方法方差_'+m,m+'方差'],['方差比_方法除以MC_'+m,m+'方差/MC']);cols.push(['方差单位','方差单位']);el('caption').textContent='全分布方差；方差/MC等于1表示一致，没有乘100。';}else{cols.push(['分位水平','分位水平'],['MC分位值','MC分位值']);for(const m of ['AQGMM','标准TMM','分组t'])cols.push(['方法分位值_'+m,m+'分位值'],['分位误差_'+m,m+'带符号误差'],['绝对分位误差_'+m,m+'绝对误差']);cols.push(['单位','单位'],['绝对误差数值由小到大','绝对误差从小到大']);el('caption').textContent='分位误差＝方法分位值−MC分位值；绝对误差越小越准确。';}el('head').replaceChildren();const tr=document.createElement('tr');for(const[key,label]of cols){const th=document.createElement('th');th.textContent=label;tr.appendChild(th);}el('head').appendChild(tr);el('body').replaceChildren();for(const row of filtered.slice(page*size,(page+1)*size)){const tr=document.createElement('tr');for(const[key]of cols){const td=document.createElement('td');td.textContent=fmt(row[key]);tr.appendChild(td);}el('body').appendChild(tr);}const total=Math.max(1,Math.ceil(filtered.length/size));el('count').textContent=`共${filtered.length}行，第${page+1}/${total}页`;el('prev').disabled=page===0;el('next').disabled=page+1>=total;}
for(const id of ['mode','scope','scene','k','kind','side','p','object'])el(id).addEventListener(id==='object'?'input':'change',select);el('prev').onclick=()=>{if(page>0){page--;render();}};el('next').onclick=()=>{if((page+1)*size<filtered.length){page++;render();}};select();</script></html>'''
    payload=json.dumps({'cdf':c.to_dict('records'),'q':q.to_dict('records'),'w1':w.to_dict('records'),'mean':m.to_dict('records'),'variance':m.to_dict('records')},ensure_ascii=False,separators=(',',':'),allow_nan=False)
    (ROOT/'results_browser.html').write_text(page.replace('__DATA__',payload.replace('</','<\\/')))
    print(json.dumps(dict(tail_CDF_comparison_rows=len(c),quantile_comparison_rows=len(q),focus_CDF_rows=int(c.MC尾部延伸.sum())),ensure_ascii=False))

if __name__=='__main__':main()
