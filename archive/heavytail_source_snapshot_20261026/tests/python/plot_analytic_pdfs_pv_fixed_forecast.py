"""Diagnostic full and tail-zoomed PDFs with analytically computed model densities.

Model densities follow from inverting the characteristic function of each voltage quadratic
(tests/analytic_limit_risk_pv_fixed_forecast.py). The AC curve is the density of the AC power-flow
solutions of every source-load scenario. Both use the same Gaussian smoothing width h: the AC
curve as a kernel estimate, the model curve by multiplying its characteristic function by
exp(-h^2 t^2 / 2), which is the exact density of the same smoothing. Not formal paper figures.
"""
import importlib.util, sys, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from numpy.polynomial.legendre import leggauss

REPO = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('risk', REPO/'tests/python/analytic_limit_risk_pv_fixed_forecast.py')
risk = importlib.util.module_from_spec(spec); spec.loader.exec_module(risk)
D = risk.D; OUT = D/'diagnostic_analytic_pdfs'; OUT.mkdir(exist_ok=True); CACHE = OUT/'cache'; CACHE.mkdir(exist_ok=True)
KS = [1, 3, 6, 12]
FAMILIES = [('GMM', '--', 'tab:blue', 'GMM'), ('TMM', '-', 'tab:red', 'TMM'), ('BTMM', '-.', 'tab:green', 'Block TMM')]
if (D/'selected_models_multiblock.csv').exists():
    FAMILIES.append(('MBTMM', ':', 'tab:purple', 'Two-block TMM'))
KS = [int(k) for k in os.environ.get('PV_PDF_K', '1,3,6,12').split(',')]
# (file name, quantity index, axis label, tail of interest, value transform from quantity to displayed unit)
OBJECTS = [('V18', 16, 'Node 18 voltage magnitude (p.u.)', 'upper: overvoltage'),
           ('V29', 27, 'Node 29 voltage magnitude (p.u.)', 'lower: undervoltage'),
           ('V31', 29, 'Node 31 voltage magnitude (p.u.)', 'lower: undervoltage'),
           ('V33', 31, 'Node 33 voltage magnitude (p.u.)', 'lower: undervoltage'),
           ('P12', 32+11, 'Branch 12 sending-end active power (MW)', 'lower: larger reverse flow'),
           ('P13', 32+12, 'Branch 13 sending-end active power (MW)', 'lower: larger reverse flow'),
           ('P19', 32+18, 'Branch 19 sending-end active power (MW)', 'upper: reverse flow to import'),
           ('P31', 32+30, 'Branch 31 sending-end active power (MW)', 'upper: reverse flow to import'),
           ('loss', 64, 'System active power loss (MW)', 'both')]
if os.environ.get('PV_PDF_OBJECTS_FROM_TARGETS'):
    impacts = {'undervoltage_side': 'lower: undervoltage', 'overvoltage_side': 'upper: overvoltage',
               'reverse_flow_increase': 'lower: larger reverse flow', 'reversal_reverse_to_import': 'upper: reverse flow to import',
               'reversal_import_to_reverse': 'lower: import to reverse flow'}
    OBJECTS = []
    for _, t in pd.read_csv(D/'targets.csv').iterrows():
        if t.output == 'V':
            OBJECTS.append((f'V{t.node}', int(t.node)-2, f'Node {t.node} voltage magnitude (p.u.)', impacts.get(t.operating_impact, t.side)))
        else:
            OBJECTS.append((f'P{t.branch}', 32+int(t.branch)-1, f'Branch {t.branch} sending-end active power (MW)', impacts.get(t.operating_impact, t.side)))
    OBJECTS.append(('loss', 64, 'System active power loss (MW)', 'both'))
ref_q = risk.ac_quantities(np.load(D/'reference_AC.npz')['voltage'])


def pdf_on_grid(terms, grid_q, h):
    """Density of the quantity (in its own units) smoothed by a Gaussian of width h."""
    w = np.array([t[0] for t in terms]); m0 = np.array([t[1] for t in terms])
    lam = np.array([t[2] for t in terms]); beta = np.array([t[3] for t in terms])
    mean_k = m0+lam.sum(1); var_k = 2*(lam**2).sum(1)+(beta**2).sum(1)
    mean = w@mean_k; sd = np.sqrt(w@(var_k+mean_k**2)-mean**2)
    keep = w > 1e-14; tmax = min(12/np.sqrt(var_k[keep]/sd**2).min(), 12*sd/h)
    nodes, weights = leggauss(64); edges = np.linspace(0, tmax, int(np.ceil(tmax))+2)
    tau = ((edges[1:]-edges[:-1])[:, None]*(nodes+1)/2+edges[:-1, None]).ravel()
    wt = ((edges[1:]-edges[:-1])[:, None]*weights/2).ravel()
    t = tau/sd; phi = np.zeros(len(t), complex)
    for j in np.flatnonzero(keep):
        z = 1-2j*np.outer(t, lam[j])
        phi += w[j]*np.exp(1j*t*(m0[j]-mean)-.5*np.log(z).sum(1)-(np.outer(t**2, beta[j]**2)/(2*z)).sum(1))
    phi *= np.exp(-.5*(h*t)**2)
    y = (grid_q-mean)/sd; out = np.empty(len(y))
    for i in range(0, len(y), 200):
        out[i:i+200] = np.real(np.exp(-1j*np.outer(y[i:i+200], tau))*phi)@wt
    return out/(np.pi*sd)


rows = []
for name, q, xlabel, tail in OBJECTS:
    voltage = name.startswith('V')
    r = np.sqrt(ref_q[:, q]) if voltage else ref_q[:, q]; n = len(r)
    iqr = np.subtract(*np.quantile(r, [.75, .25])); h = .9*min(r.std(), iqr/1.349)*n**-.2
    lo, hi = r.min()-4*h, r.max()+4*h; grid = np.linspace(lo, hi, 2401); dx = grid[1]-grid[0]
    hist, _ = np.histogram(r, np.r_[grid-dx/2, grid[-1]+dx/2]); ac = gaussian_filter1d(hist/(n*dx), h/dx, mode='constant', truncate=5)
    windows = {'Full distribution': (lo, hi), 'Lower tail (AC 0.1%-5%)': tuple(np.quantile(r, [.001, .05])),
               'Upper tail (AC 95%-99.9%)': tuple(np.quantile(r, [.95, .999]))}
    for k in KS:
        curves = {}
        for family, _, _, _ in FAMILIES:
            cache = CACHE/f'{name}_{family}_K{k}.npy'
            if cache.exists():
                curves[family] = np.load(cache); continue
            sel = risk.selected[(risk.selected.K == k) & (risk.selected.family == family)].iloc[0]
            model = dict(np.load(D/'candidates'/(sel.candidate+'.npz')))
            terms = risk.component_terms(model, family, q)
            if voltage:
                # The quadratic is |V|^2; smooth in |V| units by mapping the density of |V|^2 through |V| = sqrt(.).
                hq = 2*np.median(r)*h; dens_sq = pdf_on_grid(terms, grid**2, hq); dens = dens_sq*2*grid
            else:
                dens = pdf_on_grid(terms, grid, h)
            np.save(cache, dens); curves[family] = dens
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        for ax, (title, (a, b)) in zip(axes, windows.items()):
            m = (grid >= a) & (grid <= b)
            ax.plot(grid[m], ac[m], 'k-', lw=1.8, label='AC power flow')
            for family, style, color, label in FAMILIES:
                ax.plot(grid[m], curves[family][m], style, color=color, lw=1.4, label=label)
            top = max(ac[m].max(), *(c[m].max() for c in curves.values()))
            ax.set_xlim(a, b); ax.set_ylim(0, 1.08*top); ax.set_title(title); ax.grid(alpha=.3); ax.set_xlabel(xlabel)
        axes[0].set_ylabel('Probability density'); axes[0].legend(fontsize=8)
        fig.suptitle(f'{xlabel}, K = {k}; tail of interest: {tail}', fontsize=11)
        fig.text(.5, .01, 'Model densities by characteristic-function inversion; AC density from AC power flow of every scenario; '
                 'same smoothing width for all curves; diagnostic only', ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .04, 1, .95)); fig.savefig(OUT/f'{name}_K{k}_analytic_pdf.png', dpi=130); plt.close(fig)
        for family in curves:
            rows.append(dict(object=name, K=k, family=family, smoothing_width=h,
                             integral_on_grid=float(curves[family].sum()*dx)))
    print('done', name, flush=True)
pd.DataFrame(rows).to_csv(OUT/'density_integral_check.csv', index=False)
