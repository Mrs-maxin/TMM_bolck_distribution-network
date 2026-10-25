"""Analytic distribution of electrical quantities for AQGMM, standard TMM and the grouped t mixture with mean shift
(method_zh.md section 4.5, appendix A).

Squared voltage magnitudes, sending-end branch P and Q and the system loss are quadratic in the voltage. Given a
component and a strength the voltage is Gaussian, so each quantity has a closed-form conditional characteristic
function; the strength is integrated (grouped t: Gauss-Legendre over the weather-situation variable U on [0, 1],
refined towards strong weather; standard TMM: generalized Gauss-Laguerre), components are weighted and the
cumulative distribution follows from Gil-Pelaez inversion.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import stats
from scipy.special import roots_genlaguerre, gammaincinv

import grouped_strength_tmm_case33 as gs
base = gs.base
STRENGTH_NODES = 40
U_PIECES = [(0., .9), (.9, .99), (.99, .999), (.999, 1.)]
U_NODES = 12


def ac_quantities(v):
    return np.c_[abs(v[:, 1:])**2, base.evaluate_outputs(v, base.BASE, base.BUS, base.BRANCH)]


def quadratic_forms(data):
    """Coefficients of every quantity in the fitting coordinates xi (x = center + T xi)."""
    q = lambda x: ac_quantities(base.voltage(np.atleast_2d(x)))
    d = 64; eye = np.eye(d); c0 = q(np.zeros(d))[0]; plus = q(eye); minus = q(-eye); nq = len(c0)
    b = (plus-minus)/2; A = np.zeros((nq, d, d)); A[:, np.arange(d), np.arange(d)] = ((plus+minus)/2-c0).T
    for i in range(d-1):
        off = (q(eye[i]+eye[i+1:])-plus[i]-plus[i+1:]+c0)/2; A[:, i, i+1:] = off.T; A[:, i+1:, i] = off.T
    T = data['basis']*data['scale']; center = data['center']
    probe = center+np.random.default_rng(1).normal(scale=.02, size=(50, d))
    check = float(abs(q(probe)-(np.einsum('ni,qij,nj->nq', probe, A, probe)+probe@b+c0)).max())
    At = np.einsum('ia,qij,jb->qab', T, A, T); bt = 2*np.einsum('ia,qij,j->qa', T, A, center)+b.T@T
    ct = np.einsum('i,qij,j->q', center, A, center)+b.T@center+c0
    return dict(A=At, b=bt, c=ct, check=check)


def u_nodes():
    p, w = leggauss(U_NODES); nodes, weights = [], []
    for a, b in U_PIECES:
        nodes.append(a+(b-a)*(p+1)/2); weights.append((b-a)*w/2)
    return np.concatenate(nodes), np.concatenate(weights)


def t_nodes(nu):
    if not np.isfinite(nu) or nu > 1e3:
        return np.array([1.]), np.array([1.])
    if nu > 300:
        p, w = leggauss(STRENGTH_NODES); p = (p+1)/2
        return 1/(2*gammaincinv(nu/2, p)/nu), w/w.sum()
    u, w = roots_genlaguerre(STRENGTH_NODES, nu/2-1)
    return nu/(2*u), w/w.sum()


def gaussian_states(model, kind):
    """Yield (weight, mean, covariance) in xi coordinates for every component and strength node."""
    if kind in ('GMM', 'TMM'):
        for k, pi in enumerate(model['weights']):
            nodes, weights = (np.array([1.]), np.array([1.])) if kind == 'GMM' else t_nodes(model['df'][k])
            for s, w in zip(nodes, weights):
                yield pi*w, model['means'][k], s*model['scatter'][k]
        return
    Q1, Q2 = model['Q1'], model['Q2']; Q = np.c_[Q1, Q2]; un, uw = u_nodes(); total = sum(c['weight'] for c in model['shifted'])
    for c in model['shifted']:
        grids = np.stack([np.ones(len(un)) if t.gauss else t.nu/stats.chi2.isf(un*t.pmax, t.nu)/t.mean for t in c['trunc']])
        G = c['coefficient']
        for i, w in enumerate(uw):
            s = grids[c['gid'], i]; mu_u = c['mu']+c['shift']*(s-1); D = np.sqrt(s); Su = D[:, None]*c['scatter']*D[None, :]
            mean = Q1@mu_u+Q2@(c['intercept']+G@mu_u)
            cov = Q@np.block([[Su, Su@G.T], [G@Su, G@Su@G.T+c['residual']]])@Q.T
            yield c['weight']/total*w, mean, cov


def terms(model, kind, forms):
    """Per quantity: arrays (weight, m0, lambda, beta) over all Gaussian states."""
    nq = len(forms['c']); out = [[] for _ in range(nq)]
    for w, mean, cov in gaussian_states(model, kind):
        vals, vecs = np.linalg.eigh(cov); L = vecs*np.sqrt(np.clip(vals, 0, None))
        for q in range(nq):
            A = forms['A'][q]; lin = 2*A@mean+forms['b'][q]
            lam, U = np.linalg.eigh(L.T@A@L); out[q].append((w, mean@A@mean+forms['b'][q]@mean+forms['c'][q], lam, U.T@(L.T@lin)))
    return out


def cdf_at(tm, thresholds):
    w = np.array([t[0] for t in tm]); m0 = np.array([t[1] for t in tm]); lam = np.array([t[2] for t in tm]); beta = np.array([t[3] for t in tm])
    mean_k = m0+lam.sum(1); var_k = 2*(lam**2).sum(1)+(beta**2).sum(1)
    mean = w@mean_k; sd = np.sqrt(max(w@(var_k+mean_k**2)-mean**2, 1e-300))
    keep = w > 1e-14; tmax = 12/np.sqrt(var_k[keep]/sd**2).min()
    nodes, weights = leggauss(64); edges = np.linspace(0, tmax, int(np.ceil(tmax*4))+2)
    tau = ((edges[1:]-edges[:-1])[:, None]*(nodes+1)/2+edges[:-1, None]).ravel(); wt = ((edges[1:]-edges[:-1])[:, None]*weights/2).ravel()
    t = tau/sd; phi = np.zeros(len(t), complex)
    for j in np.flatnonzero(keep):
        z = 1-2j*np.outer(t, lam[j])
        phi += w[j]*np.exp(1j*t*(m0[j]-mean)-.5*np.log(z).sum(1)-(np.outer(t**2, beta[j]**2)/(2*z)).sum(1))
    y = (np.asarray(thresholds)-mean)/sd
    return np.clip(.5-(np.imag(np.exp(-1j*np.outer(y, tau))*phi)/tau@wt)/np.pi, 0, 1), float(abs(phi[-1]))
