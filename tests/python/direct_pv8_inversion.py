"""Unclipped quadratic-form Fourier inversion for the direct-output diagnostic.

The Gaussian conditional characteristic function is unchanged. Composite
Gauss-Legendre integration checks both frequency truncation and panel order.
"""
import numpy as np
import hashlib
from numpy.polynomial.legendre import leggauss

_FFT_CACHE = {}


def moments(terms):
    w = np.array([v[0] for v in terms])
    m = np.array([v[1] + np.sum(v[2]) for v in terms])
    var = np.array([2*np.sum(v[2]**2)+np.sum(v[3]**2) for v in terms])
    mean = w @ m
    return mean, np.sqrt(w @ (var+(m-mean)**2)), var


def inverse_pdf(terms, thresholds, np=np):
    mean, sd, variance = moments(terms)
    assert np.isfinite(sd) and sd > 0
    y = (np.asarray(thresholds)-mean)/sd
    reduced = []
    for w, m, lam, beta in terms:
        # Preserve the Gaussian linear variance in directions of zero curvature.
        active = abs(lam) > max(sd*1e-13, 1e-30)
        reduced.append((w, (m-mean)/sd, lam[active]/sd,
                        beta[active]/sd, np.sum((beta[~active]/sd)**2)))

    def characteristic(t):
        out = np.zeros(len(t), complex)
        for w, m, lam, beta, linear_var in reduced:
            z = 1-2j*np.outer(t, lam)
            out += w*np.exp(1j*t*m-.5*np.log(z).sum(1)
                            -.5*t*t*(linear_var+(beta*beta/z).sum(1)))
        return out

    if max(len(v[2]) for v in reduced) > 4:
        # Loss has many positive curvatures and slow Fourier decay. A uniform
        # Fourier sum resolves its narrow peak without a quadratic dense matrix.
        signature = hashlib.sha256()
        for term in terms:
            for part in term:
                signature.update(np.asarray(part).tobytes())
        return fft_inverse(characteristic, y, sd, signature.hexdigest())

    def integrate(tmax, order):
        nodes, weights = leggauss(order)
        # Resolve even the most distant requested threshold's phase.
        width = min(.5, 16/max(abs(y).max(), 1))
        edges = np.linspace(0, tmax, int(np.ceil(tmax/width))+1)
        pdf, cdf = np.zeros(len(y)), np.zeros(len(y))
        for panels in np.array_split(np.arange(len(edges)-1), max(1, int(tmax/32))):
            tau = (edges[panels, None]+(edges[panels+1]-edges[panels])[:, None]*(nodes+1)/2).ravel()
            wt = ((edges[panels+1]-edges[panels])[:, None]*weights/2).ravel()
            phi = characteristic(tau)
            for start in range(0, len(y), 128):
                value = np.exp(-1j*np.outer(y[start:start+128], tau))*phi
                pdf[start:start+128] += value.real@wt/np.pi
                cdf[start:start+128] -= value.imag@(wt/tau)/np.pi
        return pdf/sd, cdf+.5

    tmax = max(12., 12*sd/np.sqrt(variance).min())
    history, previous = [], None
    for attempt in range(12):
        pdf, cdf = integrate(tmax, 24)
        endpoint = float(abs(characteristic(np.linspace(.9*tmax, tmax, 33))).max())
        change_pdf = float('inf') if previous is None else float(abs(pdf-previous[0]).max()*sd)
        change_cdf = float('inf') if previous is None else float(abs(cdf-previous[1]).max())
        history.append(dict(frequency_max=float(tmax), cf_endpoint=endpoint,
                            minimum_pdf=float(pdf.min()), raw_cdf_min=float(cdf.min()),
                            raw_cdf_max=float(cdf.max()), scaled_pdf_change=change_pdf,
                            cdf_change=change_cdf))
        if previous is not None and endpoint < 1e-9 and change_pdf < 2e-7 and change_cdf < 2e-8:
            check_pdf, check_cdf = integrate(tmax, 40)
            order_pdf = float(abs(pdf-check_pdf).max()*sd)
            order_cdf = float(abs(cdf-check_cdf).max())
            if order_pdf < 2e-7 and order_cdf < 2e-8:
                assert pdf.min()*sd > -2e-7 and cdf.min() > -2e-8 and cdf.max() < 1+2e-8
                return check_pdf, check_cdf, dict(
                    frequency_max=float(tmax), cf_endpoint=endpoint,
                    minimum_pdf=float(check_pdf.min()), raw_cdf_min=float(check_cdf.min()),
                    raw_cdf_max=float(check_cdf.max()), frequency_history=history,
                    scaled_pdf_order_error=order_pdf, cdf_order_error=order_cdf,
                    clipping=False, smoothing=False, curvature_tolerance_relative=1e-13)
        previous = pdf, cdf
        tmax *= 2
        if tmax > 32768:
            break
    raise RuntimeError(f'Fourier quadrature did not converge: {history}')


def fft_inverse(characteristic, points, sd, signature):
    from scipy.fft import rfft, irfft
    history = []
    previous = None
    length = max(128., 4*abs(points).max())
    cache_key = signature, length
    first_power = 15
    if cache_key in _FFT_CACHE:
        grid, density, cdf, info = _FFT_CACHE[cache_key]
        from scipy.interpolate import CubicSpline
        pdf_at=np.interp(points,grid,density)/sd
        cdf_at=np.interp(points,grid,cdf)
        interpolation_pdf=float(abs(CubicSpline(grid,density)(points)/sd-pdf_at).max()*sd)
        interpolation_cdf=float(abs(CubicSpline(grid,cdf)(points)-cdf_at).max())
        if interpolation_pdf<2e-5 and interpolation_cdf<2e-6:
            return pdf_at,cdf_at,dict(info,reused_fourier_grid=True,
                interpolation_pdf_check=interpolation_pdf,interpolation_cdf_check=interpolation_cdf)
        # A denser requested curve can resolve a peak missed by the earlier
        # query points. Check the next Fourier resolution at the new points.
        previous=pdf_at,cdf_at
        first_power=int(np.log2(len(grid)))+1
    for power in range(first_power, 23):
        n = 2**power
        t = 2*np.pi*np.arange(n//2+1)/length
        phi = np.concatenate([characteristic(v) for v in np.array_split(t, max(1,len(t)//2048))])
        phase = np.exp(1j*t*length/2)
        # irfft uses the positive exponential; conjugate the CF coefficients.
        density = irfft(np.conj(phi*phase), n=n)*n/length
        anti = np.zeros_like(phi)
        anti[1:] = phi[1:]*phase[1:]/(-1j*t[1:])
        primitive = irfft(np.conj(anti), n=n)*n/length
        grid = -length/2+np.arange(n)*length/n
        cdf = (grid+length/2)/length+primitive-primitive[0]
        pdf_at = np.interp(points,grid,density)/sd
        cdf_at = np.interp(points,grid,cdf)
        dp = float('inf') if previous is None else float(abs(pdf_at-previous[0]).max()*sd)
        dc = float('inf') if previous is None else float(abs(cdf_at-previous[1]).max())
        record = dict(nodes=n, period_sd=length, cf_endpoint=float(abs(phi[-1])),
                      minimum_pdf=float(density.min()/sd), raw_cdf_min=float(cdf.min()),
                      raw_cdf_max=float(cdf.max()), scaled_pdf_change=dp,cdf_change=dc,
                      total_mass=float(np.sum(density)*length/n))
        history.append(record)
        print(f'Fourier grid nodes={n} period={length:g} scaled_pdf_change={dp:.3g} cdf_change={dc:.3g}',flush=True)
        if previous is not None and dp < 2e-5 and dc < 2e-6 and abs(phi[-1]) < 1e-9:
            # The much wider period tests contamination from periodic copies.
            n2=2*n; length2=2*length
            t2=2*np.pi*np.arange(n2//2+1)/length2
            phi2=np.concatenate([characteristic(v) for v in np.array_split(t2,max(1,len(t2)//2048))])
            phase2=np.exp(1j*t2*length2/2)
            d2=irfft(np.conj(phi2*phase2),n=n2)*n2/length2
            anti2=np.zeros_like(phi2);anti2[1:]=phi2[1:]*phase2[1:]/(-1j*t2[1:])
            primitive2=irfft(np.conj(anti2),n=n2)*n2/length2
            grid2=-length2/2+np.arange(n2)*length2/n2
            c2=(grid2+length2/2)/length2+primitive2-primitive2[0]
            alias_pdf=float(abs(np.interp(points,grid2,d2)/sd-pdf_at).max()*sd)
            alias_cdf=float(abs(np.interp(points,grid2,c2)-cdf_at).max())
            assert alias_pdf<2e-5 and alias_cdf<2e-6, (alias_pdf,alias_cdf)
            assert density.min()>-2e-7
            info=dict(record,frequency_history=history,method='uniform Fourier sum',
                scaled_pdf_order_error=dp,cdf_order_error=dc,alias_pdf=alias_pdf,alias_cdf=alias_cdf,
                clipping=False,smoothing=False)
            _FFT_CACHE[cache_key] = grid,density,cdf,info
            return pdf_at,cdf_at,info
        previous=pdf_at,cdf_at
    raise RuntimeError(f'Uniform Fourier inversion did not converge: {history}')
