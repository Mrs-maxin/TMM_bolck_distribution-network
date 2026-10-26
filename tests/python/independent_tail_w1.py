"""W1 of separately probability-trimmed, normalized distributions, in physical units."""
import numpy as np
from numpy.polynomial.legendre import leggauss


def tail_w1(model_quantile,reference,a,b,model_cdf=None):
 reference=np.sort(np.asarray(reference));n=len(reference)
 jumps=np.arange(int(np.floor(a*n))+1,int(np.ceil(b*n)))/n
 edges=np.unique(np.r_[a,jumps[(jumps>a)&(jumps<b)],b])
 if model_cdf is not None:
  mid=(edges[:-1]+edges[1:])/2;value=reference[np.ceil(n*mid).astype(int)-1];cross=np.asarray(model_cdf(value));keep=(cross>edges[:-1])&(cross<edges[1:]);edges=np.unique(np.r_[edges,cross[keep]])
 def integrate(order):
  z,w=leggauss(order);width=np.diff(edges);p=edges[:-1,None]+width[:,None]*(z+1)/2;ac=reference[np.clip(np.ceil(n*p).astype(int)-1,0,n-1)];q=model_quantile(p.ravel()).reshape(p.shape)
  return float(np.sum(width[:,None]*w/2*abs(q-ac)))
 history=[];previous=None;tolerance=max(1e-10,1e-7*np.std(reference))
 for order in [8,16,32,64,128]:
  integral=integrate(order);value=integral/(b-a);change=float('inf') if previous is None else abs(value-previous);history.append(dict(order=order,unnormalized_integral=integral,normalized_W1=value,change=change))
  if previous is not None and change<tolerance:return value,dict(window=[a,b],empirical_inverse='ceil(N p)-1',segments=len(edges)-1,unnormalized_integral=integral,normalized_W1=value,refinement_difference=change,absolute_tolerance=tolerance,history=history)
  previous=value
 raise RuntimeError(f'W1 integration unresolved: {history}')
