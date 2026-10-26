"""Exact quadratic moments and conditional CF terms; no fitted model is shared."""
import numpy as np
from scipy import stats
from scipy.special import roots_genlaguerre
from numpy.polynomial.legendre import leggauss
from independent_block_t_fit import CORE,geometry


def specs(g):
 base=float(g['base']);bus=g['bus'];br=g['branch'];Y,yf,yt=CORE['admittance'](base,bus,br);T=g['basis']*g['scale'];center=g['center'];rawB=np.vstack([np.zeros((1,64)),np.c_[np.eye(32),1j*np.eye(32)]]);v0=np.r_[1.,np.zeros(32)];out=[]
 def add(name,kind,R,o,A,b=None,c=0.,f=0,t=0):
  out.append(dict(name=name,kind=kind,R=R@T,o=R@center+o,A=A,b=np.zeros(len(A)) if b is None else b,c=c,from_bus=f,to_bus=t))
 for node in range(1,33):add(f'V{node+1}','V',np.stack([rawB[node].real,rawB[node].imag]),np.zeros(2),np.eye(2))
 for kind in ['P','Q']:
  for number,l in enumerate(np.flatnonzero(br[:,10]>0),1):
   f,t=br[l,:2].astype(int)-1;ib=yf[l]@rawB;i0=yf[l]@v0;R=np.stack([rawB[f].real,rawB[f].imag,ib.real,ib.imag]);o=np.array([v0[f],0.,i0.real,i0.imag]);A=np.zeros((4,4))
   if kind=='P':A[0,2]=A[2,0]=A[1,3]=A[3,1]=base/2
   else:A[1,2]=A[2,1]=base/2;A[0,3]=A[3,0]=-base/2
   add(f'{kind}{number}',kind,R,o,A,f=f+1,t=t+1)
 A=np.zeros((64,64));b=np.zeros(64);c=0.
 for l in np.flatnonzero(br[:,10]>0):
  h=yf[l]@rawB;h0=yf[l]@v0;r=base*br[l,2];A+=r*np.real(np.outer(h.conj(),h));b+=2*r*np.real(h0.conjugate()*h);c+=r*abs(h0)**2
 out.append(dict(name='loss',kind='loss',R=np.eye(64),o=np.zeros(64),A=T.T@A@T,b=T.T@(2*A@center+b),c=float(center@A@center+b@center+c),from_bus=0,to_bus=0))
 node=17;ib=Y[node]@rawB;i0=Y[node]@v0;R=np.stack([rawB[node].real,rawB[node].imag,ib.real,ib.imag]);A=np.zeros((4,4));A[1,2]=A[2,1]=base/2;A[0,3]=A[3,0]=-base/2;add('Q_inj18','Qinj',R,np.array([0.,0.,i0.real,i0.imag]),A,f=18)
 return out


def components(model,family,g):
 for j,w in enumerate(model['weights']):
  nu=float(model['df'][j])
  if family=='Block_TMM':
   Q1,Q2=g['Q1'],g['Q2'];G=model['coefficient'][j];H=Q1+Q2@G;mean=Q1@model['means'][j]+Q2@(model['intercept'][j]+G@model['means'][j]);B=H@model['scatter'][j]@H.T;R=Q2@model['residual'][j]@Q2.T
  else:mean=model['means'][j];B=np.zeros((64,64)) if np.isinf(nu) else model['scatter'][j];R=model['scatter'][j] if np.isinf(nu) else np.zeros((64,64))
  yield float(w),mean,B,R,nu


def projected_parts(model,family,g,spec):
 T=spec['R'];o=spec['o']
 for w,m,B,R,nu in components(model,family,g):yield w,T@m+o,T@B@T.T,T@R@T.T,nu


def scale_moments(nu):return (1.,1.) if np.isinf(nu) else (nu/(nu-2),nu*nu/((nu-2)*(nu-4)))


def exact_moments(parts,A,b,c):
 terms=[]
 for w,m,B,R,nu in parts:
  es,es2=scale_moments(nu);ell=2*A@m+b;tB=np.trace(A@B);tR=np.trace(A@R);mean=m@A@m+b@m+c+es*tB+tR
  var=es*ell@B@ell+ell@R@ell+2*es2*np.trace(A@B@A@B)+4*es*np.trace(A@B@A@R)+2*np.trace(A@R@A@R)+(es2-es**2)*tB*tB
  terms.append((w,mean,var))
 w,m,v=map(np.array,zip(*terms));mean=float(w@m);return mean,float(w@(v+(m-mean)**2))


def scale_nodes(nu,level):
 if np.isinf(nu):return np.ones(1),np.ones(1)
 n=40*level
 if 8<nu<=300 and n<=160:
  z,w=roots_genlaguerre(n,nu/2-1);return nu/(2*z),w/w.sum()
 # Probability integration represents the actual inverse-gamma, never a forced Gaussian.
 nodes,weights=leggauss(12*level);edges=[0.,.5,.9,.99,.999,.9999,.99999,.999999,1.];p=[];w=[]
 for left,right in zip(edges[:-1],edges[1:]):p.extend(left+(right-left)*(nodes+1)/2);w.extend((right-left)*weights/2)
 p=np.minimum(np.array(p),np.nextafter(1.,0.));return nu/stats.chi2.isf(p,nu),np.array(w)


def terms(parts,spec,level):
 A,b,c=spec['A'],spec['b'],spec['c'];out=[]
 for w,m,B,R,nu in parts:
  nodes,weights=scale_nodes(nu,level);ell=2*A@m+b;m0=float(m@A@m+b@m+c)
  C=R[None]+nodes[:,None,None]*B[None];C=(C+C.transpose(0,2,1))/2;vals,U=np.linalg.eigh(C);L=U*np.sqrt(np.maximum(vals,0))[:,None,:];M=L.transpose(0,2,1)@A@L;lam,rot=np.linalg.eigh(M);beta=np.einsum('nji,nj->ni',rot,np.einsum('nji,j->ni',L,ell))
  out.extend((float(w*wt),m0,la,be) for wt,la,be in zip(weights,lam,beta))
 return out


def decompose(parts,spec):
 A,b=spec['A'],spec['b'];origin=spec['o'];ell=2*A@origin+b;rows=[]
 for w,m,B,R,nu in parts:
  delta=m-origin;es,_=scale_moments(nu);C=R+es*B;EL=ell@delta;EH,VH=exact_moments([(1.,delta,B,R,nu)],A,np.zeros(len(A)),0.);VL=ell@C@ell;cov=2*ell@C@A@delta;rows.append((w,EL,EH,VL,VH,cov))
 w,l,h,vl,vh,co=map(np.array,zip(*rows));ml=w@l;mh=w@h
 return dict(VarL=float(w@(vl+(l-ml)**2)),VarH=float(w@(vh+(h-mh)**2)),twice_covariance=float(2*w@(co+(l-ml)*(h-mh))))
