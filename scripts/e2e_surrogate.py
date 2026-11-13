"""Training surrogate of Section VIII-G (digits, linear classifier, 3000 steps; each update scales the exact minibatch
gradient by a clipped alignment factor drawn from the paired test queries of the chosen rule).
Paper: every rule other than the nearest port 94.7-95.2 percent at 0.9 lambda, no aging 95.1, nearest port 87.8
(81.2-92.9 across five seeds). Usage: python scripts/e2e_surrogate.py [out.json] [spacings, default 0.9,0.7]"""
import numpy as np, json, sys, os
ROOT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'); OUT=sys.argv[1] if len(sys.argv)>1 else 'results_E2.json'
DLIST=[float(v) for v in sys.argv[2].split(',')] if len(sys.argv)>2 else [0.9,0.7]
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
z=np.load(os.path.join(ROOT,'results','extract','test_bs2_scat0.npz')); c=np.load(os.path.join(ROOT,'results','real3','test_bs2_scat0_choices.npz'))
hf,hc=z['hf'],z['hc']; EVAL=np.arange(len(hf)); S=np.clip(np.real(hc*np.conj(hf)[:,None,None])/np.abs(hf)[:,None,None]**2,-3,3)
X,y=load_digits(return_X_y=True)
Xa,Xte,ya,yte=train_test_split(X,y,test_size=197,random_state=0); Xtr,Xva,ytr,yva=train_test_split(Xa,ya,test_size=200,random_state=0)
mu,sd=Xtr.mean(0),Xtr.std(0)+1e-6; f=lambda A: np.c_[(A-mu)/sd,np.ones(len(A))]; Xtr,Xva,Xte=f(Xtr),f(Xva),f(Xte); X=Xtr
POL=['Euc','BSdir3D','StrongCKM-H0','StrongCKM-H1','Learner-H1','denseSpec','Orc']
def svec(d,m):
    j=[0.7,0.9,1.2].index(d); ch=c[f'full_d{d}__{m}']; return S[EVAL,j,ch]
def run(sv,lr,seed,Xev,yev,T=3000):
    rng=np.random.default_rng(seed); W=np.zeros((10,X.shape[1]))
    for t in range(T):
        b=rng.integers(0,len(Xtr),32); Z=Xtr[b]@W.T; P=np.exp(Z-Z.max(1,keepdims=True)); P/=P.sum(1,keepdims=True); P[np.arange(32),ytr[b]]-=1
        g=P.T@Xtr[b]/32; s=1.0 if sv is None else sv[rng.integers(len(sv))]
        W-=lr*(s*g+0.01*rng.standard_normal(g.shape))
        if not np.all(np.isfinite(W)): return 0.0
    return float(np.mean((Xev@W.T).argmax(1)==yev))
out={}
for d in DLIST:
    res={}
    for m in POL+['Ideal']:
        sv=None if m=='Ideal' else svec(d,m)
        va={lr:np.mean([run(sv,lr,s,Xva,yva) for s in range(5)]) for lr in [0.01,0.03,0.1,0.3]}; lr=max(va,key=va.get)
        te=[run(sv,lr,100+s,Xte,yte) for s in range(5)]
        res[m]={'lr':lr,'test_acc_mean':float(np.mean(te)),'test_acc_seeds':te,'flip_rate':None if sv is None else float(np.mean(sv<0))}
        print(d,m,res[m],flush=True)
    out[str(d)]=res
r=out['0.9']; order=['Orc','denseSpec','StrongCKM-H1','StrongCKM-H0','Euc']
rho=spearmanr([r[m]['test_acc_mean'] for m in POL],[r[m]['flip_rate'] for m in POL]).correlation
out['P1']=r['Orc']['test_acc_mean']-r['Euc']['test_acc_mean']; out['P2']=r['StrongCKM-H1']['test_acc_mean']-r['Euc']['test_acc_mean']; out['P3_spearman_acc_vs_flip']=float(rho)
print({k:out[k] for k in ['P1','P2','P3_spearman_acc_vs_flip']}); json.dump(out,open(OUT,'w'),indent=1)
