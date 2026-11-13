"""Second-order sufficiency on the 3GPP CDL profiles (Section VIII-D; extended version: CDL table).
Usage: python scripts/cdl_check.py [out.json]"""
import numpy as np, json, re, warnings, sys, os
from scipy.optimize import linprog
warnings.filterwarnings('ignore')
rng=np.random.default_rng(2026102604)
HERE=os.path.dirname(os.path.abspath(__file__)); OUT=sys.argv[1] if len(sys.argv)>1 else 'cdl_results.json'
s=open(os.path.join(HERE,'..','data','cdl_38901_v18_tables.txt')).read()
alpha=np.array([0.0447,0.1413,0.2492,0.3715,0.5129,0.6797,0.8844,1.1481,1.5195,2.1551])
offs=np.concatenate([alpha,-alpha])
num=re.compile(r'-?\d+\.?\d*')
def parse(name,idx):
    a=s.index(f'Table 7.7.1-{idx}. CDL-{name}'); b=s.index('Per-Cluster Parameters',a); c=s.index('Value',b)
    casa=float(num.findall(s[c:c+200])[1])
    rays=[]  # (power_lin, aoa_deg)
    for line in s[a:b].splitlines()[2:]:
        t=num.findall(line)
        if len(t)<6: continue
        delay,pdb,aod,aoa,zod,zoa=map(float,t[-6:]); p=10**(pdb/10)
        if 'Specular' in line: rays.append((p,aoa))
        else: rays+= [(p/20,aoa+casa*o) for o in offs]
    w=np.array([r[0] for r in rays]); w=w/w.sum(); ang=np.deg2rad([r[1] for r in rays])
    return w,ang
prof={n:parse(n,i) for i,n in enumerate('ABCDE',1)}
k=2*np.pi; NT=1440; th=np.linspace(0,2*np.pi,NT,endpoint=False); U=np.stack([np.cos(th),np.sin(th)],1)
A={'M2':np.vstack([np.ones(NT),np.cos(2*th),np.sin(2*th)])}
A['M2c4']=np.vstack([A['M2'],np.cos(4*th)]); A['M4']=np.vstack([A['M2c4'],np.sin(4*th)])
def lattice(d): return np.array([[d*(i+0.5*(j%2)),d*np.sqrt(3)/2*j] for i in range(-6,7) for j in range(-6,7)])
def mid(r,Am,mom):
    f=np.cos(k*(U@r)); lo=linprog(-mom,A_ub=Am.T,b_ub=f,bounds=[(None,None)]*len(mom),method='highs')
    hi=linprog(mom,A_ub=-Am.T,b_ub=-f,bounds=[(None,None)]*len(mom),method='highs')
    return ((-lo.fun if lo.status==0 else -1)+(hi.fun if hi.status==0 else 1))/2
res={}
for d in [0.5,0.7,0.9]:
    P=lattice(d)
    for n,(w,ang0) in prof.items():
        fl={m:0 for m in ['Euc','M2','M2c4','M4','Orc']}
        for _ in range(1000):
            ang=ang0+rng.uniform(0,2*np.pi); V=np.stack([np.cos(ang),np.sin(ang)],1)
            rad=rng.uniform(0.2,0.6)*(d/0.5); a_=rng.uniform(0,2*np.pi); D=rad*np.array([np.cos(a_),np.sin(a_)])
            R=P[np.argsort(np.linalg.norm(P-D,axis=1))[:4]]-D
            c=np.array([(w*np.cos(k*(V@r))).sum() for r in R])
            ch={'Euc':int(np.argmin(np.linalg.norm(R,axis=1))),'Orc':int(np.argmax(c))}
            for m,Am in A.items():
                mom=np.array([(w*np.interp(ang%(2*np.pi),th,row,period=2*np.pi)).sum() for row in Am])
                ch[m]=int(np.argmax([mid(r,Am,mom) for r in R]))
            for m,i in ch.items(): fl[m]+=c[i]<0
        res[f'd{d}_CDL{n}']={m:v/1000 for m,v in fl.items()}; print(d,n,res[f'd{d}_CDL{n}'],flush=True)
pool={m:float(np.mean([res[f'd0.9_CDL{n}'][m] for n in 'ABCDE'])) for m in ['Euc','M2','M2c4','M4','Orc']}
res['pooled_d0.9']=pool; print('pooled',pool); json.dump(res,open(OUT,'w'),indent=1)
