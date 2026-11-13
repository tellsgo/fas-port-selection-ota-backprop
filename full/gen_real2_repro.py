"""Vehicular displacements of the scene trajectories, unrestricted lattice, Chicago low site, 500 queries (quadratic rule vs Spec, Section VIII-D).
Usage: CHICAGO_DIR=<scene dir> python full/gen_real2_repro.py 2  -> results_repro.json, per_sample_repro.json"""
import os, json, time, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]=""
import mitsuba as mi; mi.set_variant("llvm_ad_mono_polarized")
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver
S=os.environ.get("CHICAGO_DIR","data/01chicago").rstrip("/")+"/"   # Chicago scene of the dataset pipeline of arXiv:2608.10222 (see README)
cfg=json.load(open(S+"scene_config.json")); import sys; BSID=int(sys.argv[1]); traj=[p for p in json.load(open(S+"xyz_paths.json"))["paths"] if p["bs_id"]==BSID]
rng=np.random.default_rng(31415926); fc=3.5e9; lam=3e8/fc; k=2*np.pi; tau_ex=0.040; N=500; DS=[0.9,1.2]
bs=[b for b in cfg["base_stations"] if b["bs_id"]==BSID][0]; bpos=np.array(bs["position"])
def lattice(d): return np.array([[d*(i+0.5*(j%2)),d*np.sqrt(3)/2*j] for i in range(-8,9) for j in range(-8,9)])
samples=[]
for n in range(N):
    p=traj[rng.integers(len(traj))]; st=p["steps"][rng.integers(16)]
    x=np.array(st["xyz"]); v=np.array(st["velocity"])[:2]; D=v*tau_ex/lam   # in lambda
    psi=rng.uniform(0,2*np.pi); Rm=np.array([[np.cos(psi),-np.sin(psi)],[np.sin(psi),np.cos(psi)]])
    cand={}
    for d in DS:
        # local lattice around -D in body frame (fix v1 coverage bug): basis a1=(d,0), a2=(d/2, d*sqrt3/2), rotated by psi
        A=np.array([[d,0.0],[d/2,d*np.sqrt(3)/2]]).T; A=Rm@A
        c=np.linalg.solve(A,-D); i0_,j0_=np.round(c).astype(int)
        G=np.array([A@np.array([i,j]) for i in range(i0_-4,i0_+5) for j in range(j0_-4,j0_+5)])
        P=G+D; idx=np.argsort(np.linalg.norm(P,axis=1))[:4]; cand[d]=P[idx]   # residuals in lambda
    samples.append(dict(x=x,D=D,cand=cand))
pts=[]; 
for s in samples:
    s["i0"]=len(pts); pts.append(s["x"])
    for d in DS:
        for r in s["cand"][d]: pts.append(s["x"]+np.array([r[0]*lam,r[1]*lam,0.0]))
scene=load_scene(S+"01chicago.xml"); scene.frequency=fc
for arr in ("tx_array","rx_array"): setattr(scene,arr,PlanarArray(num_rows=1,num_cols=1,pattern="iso",polarization="V"))
scene.add(Transmitter("tx",position=bs["position"]))
solver=PathSolver(); H=np.zeros(len(pts),complex); PATHS={}
B=520
for b0 in range(0,len(pts),B):
    names=[]
    for i in range(b0,min(b0+B,len(pts))):
        nm=f"rx{i}"; scene.add(Receiver(nm,position=pts[i].tolist())); names.append(nm)
    t0=time.time(); P=solver(scene,max_depth=5,diffuse_reflection=True,diffraction=True,edge_diffraction=True,seed=7)
    ar,ai=P.a; A=(np.array(ar)+1j*np.array(ai)).reshape(len(names),-1); T=np.array(P.tau).reshape(len(names),-1)
    th=np.array(P.theta_r).reshape(len(names),-1); ph=np.array(P.phi_r).reshape(len(names),-1)
    H[b0:b0+len(names)]=(A*np.exp(-2j*np.pi*fc*T)).sum(1)
    for j,i in enumerate(range(b0,b0+len(names))):
        m=np.abs(A[j])>0; PATHS[i]=(np.abs(A[j][m])**2, th[j][m], ph[j][m])
    for nm in names: scene.remove(nm)
    print("batch",b0,"s",round(time.time()-t0,1),flush=True)
res={d:{m:0 for m in ["Euc","BSdir","Spec","Cmet","Orc"]} for d in DS}; npaths=[]; Dn=[]; per=[]
for s in samples:
    i0=s["i0"]; hf=H[i0]; pw,th,ph=PATHS[i0]; npaths.append(len(pw)); Dn.append(float(np.linalg.norm(s["D"])))
    uh=np.stack([np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph)],1); w=pw/pw.sum()
    C=(w[:,None,None]*uh[:,:,None]*uh[:,None,:]).sum(0)
    ub=(bpos-s["x"])[:2]; ub=ub/np.linalg.norm(ub)
    off=1
    for d in DS:
        R=s["cand"][d]; hs=H[i0+off:i0+off+4]; off+=4
        sv=np.real(hs*np.conj(hf))/abs(hf)**2
        ch={"Euc":int(np.argmin(np.linalg.norm(R,axis=1))),"BSdir":int(np.argmax(np.cos(k*R@ub))),
            "Spec":int(np.argmax([(w*np.cos(k*uh@r)).sum() for r in R])),"Cmet":int(np.argmin([r@C@r for r in R])),"Orc":int(np.argmax(sv))}
        for m,i in ch.items(): res[d][m]+= sv[i]<0
        per.append({'d':d,**{m:int(sv[i]<0) for m,i in ch.items()}})
out={f"d{d}":{m:v/N for m,v in res[d].items()} for d in DS}
out["D_over_lambda_quantiles"]=np.quantile(Dn,[.1,.5,.9]).tolist(); out["npaths_median"]=float(np.median(npaths))
print(json.dumps(out,indent=1)); json.dump(out,open("results_repro.json","w"),indent=1); json.dump(per,open("per_sample_repro.json","w"))
