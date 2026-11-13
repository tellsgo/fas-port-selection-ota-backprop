"""Ray-traced data of the unsurveyed-position test and of Fig. 2 (9600 queries per site, spacings 0.7/0.9/1.2 lambda, 8-step causal history).
Usage: CHICAGO_DIR=<scene dir> python full/gen_real3_data.py <bs_id> <max queries> <scattering coefficient> <out.npz>
Paper: bs_id 2 (low site) with scattering 0 and 0.3, bs_id 1 (high site) with 0; 9600 queries each."""
import os, json, time, sys, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]=""
import mitsuba as mi; mi.set_variant("llvm_ad_mono_polarized")
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver
BSID=int(sys.argv[1]); NMAX=int(sys.argv[2]); SCAT=float(sys.argv[3]); OUT=sys.argv[4]
S=os.environ.get("CHICAGO_DIR","data/01chicago").rstrip("/")+"/"   # Chicago scene of the dataset pipeline of arXiv:2608.10222 (see README)
cfg=json.load(open(S+"scene_config.json")); traj=[p for p in json.load(open(S+"xyz_paths.json"))["paths"] if p["bs_id"]==BSID]
fc=3.5e9; lam=3e8/fc; VW=1.2; TAU=0.040; AP=1.5; DS=[0.7,0.9,1.2]; K=8; NF=32; BW=20e6; LAT=[0.0,0.75,-0.75]
fsub=(np.arange(NF)-NF/2)*BW/NF
rng=np.random.default_rng(20261028+BSID)
def ports(d,psi):
    Rm=np.array([[np.cos(psi),-np.sin(psi)],[np.sin(psi),np.cos(psi)]]); A=Rm@np.array([[d,0.0],[d/2,d*np.sqrt(3)/2]]).T
    G=np.array([A@np.array([i,j]) for i in range(-6,7) for j in range(-6,7)]); return G[np.linalg.norm(G,axis=1)<=AP+1e-9]
Q=[]
for ti,p in enumerate(traj):
    for si,st in enumerate(p["steps"]):
        v=np.array(st["velocity"][:2]); vh=v/max(np.linalg.norm(v),1e-9); nrm=np.array([-vh[1],vh[0]])
        for lo in LAT:
            x=np.array(st["xyz"],float); x[:2]+=lo*nrm
            D=vh*VW*TAU/lam; psi=rng.uniform(0,2*np.pi); cand=[]
            for d in DS:
                P=ports(d,psi)+D; idx=np.argsort(np.linalg.norm(P,axis=1))[:4]; cand.append(P[idx])
            hist=[x-np.array([*(vh*VW*TAU*k),0.0]) for k in range(1,K+1)]
            Q.append(dict(traj=ti,step=si,lat=lo,x=x,vh=vh,D=D,psi=psi,cand=np.array(cand),hist=np.array(hist)))
Q=Q[:NMAX]
scene=load_scene(S+"01chicago.xml"); scene.frequency=fc
if SCAT>0:
    for m in scene.radio_materials.values(): m.scattering_coefficient=SCAT
for arr in ("tx_array","rx_array"): setattr(scene,arr,PlanarArray(num_rows=1,num_cols=1,pattern="iso",polarization="V"))
bs=[b for b in cfg["base_stations"] if b["bs_id"]==BSID][0]; scene.add(Transmitter("tx",position=bs["position"]))
solver=PathSolver(); NR=1+3*4+K; MAXP=160
hf=np.zeros(len(Q),complex); hc=np.zeros((len(Q),3,4),complex); Hh=np.zeros((len(Q),K,NF),complex)
PW=np.zeros((len(Q),MAXP)); TH=np.zeros((len(Q),MAXP)); PH=np.zeros((len(Q),MAXP)); TA=np.zeros((len(Q),MAXP)); NP=np.zeros(len(Q),int)
GB=25
for g0 in range(0,len(Q),GB):
    grp=Q[g0:g0+GB]; names=[]
    for qi,q in enumerate(grp):
        pts=[q["x"]]+[q["x"]+np.array([r[0]*lam,r[1]*lam,0.0]) for c in q["cand"] for r in c]+list(q["hist"])
        for j,pt in enumerate(pts): nm=f"r{qi}_{j}"; scene.add(Receiver(nm,position=pt.tolist())); names.append(nm)
    t0=time.time(); P=solver(scene,max_depth=5,diffuse_reflection=SCAT>0,diffraction=True,edge_diffraction=True,seed=7)
    ar,ai=P.a; A=(np.array(ar)+1j*np.array(ai)).reshape(len(names),-1); T=np.array(P.tau).reshape(len(names),-1)
    th=np.array(P.theta_r).reshape(len(names),-1); ph=np.array(P.phi_r).reshape(len(names),-1)
    h=(A*np.exp(-2j*np.pi*fc*T)).sum(1)
    for qi in range(len(grp)):
        b=qi*NR; n=g0+qi; hf[n]=h[b]; hc[n]=h[b+1:b+13].reshape(3,4)
        for k in range(K): Hh[n,k]=(A[b+13+k][None,:]*np.exp(-2j*np.pi*(fc+fsub)[:,None]*T[b+13+k][None,:])).sum(1)
        m=np.abs(A[b])>0; w=np.abs(A[b][m])**2; o=np.argsort(-w)[:MAXP]; NP[n]=m.sum()
        PW[n,:len(o)]=w[o]; TH[n,:len(o)]=th[b][m][o]; PH[n,:len(o)]=ph[b][m][o]; TA[n,:len(o)]=T[b][m][o]
    for nm in names: scene.remove(nm)
    print("group",g0,len(Q),"s",round(time.time()-t0,1),flush=True)
meta=np.array([[q["traj"],q["step"],q["lat"],*q["x"],*q["vh"],*q["D"],q["psi"]] for q in Q])
np.savez_compressed(OUT,meta=meta,cand=np.array([q["cand"] for q in Q]),hf=hf,hc=hc,Hhist=Hh,PW=PW,TH=TH,PH=PH,TAU=TA,NP=NP,
    bs=np.array(bs["position"]),spacings=np.array(DS),lam=lam,fsub=fsub,cols=np.array(["traj","step","lat","x","y","z","vhx","vhy","Dx","Dy","psi"]))
print("saved",OUT,len(Q))
