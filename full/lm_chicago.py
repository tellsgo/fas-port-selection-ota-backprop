"""Local map descriptor (free distance in 16 horizontal directions) for the Chicago data, used by full/xtest.py.
Usage: CHICAGO_DIR=<scene dir> python full/lm_chicago.py bs2_scat0.npz bs1_scat0.npz  -> *_LM.npy"""
import os, sys, json, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import mitsuba as mi; mi.set_variant("llvm_ad_mono_polarized")
from sionna.rt import load_scene
S=os.environ.get("CHICAGO_DIR","data/01chicago").rstrip("/")+"/"   # Chicago scene of the dataset pipeline of arXiv:2608.10222 (see README)
scene=load_scene(S+"01chicago.xml"); mis=scene.mi_scene
def cast(o,d):
    d=d/np.linalg.norm(d,axis=-1,keepdims=True)
    ray=mi.Ray3f(mi.Point3f(*[mi.Float(o[:,i]) for i in range(3)]),mi.Vector3f(*[mi.Float(d[:,i]) for i in range(3)]))
    t=np.array(mis.ray_intersect(ray).t); t[~np.isfinite(t)]=1e4; return t
NDIR=16; ang=np.arange(NDIR)*2*np.pi/NDIR; DIRS=np.c_[np.cos(ang),np.sin(ang),np.zeros(NDIR)]
for f in sys.argv[1:]:
    z=np.load(f); P=z["meta"][:,3:6].astype(float); P[:,2]=np.maximum(P[:,2],1.0)
    LM=np.minimum(cast(np.repeat(P,NDIR,0),np.tile(DIRS,(len(P),1))).reshape(len(P),NDIR),200.0)
    np.save(f.replace(".npz","_LM.npy"),LM); print(f,LM.shape,np.median(LM.max(1)))
