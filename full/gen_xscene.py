"""Ray-traced data of the four Sionna RT cities (munich, etoile, florence, san_francisco), high and low site, 3000 queries each.
Usage: python full/gen_xscene.py <scene> <site: high|low> 3000 <out.npz>   (Sionna RT 2.2, CPU)"""
import os, sys, time, json, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import mitsuba as mi; mi.set_variant("llvm_ad_mono_polarized")
import drjit as dr
import sionna.rt as rt
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver
SC, SITE, N, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
fc = 3.5e9; lam = 3e8 / fc; VW = 1.2; TAU = 0.040; AP = 1.5; DS = [0.7, 0.9, 1.2]; K = 8; NF = 32; BW = 20e6; UEZ = 1.5
fsub = (np.arange(NF) - NF / 2) * BW / NF
rng = np.random.default_rng({"munich": 1, "etoile": 2, "florence": 3, "san_francisco": 4}[SC] * 10 + (SITE == "low"))
scene = load_scene(getattr(rt.scene, SC)); scene.frequency = fc
mis = scene.mi_scene; bb = mis.bbox(); lo, hi = np.array(bb.min), np.array(bb.max)
def cast(o, d):
    o = np.asarray(o, float); d = np.asarray(d, float); d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    ray = mi.Ray3f(mi.Point3f(*[mi.Float(o[:, i]) for i in range(3)]), mi.Vector3f(*[mi.Float(d[:, i]) for i in range(3)]))
    si = mis.ray_intersect(ray); t = np.array(si.t); t[~np.isfinite(t)] = 1e4; return t
def outdoor(P):  # no geometry above, and ground below within 20 m
    up = cast(P, np.tile([0, 0, 1.0], (len(P), 1))); return up > 500
def ground_z(xy):
    o = np.c_[xy, np.full(len(xy), hi[2] + 10)]; t = cast(o, np.tile([0, 0, -1.0], (len(xy), 1))); return hi[2] + 10 - t
NDIR = 16; ang = np.arange(NDIR) * 2 * np.pi / NDIR; DIRS = np.c_[np.cos(ang), np.sin(ang), np.zeros(NDIR)]
def local_map(P):
    t = cast(np.repeat(P, NDIR, 0), np.tile(DIRS, (len(P), 1))).reshape(len(P), NDIR); return np.minimum(t, 200.0)
# --- candidate UE points ---
xy = rng.uniform(lo[:2] + 20, hi[:2] - 20, size=(N * 6, 2)); gz = ground_z(xy); P = np.c_[xy, gz + UEZ]
ok = outdoor(P) & (gz > lo[2] - 1) & (gz < lo[2] + 30); P = P[ok]
LM = local_map(P); open_ = LM.max(1) > 15; P, LM = P[open_], LM[open_]
# --- base station ---
if SITE == "high":
    cxy = (lo[:2] + hi[:2]) / 2; bsp = np.array([*cxy, hi[2] + 5.0])
else:
    j = rng.integers(len(P)); bsp = P[j].copy(); bsp[2] = gz[ok][open_][j] + 10.0 if False else P[j][2] - UEZ + 10.0
d2 = np.linalg.norm(P[:, :2] - bsp[:2], axis=1); sel = (d2 > 15) & (d2 < 250); P, LM = P[sel][:N], LM[sel][:N]
print(SC, SITE, "bs", np.round(bsp, 1), "queries", len(P), flush=True)
# --- street axis = direction with longest free path (axis, sign random) ---
axis = DIRS[LM.argmax(1) % NDIR][:, :2]; sgn = rng.choice([-1, 1], len(P))[:, None]; VH = axis * sgn
def ports(d, psi):
    Rm = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]]); A = Rm @ np.array([[d, 0.0], [d / 2, d * np.sqrt(3) / 2]]).T
    G = np.array([A @ np.array([i, j]) for i in range(-6, 7) for j in range(-6, 7)]); return G[np.linalg.norm(G, axis=1) <= AP + 1e-9]
Q = []
for n in range(len(P)):
    D = VH[n] * VW * TAU / lam; psi = rng.uniform(0, 2 * np.pi); cand = []
    for d in DS:
        G = ports(d, psi) + D; idx = np.argsort(np.linalg.norm(G, axis=1))[:4]; cand.append(G[idx])
    hist = [P[n] - np.array([*(VH[n] * VW * TAU * k), 0.0]) for k in range(1, K + 1)]
    Q.append(dict(x=P[n], D=D, psi=psi, cand=np.array(cand), hist=np.array(hist)))
for arr in ("tx_array", "rx_array"): setattr(scene, arr, PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V"))
scene.add(Transmitter("tx", position=bsp.tolist()))
solver = PathSolver(); NR = 1 + 3 * 4 + K; MAXP = 160; GB = 25
hf = np.zeros(len(Q), complex); hc = np.zeros((len(Q), 3, 4), complex); Hh = np.zeros((len(Q), K, NF), complex)
PW = np.zeros((len(Q), MAXP)); TH = np.zeros_like(PW); PH = np.zeros_like(PW); TA = np.zeros_like(PW); NP = np.zeros(len(Q), int)
for g0 in range(0, len(Q), GB):
    grp = Q[g0:g0 + GB]; names = []
    for qi, q in enumerate(grp):
        pts = [q["x"]] + [q["x"] + np.array([r[0] * lam, r[1] * lam, 0.0]) for c in q["cand"] for r in c] + list(q["hist"])
        for jj, pt in enumerate(pts): nm = f"r{qi}_{jj}"; scene.add(Receiver(nm, position=pt.tolist())); names.append(nm)
    t0 = time.time(); Pt = solver(scene, max_depth=5, diffuse_reflection=False, diffraction=True, edge_diffraction=True, seed=7)
    ar, ai = Pt.a; A = (np.array(ar) + 1j * np.array(ai)).reshape(len(names), -1); T = np.array(Pt.tau).reshape(len(names), -1)
    th = np.array(Pt.theta_r).reshape(len(names), -1); ph = np.array(Pt.phi_r).reshape(len(names), -1)
    h = (A * np.exp(-2j * np.pi * fc * T)).sum(1)
    for qi in range(len(grp)):
        b = qi * NR; n = g0 + qi; hf[n] = h[b]; hc[n] = h[b + 1:b + 13].reshape(3, 4)
        for kk in range(K): Hh[n, kk] = (A[b + 13 + kk][None, :] * np.exp(-2j * np.pi * (fc + fsub)[:, None] * T[b + 13 + kk][None, :])).sum(1)
        m = np.abs(A[b]) > 0; w = np.abs(A[b][m]) ** 2; o = np.argsort(-w)[:MAXP]; NP[n] = m.sum()
        PW[n, :len(o)] = w[o]; TH[n, :len(o)] = th[b][m][o]; PH[n, :len(o)] = ph[b][m][o]; TA[n, :len(o)] = T[b][m][o]
    for nm in names: scene.remove(nm)
    if g0 % 500 == 0: print("group", g0, len(Q), "s", round(time.time() - t0, 1), flush=True)
meta = np.array([[n, 0, 0, *q["x"], *VH[n], *q["D"], q["psi"]] for n, q in enumerate(Q)])
np.savez_compressed(OUT, meta=meta, cand=np.array([q["cand"] for q in Q]), hf=hf, hc=hc, Hhist=Hh, PW=PW, TH=TH, PH=PH, TAU=TA, NP=NP, LM=LM[:len(Q)],
                    bs=bsp, spacings=np.array(DS), lam=lam, fsub=fsub, scene=SC, site=SITE)
print("saved", OUT, len(Q), flush=True)
