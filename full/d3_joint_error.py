"""Error budget of the unsurveyed-position test (extended version, arXiv appendix): noise, phase drift, relative and absolute position
errors, orientation and port-calibration errors added one at a time and jointly. Its error-free level gives the
0.32 -> 0.27 interval [-0.3, 11.2] of the paper. Usage: python full/d3_joint_error.py <bs2_scat0.npz> <out.json>"""
import sys, json, time, numpy as np
from scipy.spatial import cKDTree
NPZ, OUT = sys.argv[1], sys.argv[2]
z = np.load(NPZ, allow_pickle=True)
meta, cand, hf, hc = z['meta'], z['cand'], z['hf'], z['hc']
PW, TH, PH, TAU, HH = z['PW'], z['TH'], z['PH'], z['TAU'], z['Hhist']
bs = z['bs'].astype(float); lam = float(z['lam']); DS = list(z['spacings']); fsub = z['fsub']
N = len(hf); k = 2 * np.pi; c0 = 3e8
traj = meta[:, 0].astype(int); X = meta[:, 3:6]; VH = meta[:, 6:8]; DL = meta[:, 8:10]
JP = DS.index(0.9); SIGN = +1
# ---------- split, identical to real3.py ----------
first = {}
for i in range(N):
    if traj[i] not in first or meta[i, 1] < meta[first[traj[i]], 1]: first[traj[i]] = i
cell = {t: tuple(np.floor(X[i, :2] / 10).astype(int)) for t, i in first.items()}
cells = sorted(set(cell.values())); rng = np.random.default_rng(7); perm = rng.permutation(len(cells))
nt, nv = int(round(.30 * len(cells))), int(round(.15 * len(cells)))
role_cell = {cells[j]: ('test' if r < nt else 'val' if r < nt + nv else 'train') for r, j in enumerate(perm)}
role = np.array([role_cell[cell[t]] for t in traj]); grp = np.array([cells.index(cell[t]) for t in traj])
te = np.where(role == 'test')[0]; dmin = cKDTree(X[te, :2]).query(X[:, :2])[0]
keep = (role == 'test') | (dmin >= 2.0)
tr = np.where((role == 'train') & keep)[0]; S = tr
TREE = cKDTree(X[S, :2])
# ---------- helpers from real3.py, with the query position and residuals passed in ----------
def paths(i):
    m = PW[i] > 0; w = PW[i][m]; th, ph, ta = TH[i][m], PH[i][m], TAU[i][m]
    u3 = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1)
    return w / w.sum(), u3, ta
PATHS = {}
def P(i):
    if i not in PATHS: PATHS[i] = paths(i)
    return PATHS[i]
def mixture(i, xq, kk, p, vs):
    d, idx = TREE.query(xq[:2], k=min(kk + 1, len(S))); d = np.atleast_1d(d); idx = np.atleast_1d(idx)
    m = S[idx] != i; d, idx = d[m][:kk], idx[m][:kk]
    wts = 1 / np.maximum(d, 0.1) ** p; wts = wts / wts.sum(); W, U, T = [], [], []
    for wt, j in zip(wts, S[idx]):
        w, u3, ta = P(j)
        if vs:
            vsrc = X[j][None, :] + u3 * (c0 * ta)[:, None]; vec = vsrc - xq[None, :]; dist = np.linalg.norm(vec, axis=1)
            u3 = vec / dist[:, None]; w = w * (c0 * ta / np.maximum(dist, 1e-3)) ** 2; w = w / w.sum(); ta = dist / c0
        W.append(wt * w); U.append(u3); T.append(ta)
    return np.concatenate(W), np.concatenate(U), np.concatenate(T)
def phi_of(w, u3, R): return (w[None, :] * np.exp(SIGN * 1j * k * (R @ u3[:, :2].T))).sum(1)
def risk(phi): return 0.5 * (1 - phi.real / np.sqrt(np.maximum(1 - phi.imag ** 2, 1e-9)))
def map_choice(i, xq, R): w, u3, _ = mixture(i, xq, 4, 0, False); return int(np.argmin(risk(phi_of(w, u3, R))))
def vs_hist_choice(i, xq, R, Hk, dx, L=24, ridge=1e-3):
    w, u3, ta = mixture(i, xq, 2, 0, True); o = np.argsort(-w)[:L]; u3, ta = u3[o], ta[o]
    scale = np.abs(Hk).mean() + 1e-30; rows = []
    for m in range(Hk.shape[0]):
        ph_sp = np.exp(SIGN * 1j * k * (dx[m] / lam) @ u3[:, :2].T)
        rows.append(np.exp(-2j * np.pi * fsub[:, None] * ta[None, :]) * ph_sp[None, :])
    A = np.concatenate(rows, 0); y = (Hk / scale).reshape(-1)
    beta = np.linalg.solve(A.conj().T @ A + ridge * np.eye(len(ta)), A.conj().T @ y)
    hfp = beta.sum(); hp = (beta[None, :] * np.exp(SIGN * 1j * k * (R @ u3[:, :2].T))).sum(1)
    return int(np.argmax(np.real(hp * np.conj(hfp)) / (np.abs(hfp) ** 2 + 1e-30)))
def los_choice(xq, R): u = (bs - xq)[:2] / np.linalg.norm(bs - xq); return int(np.argmax(np.cos(k * R @ u)))
def rot(a): return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
# ---------- one error level ----------
def run(e, seed):
    r = np.random.default_rng(seed); out = {m: np.zeros(len(te), bool) for m in ('Nearest', 'LoS', 'Map', 'VS-Hist', 'Orc')}
    snr = e['snr_db']; sphi = np.deg2rad(e['phi_deg']); srel = e['rel_m']; sabs = e['abs_m']; seps = np.deg2rad(e['eps_deg']); scal = np.deg2rad(e['cal_deg'])
    for n, i in enumerate(te):
        R0 = cand[i, JP]; Dl = DL[i]
        th = r.normal(0, scal, 4) if scal > 0 else np.zeros(4)
        s_true = np.real(hc[i, JP] * np.exp(1j * th) * np.conj(hf[i])) / np.abs(hf[i]) ** 2
        eps = r.normal(0, seps) if seps > 0 else 0.0
        xi = r.normal(0, srel, 2) / lam if srel > 0 else np.zeros(2)
        Rh = (rot(eps) @ (R0 - Dl).T).T + Dl + xi
        xq = X[i].copy()
        if sabs > 0: xq[:2] += r.normal(0, sabs, 2)
        Hk = HH[i].copy(); M = Hk.shape[0]
        if sphi > 0: Hk = Hk * np.exp(1j * np.cumsum(r.normal(0, sphi, M)))[:, None]
        if snr is not None:
            pw = np.mean(np.abs(HH[i]) ** 2); sig = np.sqrt(pw / 10 ** (snr / 10) / 2)
            Hk = Hk + sig * (r.standard_normal(Hk.shape) + 1j * r.standard_normal(Hk.shape))
        dx = np.array([-(m + 1) * VH[i] * 1.2 * 0.04 for m in range(M)])
        if srel > 0: dx = dx + r.normal(0, srel, dx.shape)
        ch = {'Nearest': int(np.argmin(np.linalg.norm(Rh, axis=1))), 'LoS': los_choice(xq, Rh), 'Map': map_choice(i, xq, Rh),
              'VS-Hist': vs_hist_choice(i, xq, Rh, Hk, dx), 'Orc': int(np.argmax(s_true))}
        for m, c in ch.items(): out[m][n] = s_true[c] < 0
    return out
G = grp[te]; UG = np.unique(G); IDX = {g: np.where(G == g)[0] for g in UG}
def boot(a, b, B=2000, seed=11):
    rr = np.random.default_rng(seed); d = []
    for _ in range(B):
        ii = np.concatenate([IDX[g] for g in rr.choice(UG, len(UG))]); d.append(a[ii].mean() - b[ii].mean())
    return [float(a.mean() - b.mean()), *np.quantile(d, [.025, .975]).tolist()]
Z = dict(snr_db=None, phi_deg=0, rel_m=0, abs_m=0, eps_deg=0, cal_deg=0)
LEVELS = [('none', {})]
for key, vals in [('snr_db', [20, 10, 5]), ('phi_deg', [5, 15, 30]), ('rel_m', [0.002, 0.005, 0.01]), ('abs_m', [1, 3]), ('eps_deg', [2, 5, 10]), ('cal_deg', [10, 20, 30])]:
    LEVELS += [(f'{key}={v}', {key: v}) for v in vals]
LEVELS += [('joint_moderate', dict(snr_db=10, phi_deg=5, rel_m=0.002, abs_m=1, eps_deg=2, cal_deg=10)),
           ('joint_severe', dict(snr_db=5, phi_deg=15, rel_m=0.005, abs_m=3, eps_deg=5, cal_deg=20))]
res = {'n_test': int(len(te)), 'n_clusters': int(len(UG)), 'survey': int(len(S)), 'levels': {}}
for li, (name, e) in enumerate(LEVELS):
    t0 = time.time(); o = run({**Z, **e}, 1000 + li)
    rates = {m: float(v.mean()) for m, v in o.items()}
    res['levels'][name] = {'err': {**Z, **e}, 'rates': rates, 'Map-VSHist': boot(o['Map'], o['VS-Hist']), 'LoS-VSHist': boot(o['LoS'], o['VS-Hist']),
                           'Nearest-VSHist': boot(o['Nearest'], o['VS-Hist']), 'LoS-Map': boot(o['LoS'], o['Map'])}
    print(name, {m: round(x, 3) for m, x in rates.items()}, 'Map-VSHist', np.round(res['levels'][name]['Map-VSHist'], 3), f'{time.time() - t0:.0f}s', flush=True)
    if name == 'none':
        ref = {'Nearest': 0.494, 'LoS': 0.366, 'Map': 0.319, 'VS-Hist': 0.270, 'Orc': 0.090}
        bad = {m: (round(rates[m], 3), v) for m, v in ref.items() if abs(rates[m] - v) > 0.0015}
        if bad: print('REPRODUCTION FAILED', bad, flush=True); json.dump(res, open(OUT, 'w'), indent=1); sys.exit(2)
    json.dump(res, open(OUT, 'w'), indent=1)
print('done', OUT, flush=True)
