"""Zero-survey test in five cities (Table III, Fig. 4): each city held out in turn, MLP predictors trained on the other four.
<datadir> holds bs2_scat0.npz, bs1_scat0.npz (+ _LM.npy from full/lm_chicago.py) and the eight npz files of full/gen_xscene.py.
Usage: python full/xtest.py <datadir> test <outprefix>  -> <outprefix>_results.json, <outprefix>_per_query.csv"""
import sys, os, json, glob, numpy as np, torch, torch.nn as nn
torch.manual_seed(0); DD, MODE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
k = 2 * np.pi; JP = 1; DS = [0.7, 0.9, 1.2]
SITES = {}
for f in sorted(glob.glob(os.path.join(DD, '*.npz'))):
    z = np.load(f, allow_pickle=True); name = os.path.basename(f)[:-4]
    if name.startswith('bs2_scat0'): scene, site = 'chicago', 'low'
    elif name.startswith('bs1_scat0'): scene, site = 'chicago', 'high'
    else: scene, site = str(z['scene']), str(z['site'])
    LM = z['LM'] if 'LM' in z.files else np.load(f.replace('.npz', '_LM.npy'))
    ok = np.isfinite(z['hf']) & (np.abs(z['hf']) > 0)
    d = {kk: z[kk][ok] for kk in ['meta', 'cand', 'hf', 'hc', 'Hhist', 'PW', 'TH', 'PH']}; d['LM'] = LM[ok]; d['bs'] = z['bs'].astype(float)
    d['S'] = np.real(d['hc'] * np.conj(d['hf'])[:, None, None]) / np.abs(d['hf'])[:, None, None] ** 2
    d['scene'], d['site'] = scene, site; SITES[name] = d
print({n: (d['scene'], d['site'], len(d['hf'])) for n, d in SITES.items()}, flush=True)
SCENES = sorted(set(d['scene'] for d in SITES.values()))
def frame(d):
    v = d['meta'][:, 6:8]; v = v / np.linalg.norm(v, axis=1, keepdims=True); ang = np.arctan2(v[:, 1], v[:, 0])
    c, s = np.cos(-ang), np.sin(-ang); return v, ang, c, s
def rot(c, s, xy): return np.stack([c * xy[..., 0] - s * xy[..., 1], s * xy[..., 0] + c * xy[..., 1]], -1)
def feats(d, j):
    v, ang, c, s = frame(d); X = d['meta'][:, 3:6]; rel = d['bs'][None, :] - X; dist = np.linalg.norm(rel, axis=1)
    relh = rot(c, s, rel[:, :2] / dist[:, None]); elev = rel[:, 2] / dist
    sh = np.round(ang / (2 * np.pi / 16)).astype(int); LM = np.stack([np.roll(d['LM'][i], -sh[i]) for i in range(len(sh))])
    base = np.concatenate([relh, elev[:, None], np.log(dist)[:, None] / 6, np.log1p(LM) / 5], 1)            # (n,20)
    R = rot(c[:, None], s[:, None], d['cand'][:, j])                                                           # (n,4,2) in body-of-motion frame
    return np.concatenate([np.repeat(base[:, None, :], 4, 1), R], 2)                                          # (n,4,22)
def fdelay(d):
    T = np.fft.ifft(d['Hhist'], axis=2)[:, :, :8]; ref = T[:, 0, :][np.arange(len(T)), np.abs(T[:, 0, :]).argmax(1)][:, None, None]
    T = T / (ref + 1e-30); T = T / np.maximum(np.abs(T).max((1, 2), keepdims=True), 1e-9)
    return np.concatenate([T.real.reshape(len(T), -1), T.imag.reshape(len(T), -1)], 1)
def euc(d, j): return np.argmin(np.linalg.norm(d['cand'][:, j], axis=2), 1)
def los3d(d, j):
    X = d['meta'][:, 3:6]; rel = d['bs'][None, :] - X; u = rel[:, :2] / np.linalg.norm(rel, axis=1, keepdims=True)
    return np.argmax(np.cos(k * np.einsum('nqc,nc->nq', d['cand'][:, j], u)), 1)
def street(d, j):
    ang = np.arange(16) * 2 * np.pi / 16; a = ang[d['LM'].argmax(1)]; u = np.stack([np.cos(a), np.sin(a)], 1)
    return np.argmax(np.cos(k * np.einsum('nqc,nc->nq', d['cand'][:, j], u)), 1)
def hist_axial(d, j, L=8, lam_steps=0.048):
    v, _, _, _ = frame(d); lam = 3e8 / 3.5e9; NF = d['Hhist'].shape[2]; fsub = (np.arange(NF) - NF / 2) * 20e6 / NF
    mus = np.linspace(-1, 1, 41); taus = np.arange(0, 2e-6, 25e-9); xk = -(np.arange(1, 9) * lam_steps) / lam     # in lambda
    A = (np.exp(1j * k * mus[None, :, None, None] * xk[:, None, None, None]) * np.exp(-2j * np.pi * fsub[None, None, :, None] * taus[None, None, None, :]))
    A = A.transpose(0, 2, 1, 3).reshape(8 * NF, len(mus) * len(taus)); An = A / np.linalg.norm(A, axis=0, keepdims=True)
    out = np.zeros(len(d['hf']), int)
    for n in range(len(d['hf'])):
        y = d['Hhist'][n].reshape(-1); y = y / (np.abs(y).max() + 1e-30); res = y.copy(); S = []
        for _ in range(L):
            S.append(int(np.argmax(np.abs(An.conj().T @ res)))); As = A[:, S]; beta = np.linalg.lstsq(As, y, rcond=None)[0]; res = y - As @ beta
        mu = mus[np.array(S) // len(taus)]; rax = d['cand'][n, j] @ v[n]
        hp = (beta[None, :] * np.exp(1j * k * mu[None, :] * rax[:, None])).sum(1); hfp = beta.sum()
        out[n] = int(np.argmax(np.real(hp * np.conj(hfp))))
    return out
def dense_spec(d, j):
    out = np.zeros(len(d['hf']), int)
    for n in range(len(d['hf'])):
        m = d['PW'][n] > 0; w = d['PW'][n][m] / d['PW'][n][m].sum(); th, ph = d['TH'][n][m], d['PH'][n][m]
        uh = np.c_[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph)]; out[n] = int(np.argmax((w[None, :] * np.cos(k * d['cand'][n, j] @ uh.T)).sum(1)))
    return out
class Net(nn.Module):
    def __init__(s, dc, dh):
        super().__init__(); s.dh = dh
        s.hist = nn.Sequential(nn.Linear(dh, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU()) if dh else None
        s.head = nn.Sequential(nn.Linear(dc + (64 if dh else 0), 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(s, fc, fh=None):
        if s.dh: fc = torch.cat([fc, s.hist(fh)[:, None, :].expand(-1, fc.shape[1], -1)], 2)
        return s.head(fc).squeeze(-1)
def cells(d): return np.floor(d['meta'][:, 3:5] / 20).astype(int)
def split_val(d, rng=None):   # deterministic per site, shared by all methods (fix A1)
    r = np.random.default_rng(abs(hash((d['scene'], d['site']))) % (2**32)) if False else np.random.default_rng(sum(map(ord, d['scene'] + d['site'])))
    c = cells(d); keys = [tuple(x) for x in c]; uk = sorted(set(keys)); vk = set(uk[i] for i in r.permutation(len(uk))[:max(1, int(.15 * len(uk)))])
    return np.array([kk in vk for kk in keys])
t = lambda a: torch.tensor(a, dtype=torch.float32)
def train(train_sites, use_hist, rng):
    Fc, Fh, Y, Vc, Vh, VS = [], [], [], [], [], []
    for d in train_sites:
        vm = split_val(d, rng)
        for j in range(3):
            f = feats(d, j); y = (d['S'][:, j] > 0).astype(np.float32); h = fdelay(d) if use_hist else None
            Fc.append(f[~vm]); Y.append(y[~vm]); Fh.append(h[~vm] if use_hist else None)
            if j == JP: Vc.append(f[vm]); Vh.append(h[vm] if use_hist else None); VS.append(d['S'][vm, j])
    Fc = np.concatenate(Fc); Y = np.concatenate(Y); Fh = np.concatenate(Fh) if use_hist else None
    Vc = np.concatenate(Vc); Vh = np.concatenate(Vh) if use_hist else None; VS = np.concatenate(VS)
    net = Net(Fc.shape[2], Fh.shape[1] if use_hist else 0); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4); best, state, bad = 9, None, 0
    for ep in range(80):
        net.train(); pr = torch.randperm(len(Fc))
        for b in range(0, len(Fc), 512):
            ii = pr[b:b + 512]; opt.zero_grad(); loss = nn.functional.binary_cross_entropy_with_logits(net(t(Fc[ii]), t(Fh[ii]) if use_hist else None), t(Y[ii])); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): sc = net(t(Vc), t(Vh) if use_hist else None).numpy()
        vf = float(np.mean(VS[np.arange(len(VS)), sc.argmax(1)] < 0))
        if vf < best - 1e-4: best, state, bad = vf, {kk: v.clone() for kk, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 12: break
    net.load_state_dict(state); return net, best
def predict(net, d, j, use_hist):
    with torch.no_grad(): return net(t(feats(d, j)), t(fdelay(d)) if use_hist else None).numpy().argmax(1)
def val_geom(train_sites, rng):
    sc = {}
    for nm, fn in [('Euc', euc), ('LoS3D', los3d), ('StreetAxis', street)]:
        fl = []
        for d in train_sites:
            vm = split_val(d, np.random.default_rng(5)); ch = fn(d, JP); fl.append((d['S'][np.arange(len(ch)), JP, ch] < 0)[vm])
        sc[nm] = float(np.mean(np.concatenate(fl)))
    return sc
def val_hist(train_sites):
    fl = []
    for d in train_sites:
        vm = split_val(d, np.random.default_rng(5)); sub = {kk: (v[vm][:300] if isinstance(v, np.ndarray) and v.shape[:1] == d['hf'].shape else v) for kk, v in d.items()}
        ch = hist_axial(sub, JP); fl.append(sub['S'][np.arange(len(ch)), JP, ch] < 0)
    return float(np.mean(np.concatenate(fl)))
results = {'mode': MODE, 'scenes': SCENES}; per = []
for held in SCENES:
    rng = np.random.default_rng(17); tr = [d for d in SITES.values() if d['scene'] != held]; te = [(n, d) for n, d in SITES.items() if d['scene'] == held]
    if MODE == 'val':   # debug: hold out the validation cells of training scenes only, never touch held-out scenes
        te = []
    gv = val_geom(tr, rng); gbest = min(gv, key=gv.get); hv = val_hist(tr)
    h1rival = 'HistAxial' if hv < gv[gbest] else gbest
    n0, v0 = train(tr, False, np.random.default_rng(17)); n1, v1 = train(tr, True, np.random.default_rng(17))
    results[held] = {'val_geom': gv, 'geom_best': gbest, 'val_hist_axial': hv, 'h1_rival': h1rival, 'val_wm_h0': v0, 'val_wm_h1': v1}
    print(held, results[held], flush=True)
    for n, d in te:
        for j, dd in enumerate(DS):
            ch = {'Euc': euc(d, j), 'LoS3D': los3d(d, j), 'StreetAxis': street(d, j), 'HistAxial': hist_axial(d, j), 'WM-H0': predict(n0, d, j, False), 'WM-H1': predict(n1, d, j, True),
                  'denseSpec': dense_spec(d, j), 'Orc': d['S'][:, j].argmax(1)}
            ch['Rival-H0'] = ch[gbest]; ch['Rival-H1'] = ch[h1rival]
            for i in range(len(d['hf'])):
                per.append(dict(scene=held, site=d['site'], cell=f"{held}_{d['site']}_{tuple(cells(d)[i])}", d=dd, **{m: float(np.clip(d['S'][i, j, ch[m][i]], -3, 3)) for m in ch}))
if MODE == 'test':
    import pandas as pd
    df = pd.DataFrame(per); df.to_csv(OUT + '_per_query.csv', index=False); brng = np.random.default_rng(11); summ = {}
    for dd in DS:
        sub = df[df.d == dd]; ms = [m for m in sub.columns if m not in ('scene', 'site', 'cell', 'd')]
        summ[str(dd)] = {'flip': {m: float((sub[m] < 0).mean()) for m in ms}, 'mean_s': {m: float(sub[m].mean()) for m in ms},
                         'by_scene_site': {f'{a}_{b}': {m: float((g[m] < 0).mean()) for m in ms} for (a, b), g in sub.groupby(['scene', 'site'])}}
        ug = sub.cell.unique(); idx = {g: np.where(sub.cell.values == g)[0] for g in ug}
        for a, b in [('Rival-H0', 'WM-H0'), ('Rival-H1', 'WM-H1'), ('WM-H1', 'denseSpec')]:
            fa = (sub[a].values < 0).astype(float); fb = (sub[b].values < 0).astype(float); bs = []
            for _ in range(2000):
                gg = brng.choice(ug, len(ug)); ii = np.concatenate([idx[g] for g in gg]); bs.append(fa[ii].mean() - fb[ii].mean())
            summ[str(dd)][f'{a} - {b}'] = [float(fa.mean() - fb.mean()), *np.quantile(bs, [.025, .975]).tolist()]
        summ[str(dd)]['n_clusters'] = int(len(ug)); summ[str(dd)]['n_queries'] = int(len(sub))
    results['summary'] = summ
json.dump(results, open(OUT + '_results.json', 'w'), indent=1, default=str); print('done', flush=True)
