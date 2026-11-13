"""Unsurveyed-position test (Tables II and IV, Fig. 5, text intervals): survey maps (IDW, virtual-source transport, VS-Hist) and
learned predictors (MLP, from scratch), tuned on a validation split, test split used once.
Usage: python full/real3_model.py <bs2_scat0.npz | bs2_scat03.npz | bs1_scat0.npz> test <outprefix>
  -> <outprefix>_results.json, <outprefix>_choices.npz (reference copies in results/real3/). Several CPU hours (not timed precisely)."""
import sys, json, time, numpy as np, torch, torch.nn as nn
from scipy.spatial import cKDTree
torch.manual_seed(0)
NPZ, MODE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
z = np.load(NPZ, allow_pickle=True)
meta, cand, hf, hc = z['meta'], z['cand'], z['hf'], z['hc']
PW, TH, PH, TAU, HH = z['PW'], z['TH'], z['PH'], z['TAU'], z['Hhist']
bs = z['bs'].astype(float); lam = float(z['lam']); DS = list(z['spacings']); fsub = z['fsub']
N = len(hf); k = 2 * np.pi; c0 = 3e8; fc = c0 / lam
traj = meta[:, 0].astype(int); X = meta[:, 3:6]; VH = meta[:, 6:8]
S_true = np.real(hc * np.conj(hf)[:, None, None]) / np.abs(hf)[:, None, None] ** 2      # (N,3,4)
JP = DS.index(0.9)
# ---------- split ----------
first = {}
for i in range(N):
    if traj[i] not in first or meta[i, 1] < meta[first[traj[i]], 1]: first[traj[i]] = i
cell = {t: tuple(np.floor(X[i, :2] / 10).astype(int)) for t, i in first.items()}
cells = sorted(set(cell.values())); rng = np.random.default_rng(7); perm = rng.permutation(len(cells))
nt, nv = int(round(.30 * len(cells))), int(round(.15 * len(cells)))
role_cell = {cells[j]: ('test' if r < nt else 'val' if r < nt + nv else 'train') for r, j in enumerate(perm)}
role = np.array([role_cell[cell[t]] for t in traj]); grp = np.array([cells.index(cell[t]) for t in traj])
te = np.where(role == 'test')[0]; tree_te = cKDTree(X[te, :2])
dmin = tree_te.query(X[:, :2])[0]
keep = (role == 'test') | (dmin >= 2.0)
tr = np.where((role == 'train') & keep)[0]; va = np.where((role == 'val') & keep)[0]
EVAL = te if MODE == 'test' else va
print(f"cells {len(cells)} test {nt} val {nv}; queries train {len(tr)} val {len(va)} test {len(te)}", flush=True)
# ---------- surveys (nested thinning on train) ----------
order = rng.permutation(tr); surveys = {}; sel = []
for s in [20, 10, 5, 2]:
    for i in order:
        if not sel or np.min(np.linalg.norm(X[sel, :2] - X[i, :2], axis=1)) >= s:
            if i not in sel: sel.append(i)
    surveys[s] = np.array(sel)
surveys['full'] = tr.copy()
print({kk: len(v) for kk, v in surveys.items()}, flush=True)
# ---------- path helpers ----------
def paths(i):
    m = PW[i] > 0; w = PW[i][m]; th, ph, ta = TH[i][m], PH[i][m], TAU[i][m]
    u3 = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1)
    return w / w.sum(), u3, ta
SIGN = None
def phi_of(w, u3, R, sign):  # R: (4,2) in lambda
    return (w[None, :] * np.exp(sign * 1j * k * (R @ u3[:, :2].T))).sum(1)
def risk(phi):
    return 0.5 * (1 - phi.real / np.sqrt(np.maximum(1 - phi.imag ** 2, 1e-9)))
def spec_scores(w, u3, R): return (w[None, :] * np.cos(k * (R @ u3[:, :2].T))).sum(1)
# sign convention check on train only
def flip(i, j, scores, higher=True):
    idx = int(np.argmax(scores) if higher else np.argmin(scores)); return S_true[i, j, idx] < 0
fl = {+1: 0, -1: 0}
for i in tr[:2000]:
    w, u3, _ = paths(i)
    for sg in (+1, -1): fl[sg] += flip(i, JP, risk(phi_of(w, u3, cand[i, JP], sg)), higher=False)

# ---------- CKM mixtures ----------
_TREES = {}
def mixture(i, S, kk, p, vs):
    key = (len(S), int(S[0]), int(S[-1]))
    if key not in _TREES: _TREES[key] = cKDTree(X[S, :2])
    tree = _TREES[key]; d, idx = tree.query(X[i, :2], k=min(kk + 1, len(S))); d = np.atleast_1d(d); idx = np.atleast_1d(idx)
    m = S[idx] != i; d, idx = d[m][:kk], idx[m][:kk]   # leave-one-out when the query itself is a survey point
    wts = 1 / np.maximum(d, 0.1) ** p; wts = wts / wts.sum(); W, U, T = [], [], []
    for wt, j in zip(wts, S[idx]):
        w, u3, ta = paths(j)
        if vs:
            vsrc = X[j][None, :] + u3 * (c0 * ta)[:, None]; vec = vsrc - X[i][None, :]; dist = np.linalg.norm(vec, axis=1)
            u3 = vec / dist[:, None]; w = w * (c0 * ta / np.maximum(dist, 1e-3)) ** 2; w = w / w.sum(); ta = dist / c0
        W.append(wt * w); U.append(u3); T.append(ta)
    return np.concatenate(W), np.concatenate(U), np.concatenate(T)
def ckm_choice(i, j, S, kk, p, vs, use_risk):
    w, u3, _ = mixture(i, S, kk, p, vs); R = cand[i, j]
    return int(np.argmin(risk(phi_of(w, u3, R, SIGN)))) if use_risk else int(np.argmax(spec_scores(w, u3, R)))
def vs_hist_choice(i, j, S, kk, p, L=24, ridge=1e-3, sign=None):
    return int(np.argmax(vs_hist_scores(i, j, S, kk, p, L, ridge, sign)))
def vs_hist_scores(i, j, S, kk, p, L=24, ridge=1e-3, sign=None):
    sg = SIGN if sign is None else sign
    w, u3, ta = mixture(i, S, kk, p, True); o = np.argsort(-w)[:L]; u3, ta = u3[o], ta[o]
    dx = np.array([-(kk_ + 1) * VH[i] * 1.2 * 0.04 for kk_ in range(HH.shape[1])])          # meters, horizontal
    Hk = HH[i]; scale = np.abs(Hk).mean() + 1e-30
    rows = []
    for kidx in range(Hk.shape[0]):
        ph_sp = np.exp(sg * 1j * k * (dx[kidx] / lam) @ u3[:, :2].T)            # (L,)
        rows.append(np.exp(-2j * np.pi * fsub[:, None] * ta[None, :]) * ph_sp[None, :])   # (NF,L)
    A = np.concatenate(rows, 0); y = (Hk / scale).reshape(-1)
    beta = np.linalg.solve(A.conj().T @ A + ridge * np.eye(len(ta)), A.conj().T @ y)
    R = cand[i, j]; hfp = beta.sum(); hp = (beta[None, :] * np.exp(sg * 1j * k * (R @ u3[:, :2].T))).sum(1)
    return np.real(hp * np.conj(hfp)) / (np.abs(hfp) ** 2 + 1e-30)
SIGN = +1
fl2 = {+1: 0, -1: 0}
for i in tr[:400]:
    for sg in (+1, -1): fl2[sg] += S_true[i, JP, int(np.argmax(vs_hist_scores(i, JP, tr, 8, 1, sign=sg)))] < 0
SIGN = +1 if fl2[+1] <= fl2[-1] else -1
print("sign check A2 (VS-Hist train flips):", fl2, "-> SIGN", SIGN, "| old risk check", fl, flush=True)
# ---------- simple methods ----------
def euc(i, j): return int(np.argmin(np.linalg.norm(cand[i, j], axis=1)))
def bsdir(i, j):
    u = (bs - X[i])[:2] / np.linalg.norm(bs - X[i]); return int(np.argmax(np.cos(k * cand[i, j] @ u)))
def dense(i, j, use_risk):
    w, u3, _ = paths(i); R = cand[i, j]
    return int(np.argmin(risk(phi_of(w, u3, R, SIGN)))) if use_risk else int(np.argmax(spec_scores(w, u3, R)))
def orc(i, j): return int(np.argmax(S_true[i, j]))
# ---------- neural ----------
def feat_h0(I, j):
    rel = (bs[None, :] - X[I]); rel = rel / np.linalg.norm(rel, axis=1, keepdims=True)
    base = np.concatenate([X[I, :2] / 200.0, rel, VH[I]], 1)                              # (n,7)
    return np.concatenate([np.repeat(base[:, None, :], 4, 1), cand[I, j]], 2)              # (n,4,9)
def feat_hist(I):
    H = HH[I]; ref = H[:, 0, H.shape[2] // 2][:, None, None]; H = H / (ref + 1e-30)
    H = H / np.maximum(np.abs(H).mean((1, 2), keepdims=True), 1e-9)
    return np.concatenate([H.real.reshape(len(I), -1), H.imag.reshape(len(I), -1)], 1)    # (n,512)
def feat_delay(I):
    T = np.fft.ifft(HH[I], axis=2)[:, :, :8]                                  # (n,8,8) taps
    ref = T[:, 0, :][np.arange(len(I)), np.abs(T[:, 0, :]).argmax(1)][:, None, None]
    T = T / (ref + 1e-30); T = T / np.maximum(np.abs(T).max((1, 2), keepdims=True), 1e-9)
    return np.concatenate([T.real.reshape(len(I), -1), T.imag.reshape(len(I), -1)], 1)   # (n,128)
PHYS = {}
def phys_scores(I, j, S, cfg, use_hist):
    out = np.zeros((len(I), 4, 4 if use_hist else 3))
    for n, i in enumerate(I):
        w, u3, _ = mixture(i, S, cfg['IDW'][0], cfg['IDW'][1], False); out[n, :, 0] = spec_scores(w, u3, cand[i, j])
        w, u3, _ = mixture(i, S, cfg['VS'][0], cfg['VS'][1], True); out[n, :, 1] = spec_scores(w, u3, cand[i, j]); out[n, :, 2] = -risk(phi_of(w, u3, cand[i, j], SIGN))
        if use_hist: out[n, :, 3] = np.clip(vs_hist_scores(i, j, S, cfg['VS'][0], cfg['VS'][1]), -3, 3)
    return out
def build(I, j, variant, use_hist, S, cfg):
    fc_ = feat_h0(I, j); fh = None
    if variant == 'L1': fh = feat_hist(I) if use_hist else None
    if variant == 'L2': fh = feat_delay(I) if use_hist else None
    if variant == 'L3':
        fc_ = np.concatenate([fc_, phys_scores(I, j, S, cfg, use_hist)], 2); fh = feat_delay(I) if use_hist else None
    return fc_, fh
class Net(nn.Module):
    def __init__(s, dc, dh):
        super().__init__(); s.dh = dh
        s.hist = nn.Sequential(nn.Linear(dh, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU()) if dh else None
        s.head = nn.Sequential(nn.Linear(dc + (64 if dh else 0), 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(s, fc_, fh=None):
        if s.dh: e = s.hist(fh)[:, None, :].expand(-1, fc_.shape[1], -1); fc_ = torch.cat([fc_, e], 2)
        return s.head(fc_).squeeze(-1)
def train_net(S, use_hist, epochs=60, variant='L1', cfg=None):
    parts = [build(S, j, variant, use_hist, S, cfg) for j in range(3)]
    Fc = np.concatenate([p_[0] for p_ in parts], 0); Y = np.concatenate([(S_true[S, j] > 0).astype(np.float32) for j in range(3)], 0)
    Fh = np.concatenate([p_[1] for p_ in parts], 0) if parts[0][1] is not None else None
    Vc, Vh = build(va, JP, variant, use_hist, S, cfg)
    net = Net(Fc.shape[2], Fh.shape[1] if Fh is not None else 0); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    t = lambda a: torch.tensor(a, dtype=torch.float32); best, state, bad = 1e9, None, 0
    for ep in range(epochs):
        net.train(); pr = torch.randperm(len(Fc))
        for b in range(0, len(Fc), 256):
            ii = pr[b:b + 256]; opt.zero_grad()
            out = net(t(Fc[ii]), t(Fh[ii]) if Fh is not None else None); loss = nn.functional.binary_cross_entropy_with_logits(out, t(Y[ii])); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): sc = net(t(Vc), t(Vh) if Vh is not None else None).numpy()
        vflip = np.mean(S_true[va, JP, sc.argmax(1)] < 0)
        if vflip < best - 1e-4: best, state, bad = vflip, {kk: v.clone() for kk, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 10: break
    net.load_state_dict(state); return net
def net_choice(net, I, j, use_hist, variant='L1', S=None, cfg=None):
    t = lambda a: torch.tensor(a, dtype=torch.float32); fc_, fh = build(I, j, variant, use_hist, S, cfg)
    with torch.no_grad(): sc = net(t(fc_), t(fh) if fh is not None else None).numpy()
    return sc.argmax(1)
# ---------- tuning on val (CKM k,p) ----------
GRID = [(kk, p) for kk in (1, 2, 4, 8) for p in (0, 1, 2)]
def rate(choices, I, j): return float(np.mean(S_true[I, j, choices] < 0))
results = {'n_cells': len(cells), 'n_test_cells': nt, 'n_queries': {'train': int(len(tr)), 'val': int(len(va)), 'test': int(len(te))},
           'survey_sizes': {str(a): int(len(b)) for a, b in surveys.items()}, 'sign': SIGN, 'mode': MODE}
choice_store = {}
VA_SUB = va[:: max(1, len(va) // 300)]
for dens in ['full', 10, 20, 5, 2]:
    S = surveys[dens]; t0 = time.time(); best = {}
    for fam, vs, rk in [('IDW-Spec', False, False), ('IDW-Risk', False, True), ('VS-Spec', True, False), ('VS-Risk', True, True)]:
        sc = [(rate(np.array([ckm_choice(i, JP, S, kk, p, vs, rk) for i in VA_SUB]), VA_SUB, JP), kk, p) for kk, p in GRID]
        best[fam] = min(sc)
    vsk, vsp = best['VS-Spec'][1], best['VS-Spec'][2]
    best['VS-Hist'] = (rate(np.array([vs_hist_choice(i, JP, S, vsk, vsp) for i in VA_SUB]), VA_SUB, JP), vsk, vsp)
    print(dens, 'val tuning', {f: (round(v[0], 3), v[1], v[2]) for f, v in best.items()}, 'time', round(time.time() - t0, 1), flush=True)
    h0fam = min(['IDW-Spec', 'IDW-Risk', 'VS-Spec', 'VS-Risk'], key=lambda f: best[f][0])
    h1fam = min(['IDW-Spec', 'IDW-Risk', 'VS-Spec', 'VS-Risk', 'VS-Hist'], key=lambda f: best[f][0])
    cfg = {'IDW': best['IDW-Spec'][1:], 'VS': best['VS-Spec'][1:]}
    if dens == 'full':
        VSEL = {}
        for uh in (False, True):
            sc_ = {}
            for var in ('L1', 'L2', 'L3'):
                nt_ = train_net(S, uh, variant=var, cfg=cfg); sc_[var] = rate(net_choice(nt_, va, JP, uh, var, S, cfg), va, JP)
            VSEL[uh] = min(sc_, key=sc_.get); print('variant selection (val, full, 0.9)', 'H1' if uh else 'H0', sc_, '->', VSEL[uh], flush=True)
            results['variant_selection_' + ('H1' if uh else 'H0')] = sc_
    nets = {'NeuralCKM-H0': (train_net(S, False, variant=VSEL[False], cfg=cfg), VSEL[False]), 'Learner-H1': (train_net(S, True, variant=VSEL[True], cfg=cfg), VSEL[True])} if len(S) >= 30 else {}
    for j, d in enumerate(DS):
        if dens not in ('full', 10) and j != JP: continue
        ch = {'Euc': [euc(i, j) for i in EVAL], 'BSdir3D': [bsdir(i, j) for i in EVAL], 'denseSpec': [dense(i, j, False) for i in EVAL],
              'denseRisk': [dense(i, j, True) for i in EVAL], 'Orc': [orc(i, j) for i in EVAL]}
        for f in ['IDW-Spec', 'IDW-Risk', 'VS-Spec', 'VS-Risk']:
            _, kk, p = best[f]; ch[f] = [ckm_choice(i, j, S, kk, p, f.startswith('VS'), f.endswith('Risk')) for i in EVAL]
        ch['VS-Hist'] = [vs_hist_choice(i, j, S, vsk, vsp) for i in EVAL]
        ch['StrongCKM-H0'] = ch[h0fam]; ch['StrongCKM-H1'] = ch[h1fam]
        for nm, (net, var) in nets.items(): ch[nm] = list(net_choice(net, EVAL, j, nm.endswith('H1'), var, S, cfg))
        choice_store[f'{dens}_d{d}'] = {m: np.array(v) for m, v in ch.items()}
        results[f'{dens}_d{d}'] = {'rates': {m: rate(np.array(v), EVAL, j) for m, v in ch.items()}, 'strong_h0': h0fam, 'strong_h1': h1fam,
                                    'val_tuning': {f: [float(v[0]), int(v[1]), int(v[2])] for f, v in best.items()}}
        print(dens, d, {m: round(r, 3) for m, r in results[f'{dens}_d{d}']['rates'].items()}, flush=True)
# ---------- cluster bootstrap (paired) ----------
g_eval = grp[EVAL]; ug = np.unique(g_eval); B = 2000; brng = np.random.default_rng(11)
idx_by_g = {g: np.where(g_eval == g)[0] for g in ug}
def boot_diff(key, a, b, j):
    ca, cb = choice_store[key][a], choice_store[key][b]
    fa = (S_true[EVAL, j, ca] < 0).astype(float); fb = (S_true[EVAL, j, cb] < 0).astype(float); out = []
    for _ in range(B):
        gs = brng.choice(ug, len(ug)); ii = np.concatenate([idx_by_g[g] for g in gs]); out.append(fa[ii].mean() - fb[ii].mean())
    return [float(np.mean(fa) - np.mean(fb)), *np.quantile(out, [0.025, 0.975]).tolist()]
results['n_eval_groups'] = int(len(ug))
for key in choice_store:
    d = float(key.split('_d')[1]); j = DS.index(d); cs = choice_store[key]; diffs = {}
    for a, b in [('StrongCKM-H0', 'NeuralCKM-H0'), ('StrongCKM-H1', 'Learner-H1'), ('StrongCKM-H0', 'denseSpec'), ('Euc', 'denseSpec'), ('denseSpec', 'Orc'), ('denseSpec', 'denseRisk')]:
        if a in cs and b in cs: diffs[f'{a} - {b}'] = boot_diff(key, a, b, j)
    results[key]['paired_CI'] = diffs
json.dump(results, open(OUT + '_results.json', 'w'), indent=1)
np.savez_compressed(OUT + '_choices.npz', EVAL=EVAL, **{f'{k_}__{m}': v for k_, cs in choice_store.items() for m, v in cs.items()})
print('done', OUT, flush=True)
