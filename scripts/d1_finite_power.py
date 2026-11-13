"""Training through the scalar link with truncated channel inversion (extended version, arXiv appendix: finite-power link).
Each sample uses the paired ray-traced forward and backward channels of the port chosen by each rule at the 2400
unsurveyed test queries (Chicago low site, d = 0.9 lambda, full survey); forward noise 20 dB, channel-estimate
error -20 dB, truncation |h|^2 < 0.1, backward SNR 20 dB and 10 dB, binary digits task, 2000 steps, ten seeds.
The reference gradient includes Re(h_f/hhat) (chain rule through the estimated equalizer).
Usage: python scripts/d1_finite_power.py [out.json]"""
import sys, os, json, numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, t as tdist
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); OUT = sys.argv[1] if len(sys.argv) > 1 else 'd1_results.json'
z = np.load(os.path.join(ROOT, 'results', 'extract', 'test_bs2_scat0.npz')); c = np.load(os.path.join(ROOT, 'results', 'real3', 'test_bs2_scat0_choices.npz'))
hf, hc = z['hf'], z['hc']; JP = list(z['spacings']).index(0.9)
g = z['hist_rms']                                                       # local rms from the forward history
HF = hf / g
RULES = {'Nearest': 'Euc', 'LoS': 'BSdir3D', 'Map': 'StrongCKM-H0', 'VS-Hist': 'StrongCKM-H1', 'Orc': 'Orc'}
HR = {m: hc[np.arange(len(hf)), JP, c[f'full_d0.9__{k}']] / g for m, k in RULES.items()}
HR['Ideal'] = HF.copy()
FLIP = {m: float(np.mean(np.real(HR[m] * np.conj(HF)) < 0)) for m in HR}
X, y = load_digits(return_X_y=True); y = np.where(y < 5, 1.0, -1.0)
Xa, Xte, ya, yte = train_test_split(X, y, test_size=197, random_state=0); Xtr, Xva, ytr, yva = train_test_split(Xa, ya, test_size=200, random_state=0)
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; f = lambda A: np.c_[(A - mu) / sd, np.ones(len(A))]; Xtr, Xva, Xte = f(Xtr), f(Xva), f(Xte)
SNRF, NMSE, TAU = 20, -20, 0.1
def train(m, lr, seed, snrb, T=2000, B=32):
    r = np.random.default_rng(seed); w = np.zeros(Xtr.shape[1]); cs = []; pr = []
    for it in range(T):
        b = r.integers(0, len(Xtr), B); x, yy = Xtr[b], ytr[b]; q = x @ w
        qi = r.integers(0, len(HF), B); h_f, h_r = HF[qi], HR[m][qi]
        sf = np.sqrt(max(np.mean(q ** 2), 1e-6) / 10 ** (SNRF / 10) / 2)
        yrx = h_f * q + sf * (r.standard_normal(B) + 1j * r.standard_normal(B))
        e = np.sqrt(np.mean(np.abs(HF) ** 2) * 10 ** (NMSE / 10) / 2) * (r.standard_normal(B) + 1j * r.standard_normal(B)); hh = h_f + e
        zq = np.real(yrx / hh); Lp = -yy / (1 + np.exp(yy * zq))                  # derivative of log(1+exp(-y z))
        delta = Lp / np.conj(hh); on = np.abs(hh) ** 2 >= TAU                    # truncated inversion
        pw = np.mean(np.abs(delta[on]) ** 2) if on.any() else 1.0
        sb = np.sqrt(pw / 10 ** (snrb / 10) / 2)
        rx = h_r * np.conj(delta) + sb * (r.standard_normal(B) + 1j * r.standard_normal(B))
        coef = np.where(on, np.real(np.conj(rx)), 0.0)
        gr = coef @ x / B; ge = (Lp * np.real(h_f / hh)) @ x / B
        ng = np.linalg.norm(gr) * np.linalg.norm(ge)
        if ng > 0: cs.append(gr @ ge / ng); pr.append(gr @ ge / (ge @ ge))
        w -= lr * gr
        if not np.all(np.isfinite(w)): return 0.5, (float(np.mean(cs)) if cs else 0.0, float(np.median(pr)) if pr else 0.0), w
    return None, (float(np.mean(cs)), float(np.median(pr))), w
def acc(w, Xe, ye): return float(np.mean(np.sign(Xe @ w + 1e-12) == ye))
out = {'flip_rate_paired': FLIP, 'n_queries': int(len(HF)), 'truncated_fraction_ideal': float(np.mean(np.abs(HF) ** 2 < TAU)), 'snr': {}}
for snrb in (20, 10):
    res = {}
    for m in HR:
        va = {}
        for lr in (0.01, 0.03, 0.1, 0.3, 1.0):
            v = []
            for s in range(3):
                bad, _, w = train(m, lr, s, snrb); v.append(0.5 if bad is not None else acc(w, Xva, yva))
            va[lr] = float(np.mean(v))
        lr = max(va, key=va.get); te, cs, prs = [], [], []
        for s in range(100, 110):
            bad, cm, w = train(m, lr, s, snrb); te.append(0.5 if bad is not None else acc(w, Xte, yte)); cs.append(cm[0]); prs.append(cm[1])
        te = np.array(te); hw = tdist.ppf(.975, len(te) - 1) * te.std(ddof=1) / np.sqrt(len(te))
        res[m] = {'lr': lr, 'val': va, 'test_mean': float(te.mean()), 'test_ci95': [float(te.mean() - hw), float(te.mean() + hw)], 'test_seeds': te.tolist(), 'cos_mean': float(np.mean(cs)), 'proj_median': float(np.mean(prs))}
        print(snrb, m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[m].items() if k not in ('val', 'test_seeds')}, flush=True)
    rk = [m for m in RULES if m != 'Orc'] + ['Orc']
    res['P3_spearman_cos_vs_flip'] = float(spearmanr([res[m]['cos_mean'] for m in rk], [FLIP[m] for m in rk]).correlation)
    res['P1_nearest_below_all'] = bool(all(res['Nearest']['test_ci95'][1] < res[m]['test_ci95'][0] for m in ('LoS', 'Map', 'VS-Hist')))
    out['snr'][str(snrb)] = res
    json.dump(out, open(OUT, 'w'), indent=1)
print('done', OUT, flush=True)
