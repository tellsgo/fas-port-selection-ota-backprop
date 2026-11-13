"""Extract the small per-query files that quick mode uses from the large ray-traced data sets.

Usage:
    python full/extract_small.py <dir with bs1_scat0.npz and bs2_scat0.npz> results/extract

Writes
    fig_bs1_scat0.npz, fig_bs2_scat0.npz : per query and spacing, whether the nearest port, Spec and Orc reverse
                                          (s < 0), plus the query positions and the base station (Figs. 2 and 3)
    test_bs2_scat0.npz                   : forward channel, candidate channels and history rms at the 2400 test
                                          queries of the unsurveyed-position test, with their 10 m cluster ids
                                          (Tables III and V, Fig. 5, training surrogates, the 0.32 -> 0.27 interval)
"""
import sys, os, numpy as np
from scipy.spatial import cKDTree

SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
k = 2 * np.pi


def s_true(z):
    hf, hc = z['hf'], z['hc']
    return np.real(hc * np.conj(hf)[:, None, None]) / np.abs(hf)[:, None, None] ** 2


# ---- figure data (all 9600 queries per site) ----
for name in ('bs1_scat0', 'bs2_scat0'):
    z = np.load(os.path.join(SRC, name + '.npz'))
    S = s_true(z); cand = z['cand']; PW, TH, PH = z['PW'], z['TH'], z['PH']; N = len(S)
    flags = np.zeros((N, 3, 3), np.uint8)          # (query, spacing, rule) rule = Nearest, Spec, Orc
    for j in range(3):
        e = np.argmin(np.linalg.norm(cand[:, j], axis=2), 1)
        flags[:, j, 0] = S[np.arange(N), j, e] < 0
        flags[:, j, 2] = S[:, j].max(1) < 0
        for i in range(N):
            m = PW[i] > 0; w = PW[i][m] / PW[i][m].sum()
            uh = np.c_[np.sin(TH[i][m]) * np.cos(PH[i][m]), np.sin(TH[i][m]) * np.sin(PH[i][m])]
            sp = int(np.argmax((w[None, :] * np.cos(k * cand[i, j] @ uh.T)).sum(1)))
            flags[i, j, 1] = S[i, j, sp] < 0
    np.savez_compressed(os.path.join(OUT, f'fig_{name}.npz'), flags=flags, rules=np.array(['Nearest', 'Spec', 'Orc']),
                        xy=z['meta'][:, 3:5].astype(np.float32), bs=z['bs'], spacings=z['spacings'])
    print('wrote', f'fig_{name}.npz', flush=True)

# ---- unsurveyed-position test set (same split as full/real3_model.py) ----
z = np.load(os.path.join(SRC, 'bs2_scat0.npz'))
meta = z['meta']; N = len(z['hf']); traj = meta[:, 0].astype(int); X = meta[:, 3:6]
first = {}
for i in range(N):
    if traj[i] not in first or meta[i, 1] < meta[first[traj[i]], 1]: first[traj[i]] = i
cell = {t: tuple(np.floor(X[i, :2] / 10).astype(int)) for t, i in first.items()}
cells = sorted(set(cell.values())); rng = np.random.default_rng(7); perm = rng.permutation(len(cells))
nt, nv = int(round(.30 * len(cells))), int(round(.15 * len(cells)))
role_cell = {cells[j]: ('test' if r < nt else 'val' if r < nt + nv else 'train') for r, j in enumerate(perm)}
role = np.array([role_cell[cell[t]] for t in traj]); grp = np.array([cells.index(cell[t]) for t in traj])
EVAL = np.where(role == 'test')[0]
HH = z['Hhist']
np.savez_compressed(os.path.join(OUT, 'test_bs2_scat0.npz'), EVAL=EVAL, grp=grp[EVAL], hf=z['hf'][EVAL], hc=z['hc'][EVAL],
                    hist_rms=np.sqrt(np.mean(np.abs(HH[EVAL]) ** 2, axis=(1, 2))), spacings=z['spacings'])
print('wrote test_bs2_scat0.npz', len(EVAL), 'queries in', len(np.unique(grp[EVAL])), 'cells', flush=True)
