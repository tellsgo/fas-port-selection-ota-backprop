"""Draw Figs. 1-5 of the paper from the shipped results.

Usage: python scripts/make_figures.py [outdir]
Fig. 1 certificate_region.pdf  : constants from scripts/certificate_example.py
Fig. 2 reversal_vs_spacing.pdf : results/extract/fig_bs{1,2}_scat0.npz (9600 queries per site)
Fig. 3 reversal_map.pdf        : results/extract/fig_bs2_scat0.npz
Fig. 4 per_site.pdf            : results/xscene/xtest_results.json
Fig. 5 s_cdf.pdf               : results/real3/test_bs2_scat0_choices.npz + results/extract/test_bs2_scat0.npz
Also writes fig2_reversal_vs_spacing.csv with the plotted rates.
"""
import os, sys, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); RES = os.path.join(ROOT, 'results')
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'out'); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'figure.dpi': 200})

# Fig. 2
fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.2)); rows = ['site,rule,' + ','.join(f'd{d}' for d in (0.7, 0.9, 1.2))]
for f, ls, lab in [('fig_bs2_scat0.npz', '-', 'low site'), ('fig_bs1_scat0.npz', '--', 'high site')]:
    z = np.load(os.path.join(RES, 'extract', f)); d = z['spacings']; fl = z['flags']
    for r, (m, mk) in enumerate([('Nearest', 'o'), ('Spec', 's'), ('Orc', '^')]):
        rate = fl[:, :, r].mean(0); ax.plot(d, rate, ls, marker=mk, ms=3, label=f'{m}, {lab}')
        rows.append(f'{lab},{m},' + ','.join(f'{v:.4f}' for v in rate))
ax.set_xlabel('port spacing $d/\\lambda$'); ax.set_ylabel('reversal rate'); ax.legend(fontsize=5.5, ncol=2, frameon=False); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'reversal_vs_spacing.pdf'))
open(os.path.join(OUT, 'fig2_reversal_vs_spacing.csv'), 'w').write('\n'.join(rows) + '\n')

# Fig. 3
z = np.load(os.path.join(RES, 'extract', 'fig_bs2_scat0.npz')); X = z['xy'].astype(float); fl = z['flags'][:, 1, 0].astype(float)   # d = 0.9, nearest port
fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.6)); g = 5.0; key = np.floor(X / g).astype(int); cells = {}
for kk, v in zip(map(tuple, key), fl): cells.setdefault(kk, []).append(v)
xs = np.array([kk[0] * g + g / 2 for kk in cells]); ys = np.array([kk[1] * g + g / 2 for kk in cells]); vs = np.array([np.mean(v) for v in cells.values()])
sc = ax.scatter(xs, ys, c=vs, cmap='RdYlBu_r', vmin=0, vmax=1, s=9, marker='s'); cb = fig.colorbar(sc, ax=ax); cb.set_label('nearest-port reversal rate', fontsize=7)
ax.plot(*z['bs'][:2], marker='*', ms=8, c='k', ls='none', label='base station'); ax.set_aspect('equal'); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.legend(fontsize=6, frameon=False, loc='lower right')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'reversal_map.pdf'))

# Fig. 1
fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.2)); eps = np.linspace(0, 0.2, 200)
ax.fill_between(eps, 0, np.maximum((0.1767 - 1.3094 * eps) / (2 * np.pi), 0), alpha=.35, label='certified: nearer port has negative mean correlation')
ax.plot(eps, np.maximum((0.7528 - 2.3458 * eps) / (2 * np.pi), 0), '--', label='boundary for a positive mean correlation of the farther port')
ax.set_xlabel('second-moment error $\\varepsilon$'); ax.set_ylabel('position error $\\eta/\\lambda$'); ax.set_ylim(0, 0.13); ax.legend(fontsize=6, frameon=False); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'certificate_region.pdf'))

# Fig. 4
o = json.load(open(os.path.join(RES, 'xscene', 'xtest_results.json')))['summary']['0.9']['by_scene_site']
order = ['chicago_low', 'etoile_low', 'florence_low', 'munich_low', 'san_francisco_low', 'chicago_high', 'etoile_high', 'florence_high', 'munich_high', 'san_francisco_high']
lab = {'chicago': 'CHI', 'etoile': 'ETO', 'florence': 'FLO', 'munich': 'MUC', 'san_francisco': 'SF'}
names = [lab[s.rsplit('_', 1)[0]] + ('\nlow' if s.endswith('low') else '\nhigh') for s in order]
rules = [('Euc', 'Nearest'), ('LoS3D', 'LoS rule'), ('WM-H0', 'Learned, other cities'), ('denseSpec', 'Spec (reference)'), ('Orc', 'Orc (reference)')]
fig, ax = plt.subplots(figsize=(7.0, 2.2)); x = np.arange(len(order)); w = 0.16
for i, (k, n) in enumerate(rules): ax.bar(x + (i - 2) * w, [o[s][k] for s in order], w, label=n)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=6.5); ax.set_ylabel('reversal rate'); ax.legend(fontsize=6, ncol=5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.18)); ax.grid(axis='y', alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'per_site.pdf'))

# Fig. 5
c = np.load(os.path.join(RES, 'real3', 'test_bs2_scat0_choices.npz')); t = np.load(os.path.join(RES, 'extract', 'test_bs2_scat0.npz'))
S = np.real(t['hc'] * np.conj(t['hf'])[:, None, None]) / np.abs(t['hf'])[:, None, None] ** 2; n = len(S)
fig, ax = plt.subplots(figsize=(3.4, 2.3))
for m, nm in [('Euc', 'Nearest'), ('BSdir3D', 'LoS rule'), ('StrongCKM-H0', 'Selected map'), ('VS-Hist', 'VS-Hist'), ('denseSpec', 'Spec (reference)'), ('Orc', 'Orc (reference)')]:
    s = np.sort(np.clip(S[np.arange(n), 1, c[f'full_d0.9__{m}']], -3, 3)); ax.plot(s, np.arange(1, n + 1) / n, label=nm)
ax.axvline(0, c='k', lw=.6); ax.set_xlim(-3, 3); ax.set_xlabel('alignment factor $s$ (clipped)'); ax.set_ylabel('CDF'); ax.legend(fontsize=6, frameon=False); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 's_cdf.pdf'))
print('figures written to', OUT)
