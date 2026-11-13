"""Recompute the tables of the paper from the shipped per-query results and compare them with the printed values.

Usage: python scripts/tables.py [outdir]

Table I   (pedestrian motion, Chicago)          : read from results/real4/*.json (needs ray tracing to regenerate)
Table II  (unsurveyed positions, Chicago low)   : recomputed from results/real3/test_bs2_scat0_choices.npz
                                                  and results/extract/test_bs2_scat0.npz
Table III (five cities, zero survey)            : recomputed from results/xscene/xtest_per_query.csv.gz
Table IV  (mean clipped alignment)              : recomputed as for Table II
Text      paired cluster-bootstrap intervals    : recomputed with the same resampling code and seeds
          vehicular quadratic-vs-Spec interval  : recomputed from results/real2_repro/per_sample_repro.json
Tolerances: 5e-4 for three-decimal table entries (rounding), 0.05 points for one-decimal intervals.
Every check prints PASS or FAIL; the script exits with status 1 if any check fails.
"""
import os, sys, json, gzip, numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(ROOT, 'results')
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'out')
os.makedirs(OUT, exist_ok=True)
FAIL = []
LOG = []


def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def check(label, got, paper, tol):
    ok = abs(got - paper) <= tol
    say(f'  [{"PASS" if ok else "FAIL"}] {label}: recomputed {got:.4f}, paper {paper} (tol {tol})')
    if not ok: FAIL.append(label)


# ---------------- Table I ----------------
say('Table I: reversal rate, pedestrian motion, finite aperture, Chicago (from reference JSON)')
names = [('Nearest', 'Euc'), ('LoS rule', 'BSdir'), ('Quadratic (reference)', 'Cmet'), ('Spec (reference)', 'Spec'), ('Orc (reference)', 'Orc')]
r4 = {s: json.load(open(os.path.join(RES, 'real4', f'results_real4_bs{b}.json'))) for s, b in (('low', 2), ('high', 1))}
say('  rule'.ljust(26) + ''.join(f'{s} {d}'.ljust(11) for s in ('low', 'high') for d in (0.5, 0.7, 0.9)))
for n, key in names:
    say('  ' + n.ljust(24) + ''.join(f'{r4[s][f"d{d}"][key]:<11.3f}' for s in ('low', 'high') for d in (0.5, 0.7, 0.9)))

# ---------------- Table II and IV ----------------
c = np.load(os.path.join(RES, 'real3', 'test_bs2_scat0_choices.npz'))
x = np.load(os.path.join(RES, 'extract', 'test_bs2_scat0.npz'))
assert (c['EVAL'] == x['EVAL']).all()
S = np.real(x['hc'] * np.conj(x['hf'])[:, None, None]) / np.abs(x['hf'])[:, None, None] ** 2     # (2400, 3, 4)
DS = [0.7, 0.9, 1.2]; n = len(S); grp = x['grp']


def flips(key, rule):
    d = float(key.split('_d')[1]); return (S[np.arange(n), DS.index(d), c[f'{key}__{rule}']] < 0).astype(float)


say('Table II: reversal rate at unsurveyed positions, Chicago low site, d = 0.9 lambda (2400 test queries, 22 cells)')
T2 = [('Nearest', 'Euc', 0.494, 0.494), ('LoS rule', 'BSdir3D', 0.366, 0.366), ('Selected map without history', 'StrongCKM-H0', 0.319, 0.330),
      ('Learned without history', 'NeuralCKM-H0', 0.318, 0.318), ('VS-Hist', 'VS-Hist', 0.270, 0.265), ('Learned with history', 'Learner-H1', 0.267, 0.259),
      ('Spec (reference)', 'denseSpec', 0.215, 0.215), ('Orc (reference)', 'Orc', 0.090, 0.090)]
rows = []
for lab, rule, pf, p10 in T2:
    a, b = flips('full_d0.9', rule).mean(), flips('10_d0.9', rule).mean(); rows.append((lab, a, b))
    say(f'  {lab:30s} full {a:.3f}   10 m {b:.3f}')
    check(f'Table II {lab} full', a, pf, 5.01e-4); check(f'Table II {lab} 10 m', b, p10, 5.01e-4)
pd.DataFrame(rows, columns=['rule', 'full_survey', 'survey_10m']).to_csv(os.path.join(OUT, 'table2_unsurveyed.csv'), index=False)

say('Table IV: mean clipped alignment at unsurveyed positions, Chicago low site, full survey')
T4 = [('Nearest', 'Euc', (0.496, 0.031, -0.226)), ('LoS rule', 'BSdir3D', (0.390, 0.240, 0.219)), ('Selected map without history', 'StrongCKM-H0', (0.515, 0.325, 0.273)),
      ('VS-Hist', 'StrongCKM-H1', (0.569, 0.417, 0.306)), ('Learned with history', 'Learner-H1', (0.563, 0.408, 0.299)),
      ('Spec (reference)', 'denseSpec', (0.618, 0.501, 0.349)), ('Orc (reference)', 'Orc', (1.020, 0.868, 0.721))]
rows = []
for lab, rule, pv in T4:
    v = [float(np.clip(S[np.arange(n), j, c[f'full_d{d}__{rule}']], -3, 3).mean()) for j, d in enumerate(DS)]; rows.append((lab, *v))
    say(f'  {lab:30s} ' + '  '.join(f'{d}: {a:.3f}' for d, a in zip(DS, v)))
    for d, a, p in zip(DS, v, pv): check(f'Table IV {lab} d={d}', a, p, 5e-4)
pd.DataFrame(rows, columns=['rule', 'd0.7', 'd0.9', 'd1.2']).to_csv(os.path.join(OUT, 'table4_alignment.csv'), index=False)

# paired cluster bootstrap exactly as in full/real3_model.py (one generator, seed 11, keys in stored order)
say('Paired cluster-bootstrap intervals of the unsurveyed-position test (same code and seed as full/real3_model.py)')
ref = json.load(open(os.path.join(RES, 'real3', 'test_bs2_scat0_results.json')))
ug = np.unique(grp); idx_by_g = {g: np.where(grp == g)[0] for g in ug}; brng = np.random.default_rng(11)
keys = []
for f in c.files:
    if '__' in f and f.split('__')[0] not in keys: keys.append(f.split('__')[0])
worst = 0.0; mine = {}
for key in keys:
    rules = [f.split('__')[1] for f in c.files if f.startswith(key + '__')]; mine[key] = {}
    for a, b in [('StrongCKM-H0', 'NeuralCKM-H0'), ('StrongCKM-H1', 'Learner-H1'), ('StrongCKM-H0', 'denseSpec'), ('Euc', 'denseSpec'), ('denseSpec', 'Orc'), ('denseSpec', 'denseRisk')]:
        if a in rules and b in rules:
            fa, fb = flips(key, a), flips(key, b); bs = []
            for _ in range(2000):
                ii = np.concatenate([idx_by_g[g] for g in brng.choice(ug, len(ug))]); bs.append(fa[ii].mean() - fb[ii].mean())
            v = [float(fa.mean() - fb.mean()), *np.quantile(bs, [0.025, 0.975]).tolist()]; mine[key][f'{a} - {b}'] = v
            worst = max(worst, float(np.max(np.abs(np.array(v) - np.array(ref[key]['paired_CI'][f'{a} - {b}'])))))
check('all paired intervals of the reference JSON (max abs deviation)', worst, 0.0, 1e-9)
for key, plo, phi_ in [('full_d0.9', -1.2, 2.2), ('10_d0.9', -0.8, 2.4)]:
    m = 100 * np.array(mine[key]['StrongCKM-H1 - Learner-H1'])     # positive = learned predictor with history is better than VS-Hist
    say(f'  advantage of learned with history over VS-Hist, {key}: {m[0]:.1f} [{m[1]:.1f}, {m[2]:.1f}] points (paper [{plo}, {phi_}])')
    check(f'learned-with-history advantage {key} lower', m[1], plo, 0.05); check(f'learned-with-history advantage {key} upper', m[2], phi_, 0.05)
r5 = {r: flips('5_d0.9', r).mean() for r in ('StrongCKM-H0', 'VS-Hist', 'NeuralCKM-H0', 'Learner-H1')}
check('5 m survey: selected map', r5['StrongCKM-H0'], 0.315, 5e-4); check('5 m survey: VS-Hist', r5['VS-Hist'], 0.254, 5e-4)
for nm, pv, lo, hi in [('StrongCKM-H0 - NeuralCKM-H0', 2.7, 0.9, 5.3), ('StrongCKM-H1 - Learner-H1', 7.4, 3.7, 12.0)]:
    v = 100 * np.array(mine['5_d0.9'][nm]); say(f'  5 m survey {nm}: {v[0]:.1f} [{v[1]:.1f}, {v[2]:.1f}] (paper {pv} [{lo}, {hi}])')
    check(f'5 m survey {nm} point', v[0], pv, 0.05); check(f'5 m survey {nm} lower', v[1], lo, 0.05); check(f'5 m survey {nm} upper', v[2], hi, 0.05)

# 0.32 -> 0.27 interval, as in the error-budget script (fresh generator with seed 11, map = IDW-Risk(4,0), VS-Hist(2,0))
UG = np.unique(grp); IDX = {g: np.where(grp == g)[0] for g in UG}
fa, fb = flips('full_d0.9', 'StrongCKM-H0'), flips('full_d0.9', 'StrongCKM-H1'); rr = np.random.default_rng(11); d = []
for _ in range(2000):
    ii = np.concatenate([IDX[g] for g in rr.choice(UG, len(UG))]); d.append(fa[ii].mean() - fb[ii].mean())
lo, hi = 100 * np.quantile(d, [.025, .975])
say(f'  history gain 0.32 -> 0.27: {100*(fa.mean()-fb.mean()):.1f} points, interval [{lo:.1f}, {hi:.1f}] (paper [-0.3, 11.2])')
check('history gain interval lower', lo, -0.3, 0.05); check('history gain interval upper', hi, 11.2, 0.05)

# ---------------- Table III ----------------
say('Table III: zero-survey port selection, five cities, d = 0.9 lambda, ten sites pooled')
df = pd.read_csv(os.path.join(RES, 'xscene', 'xtest_per_query.csv.gz'))
sub = df[df.d == 0.9]
T3 = [('Nearest', 'Euc', 0.367, 0.200), ('Street axis', 'StreetAxis', 0.479, 0.022), ('LoS rule', 'LoS3D', 0.205, 0.496), ('Learned, other cities', 'WM-H0', 0.214, 0.461),
      ('Learned with history, other cities', 'WM-H1', 0.225, 0.423), ('Spec (reference)', 'denseSpec', 0.144, 0.603), ('Orc (reference)', 'Orc', 0.068, 0.827)]
rows = []
for lab, rule, pr, ps in T3:
    a, b = float((sub[rule] < 0).mean()), float(sub[rule].mean()); rows.append((lab, a, b))
    say(f'  {lab:36s} reversal {a:.3f}   E[s clipped] {b:.3f}')
    check(f'Table III {lab} reversal', a, pr, 5e-4); check(f'Table III {lab} mean s', b, ps, 5e-4)
pd.DataFrame(rows, columns=['rule', 'reversal_rate', 'mean_clipped_s']).to_csv(os.path.join(OUT, 'table3_five_cities.csv'), index=False)
say(f'  queries {len(sub)} (paper 38184), clusters {sub.cell.nunique()}')
check('Table III number of queries', len(sub), 38184, 0)
# paired intervals, same code and seed as full/xtest.py (one generator over spacings 0.7, 0.9, 1.2)
xref = json.load(open(os.path.join(RES, 'xscene', 'xtest_results.json')))['summary']; brng = np.random.default_rng(11); worst = 0.0
for dd in (0.7, 0.9, 1.2):
    s_ = df[df.d == dd]; ug = s_.cell.unique(); idx = {g: np.where(s_.cell.values == g)[0] for g in ug}
    for a, b in [('Rival-H0', 'WM-H0'), ('Rival-H1', 'WM-H1'), ('WM-H1', 'denseSpec')]:
        fa = (s_[a].values < 0).astype(float); fb = (s_[b].values < 0).astype(float); bs = []
        for _ in range(2000):
            gg = brng.choice(ug, len(ug)); ii = np.concatenate([idx[g] for g in gg]); bs.append(fa[ii].mean() - fb[ii].mean())
        v = [float(fa.mean() - fb.mean()), *np.quantile(bs, [.025, .975]).tolist()]
        worst = max(worst, float(np.max(np.abs(np.array(v) - np.array(xref[str(dd)][f'{a} - {b}'])))))
        if dd == 0.9 and a == 'Rival-H0': say(f'  LoS advantage over learned (0.9): {-100*v[0]:.1f} [{-100*v[2]:.1f}, {-100*v[1]:.1f}] points (paper 0.9 [0.7, 1.2])')
        if dd == 0.7 and a == 'Rival-H0': say(f'  learned advantage over LoS (0.7): {100*v[0]:.1f} [{100*v[1]:.1f}, {100*v[2]:.1f}] points (paper 1.1 [0.4, 2.0])')
check('all five-city paired intervals of the reference JSON (max abs deviation)', worst, 0.0, 1e-9)

# ---------------- vehicular displacement, quadratic vs Spec ----------------
say('Vehicular displacements, unrestricted lattice: quadratic rule minus Spec (paired bootstrap over 500 queries)')
per = json.load(open(os.path.join(RES, 'real2_repro', 'per_sample_repro.json')))
rng = np.random.default_rng(0)
for dd, pv, plo, phi_ in [(1.2, 7.8, 4.6, 11.2), (0.9, 1.2, -0.4, 3.0)]:
    P = [p for p in per if p['d'] == dd]; dlt = np.array([p['Cmet'] - p['Spec'] for p in P], float)
    bs = [dlt[rng.integers(0, len(dlt), len(dlt))].mean() for _ in range(2000)]; lo, hi = 100 * np.quantile(bs, [.025, .975])
    say(f'  d = {dd}: {100*dlt.mean():.1f} [{lo:.1f}, {hi:.1f}] points (reference {pv} [{plo}, {phi_}]; resampling seed differs, tolerance 0.6 points)')
    check(f'vehicular d={dd} point', 100 * dlt.mean(), pv, 0.05); check(f'vehicular d={dd} lower', lo, plo, 0.6); check(f'vehicular d={dd} upper', hi, phi_, 0.6)

open(os.path.join(OUT, 'tables_log.txt'), 'w').write('\n'.join(LOG) + '\n')
say(f'\n{"ALL CHECKS PASS" if not FAIL else "FAILED: " + "; ".join(FAIL)}')
sys.exit(1 if FAIL else 0)
