"""Compare a rerun of the training surrogates or the CDL check with the reference JSON.
Usage: python scripts/check_json.py <kind: e2e|d1|cdl> <rerun.json> <reference.json>"""
import sys, json
kind, a, b = sys.argv[1], json.load(open(sys.argv[2])), json.load(open(sys.argv[3])); bad = []
def cmp(label, x, y, tol):
    ok = abs(x - y) <= tol; print(f'  [{"PASS" if ok else "FAIL"}] {label}: rerun {x:.4f}, reference {y:.4f}')
    if not ok: bad.append(label)
if kind == 'e2e':
    for d in a:
        if d in ('0.9', '0.7'):
            for m in a[d]: cmp(f'd={d} {m} test accuracy', a[d][m]['test_acc_mean'], b[d][m]['test_acc_mean'], 1e-9)
elif kind == 'd1':
    for s in a['snr']:
        for m, v in a['snr'][s].items():
            if isinstance(v, dict): cmp(f'{s} dB {m} test accuracy', v['test_mean'], b['snr'][s][m]['test_mean'], 1e-9); cmp(f'{s} dB {m} gradient cosine', v['cos_mean'], b['snr'][s][m]['cos_mean'], 1e-9)
elif kind == 'cdl':
    for key in a:
        for m in a[key]: cmp(f'{key} {m}', a[key][m], b[key][m], 1e-9)
print('ALL CHECKS PASS' if not bad else f'FAILED: {len(bad)} entries'); sys.exit(1 if bad else 0)
