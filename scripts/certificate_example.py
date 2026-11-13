"""Second-moment certificate for the example of Section IV (C = diag(0.9, 0.1), r1 = (0.30, 0) lambda, r2 = (0, 0.36) lambda).

Computes
  * the exact range of the mean real correlation c over all planar spectra with this second moment
    (Jensen lower bound and chord upper bound, both attained), quoted in the text as
    c(r1) in [-0.2158, -0.1781] and c(r2) in [0.7549, 0.8363];
  * the dual certificate vectors from a 7200-point linear program, made rigorous by a Lipschitz shift,
    and their sensitivities. These give the constants of Fig. 1:
        1.3094 * eps + 2*pi*eta/lambda < 0.1767   keeps c(r1) negative,
        2.3458 * eps + 2*pi*eta/lambda < 0.7529   keeps c(r2) positive.
Usage: python scripts/certificate_example.py [out.json]
"""
import sys, json, warnings, numpy as np
from scipy.optimize import linprog
warnings.filterwarnings('ignore')
k = 2 * np.pi
N = 7200; th = np.linspace(0, 2 * np.pi, N, endpoint=False); h = 2 * np.pi / N
phi = np.vstack([np.ones(N), np.cos(2 * th), np.sin(2 * th)])          # 3 x N
a, b = 0.9 - 0.1, 0.0                                                  # E[cos 2theta], E[sin 2theta] for C = diag(0.9, 0.1)


def f(r, t): return np.cos(k * (np.cos(t) * r[0] + np.sin(t) * r[1]))


def dual(r, upper):
    s = 1 if upper else -1
    res = linprog(s * np.array([1, a, b]), A_ub=(-s * phi).T, b_ub=-s * f(r, th), bounds=[(None, None)] * 3, method='highs')
    return res.x


def certify(r, upper):
    y = dual(r, upper)
    g = (y @ phi - f(r, th)) * (1 if upper else -1)                    # must be >= 0
    L = 2 * abs(y[1]) + 2 * abs(y[2]) + k * np.hypot(*r)                # Lipschitz bound of g in theta
    slack = L * h / 2 - g.min()                                        # shift y0 so that g >= 0 on the whole circle
    y = y.copy(); y[0] += slack if upper else -slack
    bound = y @ np.array([1, a, b])
    tt = np.linspace(0, 2 * np.pi, 10 * N, endpoint=False); pp = np.vstack([np.ones_like(tt), np.cos(2 * tt), np.sin(2 * tt)])
    gg = (y @ pp - f(r, tt)) * (1 if upper else -1)
    return y, bound, gg.min()


def exact_range(q, Et):
    """Range of E[cos(q sqrt(t))] over distributions of t in [0,1] with mean Et (F convex for q <= pi)."""
    return float(np.cos(q * np.sqrt(Et))), float(1 - Et + Et * np.cos(q))


out = {'exact_range_c': {'r1': exact_range(k * 0.30, 0.9), 'r2': exact_range(k * 0.36, 0.1)}}
for name, r, up in [('r1_upper', np.array([.30, 0]), True), ('r2_lower', np.array([0, .36]), False)]:
    y, B, chk = certify(r, up)
    sens = 2 * abs(y[1]) + 2 * abs(y[2])
    out[name] = {'bound': float(B), 'y': y.tolist(), 'fine_grid_min_slack': float(chk), 'eps_sensitivity': float(sens)}
    print(f'{name}: bound {B:.4f}, eps sensitivity {sens:.4f}, y = {np.round(y, 4)}, fine-grid slack {chk:.1e}')
print('exact range of c: r1 [%.4f, %.4f], r2 [%.4f, %.4f]' % (*out['exact_range_c']['r1'], *out['exact_range_c']['r2']))
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else 'certificate_example.json', 'w'), indent=1)
