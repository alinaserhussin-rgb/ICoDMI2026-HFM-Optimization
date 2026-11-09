"""
=============================================================================
Comparative Analysis of GD+M, FLO, and Hybrid Fuzzy-Momentum (HFM) Optimizers
Author: Ali Nasser Al-Moussawi & Uday Abdulrazzaq
University of Babylon, Department of Mathematics
=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# =============================================================================
# SECTION 1: BENCHMARK FUNCTIONS
# =============================================================================

def rosenbrock(x, y):
    """Rosenbrock function: narrow curved valley, global min at (1,1)"""
    return (1 - x)**2 + 100*(y - x**2)**2

def rosenbrock_grad(x, y):
    gx = -2*(1 - x) - 400*x*(y - x**2)
    gy = 200*(y - x**2)
    return np.array([gx, gy])

def sphere(x, y):
    """Sphere function: strictly convex, global min at (0,0)"""
    return x**2 + y**2

def sphere_grad(x, y):
    return np.array([2*x, 2*y])

def himmelblau(x, y):
    """Himmelblau function: 4 global minima at f=0"""
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def himmelblau_grad(x, y):
    gx = 4*x*(x**2 + y - 11) + 2*(x + y**2 - 7)
    gy = 2*(x**2 + y - 11) + 4*y*(x + y**2 - 7)
    return np.array([gx, gy])

def ackley(x, y):
    """Ackley function: highly multimodal with smooth outer envelope"""
    return (-20*np.exp(-0.2*np.sqrt((x**2 + y**2)/2))
            - np.exp((np.cos(2*np.pi*x) + np.cos(2*np.pi*y))/2)
            + np.e + 20)

def ackley_grad(x, y, h=1e-5):
    return np.array([(ackley(x+h, y) - ackley(x-h, y))/(2*h),
                     (ackley(x, y+h) - ackley(x, y-h))/(2*h)])

def rastrigin(x, y):
    """Rastrigin function: dense regular grid of local minima"""
    return 20 + (x**2 - 10*np.cos(2*np.pi*x)) + (y**2 - 10*np.cos(2*np.pi*y))

def rastrigin_grad(x, y):
    return np.array([2*x + 20*np.pi*np.sin(2*np.pi*x),
                     2*y + 20*np.pi*np.sin(2*np.pi*y)])

def beale(x, y):
    """Beale function: flat regions with sharp transition to minimum at (3,0.5)"""
    return ((1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2
            + (2.625 - x + x*y**3)**2)

def beale_grad(x, y, h=1e-5):
    return np.array([(beale(x+h, y) - beale(x-h, y))/(2*h),
                     (beale(x, y+h) - beale(x, y-h))/(2*h)])

def rosenbrock_nd(x):
    """n-dimensional Rosenbrock: global min at (1,...,1)"""
    return sum(100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2
               for i in range(len(x)-1))

def rosenbrock_nd_grad(x):
    g = np.zeros_like(x)
    for i in range(len(x)-1):
        g[i]   += -400*x[i]*(x[i+1] - x[i]**2) - 2*(1 - x[i])
        g[i+1] +=  200*(x[i+1] - x[i]**2)
    return g


# =============================================================================
# SECTION 2: FUZZY INFERENCE SYSTEM (zero-order Sugeno-style weighted singleton)
# =============================================================================

def fuzzy_lr(gn):
    """
    Defuzzified learning rate from weighted singleton fuzzy inference.
    Input:  gn = gradient norm ‖∇f(x)‖
    Output: lr* ∈ [0.0005, 0.005]
    Rules:
      IF gradient is Small  THEN lr = 0.005  (large step)
      IF gradient is Medium THEN lr = 0.002  (medium step)
      IF gradient is Large  THEN lr = 0.0005 (small step)
    """
    mu_small  = max(min(gn/50, (200-gn)/150), 0)
    mu_medium = max(min((gn-50)/150, (500-gn)/300), 0)
    mu_large  = max(min((gn-200)/300, 1), 0)
    denom = mu_small + mu_medium + mu_large
    if denom < 1e-12:
        return 0.002
    return (mu_small*0.005 + mu_medium*0.002 + mu_large*0.0005) / denom


def safe(v):
    """Guard against NaN/Inf values"""
    return v if (np.isfinite(v) and v < 1e15) else 1e10


# =============================================================================
# SECTION 3: OPTIMIZER IMPLEMENTATIONS
# =============================================================================

def gdm(func, grad_func, x0, alpha=0.001, beta=0.9, max_iter=500, tol=1e-8):
    """
    Gradient Descent with Momentum (GD+M)
    Update rule:
        v_{t+1} = β·v_t - α·∇f(x_t)
        x_{t+1} = x_t + v_{t+1}
    Parameters:
        alpha = 0.001 (fixed learning rate)
        beta  = 0.9   (fixed momentum coefficient)
    """
    x = x0.copy().astype(float)
    v = np.zeros(2)
    history = [safe(func(*x))]
    grad_norms = []
    for _ in range(max_iter):
        g = grad_func(*x)
        if not np.all(np.isfinite(g)):
            break
        gn = np.linalg.norm(g)
        grad_norms.append(gn)
        if gn < tol:
            break
        v = beta*v - alpha*g
        xnew = x + v
        if not np.all(np.isfinite(xnew)):
            break
        x = xnew
        history.append(safe(func(*x)))
    return x, history, grad_norms


def flo(func, grad_func, x0, max_iter=500, tol=1e-8):
    """
    Fuzzy Logic Optimizer (FLO)
    Update rule:
        lr* = fuzzy_lr(‖∇f(x_t)‖)
        x_{t+1} = x_t - lr*·∇f(x_t)
    No momentum component (pure gradient with adaptive step size).
    """
    x = x0.copy().astype(float)
    history = [safe(func(*x))]
    grad_norms = []
    lr_history = []
    for _ in range(max_iter):
        g = grad_func(*x)
        if not np.all(np.isfinite(g)):
            break
        gn = np.linalg.norm(g)
        grad_norms.append(gn)
        if gn < tol:
            break
        lr = fuzzy_lr(gn)
        lr_history.append(lr)
        xnew = x - lr*g
        if not np.all(np.isfinite(xnew)):
            break
        x = xnew
        history.append(safe(func(*x)))
    return x, history, grad_norms, lr_history


def hnf(func, grad_func, x0, beta_min=0.3, max_iter=500, tol=1e-8):
    """
    Hybrid Fuzzy-Momentum Optimizer (HFM)
    Combines three mechanisms:
    1. Fuzzy step-size control (same as FLO)
    2. Cosine-similarity adaptive momentum:
         cos θ_t = (∇f(x_{t-1}) · ∇f(x_t)) / (‖∇f(x_{t-1})‖ · ‖∇f(x_t)‖)
         β_adaptive(t) = max(β_min, 0.9 × (0.5 + 0.5·cos θ_t))
    3. EMA backtracking:
         EMA_f(t) = 0.95·EMA_f(t-1) + 0.05·f(x_t)
         scale(t) = 1.0 if f(x_t) ≤ EMA_f(t), else 0.5
    Full update:
         v_{t+1} = β_adaptive(t)·v_t - scale(t)·lr*·∇f(x_t)
         x_{t+1} = x_t + v_{t+1}
    """
    x = x0.copy().astype(float)
    v = np.zeros(2)
    ema_f = safe(func(*x))
    history = [ema_f]
    grad_norms = []
    lr_history = []
    beta_history = []
    prev_g = None

    for _ in range(max_iter):
        g = grad_func(*x)
        if not np.all(np.isfinite(g)):
            break
        gn = np.linalg.norm(g)
        grad_norms.append(gn)
        if gn < tol:
            break

        # Cosine similarity for adaptive momentum
        if prev_g is not None and np.linalg.norm(prev_g) > 1e-12:
            cos_theta = np.clip(
                np.dot(prev_g, g) / (np.linalg.norm(prev_g) * gn), -1, 1)
        else:
            cos_theta = 1.0
        beta_t = max(beta_min, 0.9 * (0.5 + 0.5*cos_theta))
        beta_history.append(beta_t)

        # Fuzzy learning rate
        lr = fuzzy_lr(gn)
        lr_history.append(lr)

        # EMA backtracking
        f_cur = safe(func(*x))
        ema_f = 0.95*ema_f + 0.05*f_cur
        scale = 1.0 if f_cur <= ema_f else 0.5

        v = beta_t*v - scale*lr*g
        xnew = x + v
        if not np.all(np.isfinite(xnew)):
            break
        x = xnew
        history.append(safe(func(*x)))
        prev_g = g.copy()

    return x, history, grad_norms, lr_history, beta_history


def gdm_nd(func, grad_func, x0, alpha=0.001, beta=0.9, max_iter=500, tol=1e-8):
    x = x0.copy().astype(float)
    v = np.zeros_like(x)
    history = [safe(func(x))]
    for _ in range(max_iter):
        g = grad_func(x)
        if not np.all(np.isfinite(g)): break
        gn = np.linalg.norm(g)
        if gn < tol: break
        v = beta*v - alpha*g; xn = x + v
        if not np.all(np.isfinite(xn)): break
        x = xn; history.append(safe(func(x)))
    return x, history


def flo_nd(func, grad_func, x0, max_iter=500, tol=1e-8):
    x = x0.copy().astype(float)
    history = [safe(func(x))]
    for _ in range(max_iter):
        g = grad_func(x)
        if not np.all(np.isfinite(g)): break
        gn = np.linalg.norm(g)
        if gn < tol: break
        lr = fuzzy_lr(gn); xn = x - lr*g
        if not np.all(np.isfinite(xn)): break
        x = xn; history.append(safe(func(x)))
    return x, history


def hnf_nd(func, grad_func, x0, beta_min=0.3, max_iter=500, tol=1e-8):
    x = x0.copy().astype(float)
    v = np.zeros_like(x)
    ema_f = safe(func(x))
    history = [ema_f]
    prev_g = None
    for _ in range(max_iter):
        g = grad_func(x)
        if not np.all(np.isfinite(g)): break
        gn = np.linalg.norm(g)
        if gn < tol: break
        if prev_g is not None and np.linalg.norm(prev_g) > 1e-12:
            cos_theta = np.clip(np.dot(prev_g, g) / (np.linalg.norm(prev_g)*gn), -1, 1)
        else:
            cos_theta = 1.0
        beta_t = max(beta_min, 0.9*(0.5 + 0.5*cos_theta))
        lr = fuzzy_lr(gn)
        f_cur = safe(func(x)); ema_f = 0.95*ema_f + 0.05*f_cur
        scale = 1.0 if f_cur <= ema_f else 0.5
        v = beta_t*v - scale*lr*g; xn = x + v
        if not np.all(np.isfinite(xn)): break
        x = xn; history.append(safe(func(x))); prev_g = g.copy()
    return x, history


# =============================================================================
# SECTION 4: EXPERIMENTS (30 runs × 7 benchmarks)
# =============================================================================

benchmarks_2d = {
    'Rosenbrock 2D': (rosenbrock, rosenbrock_grad, [-2, 2]),
    'Sphere 2D':     (sphere,     sphere_grad,     [-2, 2]),
    'Himmelblau 2D': (himmelblau, himmelblau_grad, [-2, 2]),
    'Ackley 2D':     (ackley,     ackley_grad,     [-5, 5]),
    'Rastrigin 2D':  (rastrigin,  rastrigin_grad,  [-5, 5]),
    'Beale 2D':      (beale,      beale_grad,      [-2, 2]),
}

N_RUNS = 30
results = {}

print("Running experiments...")
for name, (func, gfunc, domain) in benchmarks_2d.items():
    gdm_vals, flo_vals, hnf_vals = [], [], []
    for i in range(N_RUNS):
        rng = np.random.RandomState(i)
        x0 = rng.uniform(domain[0], domain[1], 2)
        _, hg, _ = gdm(func, gfunc, x0)
        gdm_vals.append(hg[-1])
        _, hf, _, _ = flo(func, gfunc, x0)
        flo_vals.append(hf[-1])
        _, hh, _, _, _ = hnf(func, gfunc, x0)
        hnf_vals.append(hh[-1])
    results[name] = {
        'GD+M': np.array(gdm_vals),
        'FLO':  np.array(flo_vals),
        'HFM':  np.array(hnf_vals),
    }
    print(f"  {name}: done")

# 10D Rosenbrock
gdm_10d, flo_10d, hnf_10d = [], [], []
for i in range(N_RUNS):
    rng = np.random.RandomState(i + 1000)
    x0 = rng.uniform(-2, 2, 10)
    _, h = gdm_nd(rosenbrock_nd, rosenbrock_nd_grad, x0); gdm_10d.append(h[-1])
    _, h = flo_nd(rosenbrock_nd, rosenbrock_nd_grad, x0); flo_10d.append(h[-1])
    _, h = hnf_nd(rosenbrock_nd, rosenbrock_nd_grad, x0); hnf_10d.append(h[-1])
results['Rosenbrock 10D'] = {
    'GD+M': np.array(gdm_10d),
    'FLO':  np.array(flo_10d),
    'HFM':  np.array(hnf_10d),
}
print("  Rosenbrock 10D: done")


# =============================================================================
# SECTION 5: TABLE 3 – FINAL OBJECTIVE VALUES
# =============================================================================

print("\n" + "="*90)
print("TABLE 3: Final objective values: mean ± std across 30 independent runs")
print("Bold (**) = best algorithm for that benchmark")
print("="*90)
print(f"{'Benchmark':<20}  {'GD+M':^28}  {'FLO':^28}  {'HFM':^28}")
print("-"*90)
for name, res in results.items():
    best = min(['GD+M', 'FLO', 'HFM'], key=lambda a: np.nanmean(res[a]))
    row = f"{name:<20}"
    for alg in ['GD+M', 'FLO', 'HFM']:
        m, s = np.nanmean(res[alg]), np.nanstd(res[alg])
        star = ' **' if alg == best else '   '
        row += f"  {m:.2e} ± {s:.1e}{star}"
    print(row)

# =============================================================================
# SECTION 6: TABLE 4 – β_min SENSITIVITY
# =============================================================================

print("\n" + "="*60)
print("TABLE 4: HFM sensitivity to β_min (Rosenbrock 2D, 30 runs)")
print("="*60)
bmin_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
mf_bmin = []
for bm in bmin_vals:
    vals = []
    for i in range(N_RUNS):
        rng = np.random.RandomState(i)
        x0 = rng.uniform(-2, 2, 2)
        _, hh, _, _, _ = hnf(rosenbrock, rosenbrock_grad, x0, beta_min=bm)
        vals.append(hh[-1])
    mf_bmin.append(np.nanmean(vals))
print("  β_min : " + "  ".join(f"{b:.1f}" for b in bmin_vals))
print("  Mean f: " + "  ".join(f"{v:.2f}" for v in mf_bmin))
print(f"  Optimal β_min = {bmin_vals[np.argmin(mf_bmin)]:.1f}")

# =============================================================================
# SECTION 7: WILCOXON SIGNED-RANK TESTS
# =============================================================================

print("\n" + "="*70)
print("WILCOXON SIGNED-RANK TESTS: HFM vs GD+M (α = 0.05, two-tailed)")
print("="*70)
for name, res in results.items():
    a = res['HFM']; b = res['GD+M']
    mask = np.isfinite(a) & np.isfinite(b)
    a2, b2 = a[mask], b[mask]
    if len(a2) < 5:
        print(f"{name:<20}  Insufficient data")
        continue
    try:
        stat, p = wilcoxon(a2, b2)
        n = len(a2)
        r = 1 - 4*stat / (n*(n+1))
        sig = "  * significant" if p < 0.05 else "  (not significant)"
        effect = "large" if abs(r) > 0.5 else ("medium" if abs(r) > 0.3 else "small")
        print(f"{name:<20}  W={stat:6.0f}  p={p:.4f}{sig}  |r|={abs(r):.2f} ({effect} effect)")
    except Exception as e:
        print(f"{name:<20}  Error: {e}")


# =============================================================================
# SECTION 8: FIGURES
# =============================================================================

colors = {'GD+M': '#2196F3', 'FLO': '#FF9800', 'HFM': '#4CAF50'}

# --- FIGURE 1: Conceptual Comparison ---
rng = np.random.RandomState(7)
x0_demo = rng.uniform(-1.5, 1.5, 2)
_, h_gdm, gn_gdm = gdm(rosenbrock, rosenbrock_grad, x0_demo)
_, h_flo, gn_flo, lr_flo = flo(rosenbrock, rosenbrock_grad, x0_demo)
_, h_hnf, gn_hnf, lr_hnf, bt_hnf = hnf(rosenbrock, rosenbrock_grad, x0_demo)

fig1, axes1 = plt.subplots(2, 3, figsize=(16, 10))
fig1.suptitle('Figure 1: Conceptual Comparison – GD+M vs FLO vs HFM\n(Rosenbrock Benchmark)',
              fontsize=13, fontweight='bold')

ax = axes1[0, 0]
ax.semilogy([max(v, 1e-12) for v in h_gdm], color=colors['GD+M'], lw=2, label='GD+M')
ax.semilogy([max(v, 1e-12) for v in h_flo], color=colors['FLO'],  lw=2, label='FLO')
ax.semilogy([max(v, 1e-12) for v in h_hnf], color=colors['HFM'],  lw=2, label='HFM')
ax.set_xlabel('Iteration'); ax.set_ylabel('f(x) [log scale]')
ax.set_title('(a) Objective Convergence'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes1[0, 1]
if gn_gdm: ax.semilogy(gn_gdm, color=colors['GD+M'], lw=2, label='GD+M')
if gn_flo: ax.semilogy(gn_flo, color=colors['FLO'],  lw=2, label='FLO')
if gn_hnf: ax.semilogy(gn_hnf, color=colors['HFM'],  lw=2, label='HFM')
ax.set_xlabel('Iteration'); ax.set_ylabel('‖∇f‖ [log scale]')
ax.set_title('(b) Gradient Norm Decay'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes1[0, 2]
gv = np.linspace(0, 600, 500)
mu_s = [max(min(g/50, (200-g)/150), 0) for g in gv]
mu_m = [max(min((g-50)/150, (500-g)/300), 0) for g in gv]
mu_l = [max(min((g-200)/300, 1), 0) for g in gv]
ax.fill_between(gv, mu_s, alpha=0.2, color='blue')
ax.fill_between(gv, mu_m, alpha=0.2, color='green')
ax.fill_between(gv, mu_l, alpha=0.2, color='red')
ax.plot(gv, mu_s, 'b-', lw=2, label='μ_Small → lr=0.005')
ax.plot(gv, mu_m, 'g-', lw=2, label='μ_Medium → lr=0.002')
ax.plot(gv, mu_l, 'r-', lw=2, label='μ_Large → lr=0.0005')
ax.axvline(50, color='gray', ls='--', alpha=0.6)
ax.axvline(200, color='gray', ls='--', alpha=0.6)
ax.set_xlabel('Gradient Norm ‖∇f‖'); ax.set_ylabel('Membership Degree')
ax.set_title('(c) Fuzzy Membership Functions'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes1[1, 0]
lr_vals = [fuzzy_lr(g) for g in gv]
ax.plot(gv, lr_vals, 'purple', lw=2.5, label='Fuzzy lr* (FLO/HFM)')
ax.fill_between(gv, lr_vals, alpha=0.15, color='purple')
ax.axhline(0.001, color='b', ls='--', lw=2, label='GD+M: fixed α=0.001')
ax.set_xlabel('Gradient Norm ‖∇f‖'); ax.set_ylabel('Learning Rate lr*')
ax.set_title('(d) Defuzzified Learning Rate'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes1[1, 1]
if bt_hnf:
    ax.plot(bt_hnf, color=colors['HFM'], lw=2, label='β_adaptive (HFM)')
ax.axhline(0.9, color='b', ls='--', lw=1.5, label='GD+M: fixed β=0.9')
ax.axhline(0.3, color='r', ls=':', lw=1.5, label='β_min=0.3')
ax.set_ylim(0, 1); ax.set_xlabel('Iteration'); ax.set_ylabel('Momentum β')
ax.set_title('(e) HFM Adaptive Momentum'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes1[1, 2]
data_clean = [np.clip(results['Rosenbrock 2D'][a], 0, 1e5)
              for a in ['GD+M', 'FLO', 'HFM']]
bp = ax.boxplot(data_clean, labels=['GD+M', 'FLO', 'HFM'], patch_artist=True,
                medianprops={'color': 'black', 'lw': 2}, showfliers=True)
for patch, c in zip(bp['boxes'], [colors['GD+M'], colors['FLO'], colors['HFM']]):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax.set_yscale('log')
ax.set_ylabel('Final f(x)'); ax.set_title('(f) Distribution of Final Values\n(Rosenbrock 2D, 30 runs)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Figure1_Conceptual_Comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure 1 saved as Figure1_Conceptual_Comparison.png")

# --- FIGURE 2a: Summary Statistics ---
bench_names = list(benchmarks_2d.keys())
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Figure 2 (Panel A-C): Summary Statistics', fontsize=13, fontweight='bold')

ax = axes[0]
xp = np.arange(len(bench_names)); w = 0.25
mg = np.clip([np.nanmean(results[b]['GD+M']) for b in bench_names], 1e-20, 1e5)
mf = np.clip([np.nanmean(results[b]['FLO'])  for b in bench_names], 1e-20, 1e5)
mh = np.clip([np.nanmean(results[b]['HFM'])  for b in bench_names], 1e-20, 1e5)
ax.bar(xp-w, mg, w, label='GD+M', color=colors['GD+M'], alpha=0.85)
ax.bar(xp,   mf, w, label='FLO',  color=colors['FLO'],  alpha=0.85)
ax.bar(xp+w, mh, w, label='HFM',  color=colors['HFM'],  alpha=0.85)
ax.set_yscale('log'); ax.set_xticks(xp)
ax.set_xticklabels([b[:7] for b in bench_names], rotation=30, fontsize=8)
ax.set_ylabel('Mean f(x) [log]'); ax.set_title('Panel A: Mean Final Objective Values')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
mf_bmin_plot = []
for bm in bmin_vals:
    vals = []
    for i in range(N_RUNS):
        rng = np.random.RandomState(i); x0 = rng.uniform(-2, 2, 2)
        _, hh, _, _, _ = hnf(rosenbrock, rosenbrock_grad, x0, beta_min=bm)
        vals.append(hh[-1])
    mf_bmin_plot.append(np.nanmean(vals))
bar_c = ['#FF7043' if b == 0.3 else '#90CAF9' for b in bmin_vals]
ax.bar([str(b) for b in bmin_vals], mf_bmin_plot, color=bar_c)
for i, v in enumerate(mf_bmin_plot): ax.text(i, v+0.002, f'{v:.2f}', ha='center', fontsize=8)
ax.set_xlabel('β_min'); ax.set_ylabel('Mean f(x)')
ax.set_title('Panel B: β_min Sensitivity (HFM, Rosenbrock 2D)'); ax.grid(True, alpha=0.3, axis='y')

ax = axes[2]
for i in range(5):
    rng = np.random.RandomState(i+200); x0 = rng.uniform(-2, 2, 10)
    _, hg = gdm_nd(rosenbrock_nd, rosenbrock_nd_grad, x0)
    _, hf = flo_nd(rosenbrock_nd, rosenbrock_nd_grad, x0)
    _, hh = hnf_nd(rosenbrock_nd, rosenbrock_nd_grad, x0)
    ax.semilogy([max(v, 1e-6) for v in hg], color=colors['GD+M'], alpha=0.6, lw=1)
    ax.semilogy([max(v, 1e-6) for v in hf], color=colors['FLO'],  alpha=0.6, lw=1)
    ax.semilogy([max(v, 1e-6) for v in hh], color=colors['HFM'],  alpha=0.6, lw=1)
leg = [Line2D([0],[0], color=colors[a], lw=2, label=a) for a in ['GD+M','FLO','HFM']]
ax.legend(handles=leg, fontsize=8); ax.set_xlabel('Iteration'); ax.set_ylabel('f(x) [log]')
ax.set_title('Panel C: Rosenbrock 10D Convergence'); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Figure2a_Summary.png', dpi=150)
plt.close()

# --- FIGURE 2b: Per-Benchmark Convergence ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Figure 2 (Panels D-I): Per-Benchmark Convergence Curves', fontsize=13, fontweight='bold')
for idx, (name, (fn, gf, dom)) in enumerate(list(benchmarks_2d.items())):
    ax = axes[idx//3][idx%3]
    for i in range(6):
        rng = np.random.RandomState(i); x0 = rng.uniform(dom[0], dom[1], 2)
        _, hg, _ = gdm(fn, gf, x0)
        _, hf, _, _ = flo(fn, gf, x0)
        _, hh, _, _, _ = hnf(fn, gf, x0)
        ax.semilogy([max(v, 1e-15) for v in hg], color=colors['GD+M'], alpha=0.6, lw=1.2)
        ax.semilogy([max(v, 1e-15) for v in hf], color=colors['FLO'],  alpha=0.6, lw=1.2)
        ax.semilogy([max(v, 1e-15) for v in hh], color=colors['HFM'],  alpha=0.6, lw=1.2)
    leg = [Line2D([0],[0], color=colors[a], lw=2, label=a) for a in ['GD+M','FLO','HFM']]
    ax.legend(handles=leg, fontsize=8); ax.set_xlabel('Iteration'); ax.set_ylabel('f(x) [log]')
    ax.set_title(f'Panel {chr(68+idx)}: {name}', fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Figure2b_Convergence.png', dpi=150)
plt.close()

# --- FIGURE 2c: Trajectories, Fuzzy LR, Box Plots ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Figure 2 (Panels J-L): Trajectories, Fuzzy LR, Box Plots', fontsize=13, fontweight='bold')

ax = axes[0]
XX, YY = np.meshgrid(np.linspace(-2,2,200), np.linspace(-1,3,200))
ZZ = rosenbrock(XX, YY)
ax.contourf(XX, YY, np.log1p(ZZ), levels=50, cmap='viridis', alpha=0.85)
ax.contour(XX, YY, np.log1p(ZZ), levels=20, colors='white', alpha=0.3, linewidths=0.5)
rng = np.random.RandomState(3); x0t = rng.uniform(-1.5, 1.5, 2)
for alg_name in ['GD+M', 'FLO', 'HFM']:
    traj = [x0t.copy()]; xc = x0t.copy().astype(float); v = np.zeros(2)
    ema_ft = rosenbrock(*xc); pg = None
    for _ in range(300):
        g = rosenbrock_grad(*xc)
        if not np.all(np.isfinite(g)): break
        gn = np.linalg.norm(g)
        if gn < 1e-8: break
        if alg_name == 'GD+M':
            v = 0.9*v - 0.001*g; xn = xc + v
        elif alg_name == 'FLO':
            lr = fuzzy_lr(gn); xn = xc - lr*g; v = np.zeros(2)
        else:
            ct = np.clip(np.dot(pg,g)/(np.linalg.norm(pg)*gn),-1,1) if pg is not None and np.linalg.norm(pg)>1e-12 else 1.0
            bt = max(0.3, 0.9*(0.5+0.5*ct)); lr = fuzzy_lr(gn)
            fc = rosenbrock(*xc); ema_ft = 0.95*ema_ft+0.05*fc; sc = 1.0 if fc <= ema_ft else 0.5
            v = bt*v - sc*lr*g; xn = xc+v; pg = g.copy()
        if not np.all(np.isfinite(xn)): break
        xc = xn; traj.append(xc.copy())
    traj = np.array(traj)
    ax.plot(traj[:,0], traj[:,1], color=colors[alg_name], lw=2, alpha=0.9, label=alg_name)
ax.plot(1, 1, 'r*', ms=15, zorder=10, label='Global min')
ax.set_xlim(-2, 2); ax.set_ylim(-1, 3)
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('Panel J: Rosenbrock 2D Trajectories'); ax.legend(fontsize=8)

ax = axes[1]
gvals = np.linspace(0, 600, 500); lrv = [fuzzy_lr(g) for g in gvals]
ax.plot(gvals, lrv, 'purple', lw=2.5, label='Fuzzy lr*')
ax.fill_between(gvals, lrv, alpha=0.15, color='purple')
ax.axhline(0.001, color='b', ls='--', lw=2, label='GD+M α=0.001')
ax.axvline(50, color='gray', ls='--', lw=1, alpha=0.6)
ax.axvline(200, color='gray', ls='--', lw=1, alpha=0.6)
ax.set_xlabel('Gradient Norm ‖∇f‖'); ax.set_ylabel('Learning Rate lr*')
ax.set_title('Panel K: Defuzzified Learning Rate (FLO/HFM)'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[2]
all_bnames = list(results.keys())
positions = []; all_data_bp = []
for i, bn in enumerate(all_bnames):
    for k, alg in enumerate(['GD+M', 'FLO', 'HFM']):
        d = results[bn][alg]; d = d[np.isfinite(d)]
        d_log = np.log10(np.clip(d, 1e-20, 1e10))
        all_data_bp.append(d_log); positions.append(i*2 + k*0.55)
bp = ax.boxplot(all_data_bp, positions=positions, widths=0.45, patch_artist=True,
                medianprops={'color': 'black', 'lw': 1.5}, showfliers=False)
clr_cycle = [colors['GD+M'], colors['FLO'], colors['HFM']] * len(all_bnames)
for patch, c in zip(bp['boxes'], clr_cycle): patch.set_facecolor(c); patch.set_alpha(0.65)
group_centers = [i*2 + 0.55 for i in range(len(all_bnames))]
ax.set_xticks(group_centers)
ax.set_xticklabels([b[:9] for b in all_bnames], rotation=35, fontsize=7)
leg = [Line2D([0],[0], color=colors[a], lw=5, alpha=0.65, label=a) for a in ['GD+M','FLO','HFM']]
ax.legend(handles=leg, fontsize=8); ax.set_ylabel('log₁₀(Final f(x))')
ax.set_title('Panel L: Box Plots – All Benchmarks'); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('Figure2c_ExtraPlots.png', dpi=150)
plt.close()

# --- FIGURE 3: 3D Surface Plots ---
fig3 = plt.figure(figsize=(18, 10))
fig3.suptitle('Figure 3: Benchmark Function 3D Surface Plots', fontsize=14, fontweight='bold')
surf_info = [
    ('Rosenbrock', rosenbrock, (-2,2), (-1,3), 'plasma'),
    ('Sphere',     sphere,     (-3,3), (-3,3), 'viridis'),
    ('Himmelblau', himmelblau, (-5,5), (-5,5), 'inferno'),
    ('Ackley',     ackley,     (-4,4), (-4,4), 'coolwarm'),
    ('Rastrigin',  rastrigin,  (-5,5), (-5,5), 'RdYlGn_r'),
    ('Beale',      beale,      (-4,4.5), (-1,1), 'ocean'),
]
for k, (fname, ffunc, xr, yr, cmap) in enumerate(surf_info):
    ax3 = fig3.add_subplot(2, 3, k+1, projection='3d')
    X = np.linspace(xr[0], xr[1], 60); Y = np.linspace(yr[0], yr[1], 60)
    X, Y = np.meshgrid(X, Y); Z = ffunc(X, Y)
    p95 = np.percentile(Z[np.isfinite(Z)], 95); Z = np.clip(Z, 0, p95)
    ax3.plot_surface(X, Y, Z, cmap=cmap, alpha=0.85, linewidth=0, antialiased=True)
    ax3.set_title(fname, fontsize=10, fontweight='bold')
    ax3.set_xlabel('x', fontsize=8); ax3.set_ylabel('y', fontsize=8); ax3.set_zlabel('f', fontsize=8)
    ax3.tick_params(labelsize=6)
plt.tight_layout()
plt.savefig('Figure3_Surface_Plots.png', dpi=150)
plt.close()

print("\n" + "="*60)
print("All figures saved:")
print("  Figure1_Conceptual_Comparison.png")
print("  Figure2a_Summary.png")
print("  Figure2b_Convergence.png")
print("  Figure2c_ExtraPlots.png")
print("  Figure3_Surface_Plots.png")
print("="*60)
