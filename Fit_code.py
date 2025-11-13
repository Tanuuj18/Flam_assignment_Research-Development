import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, least_squares
import matplotlib.pyplot as plt
import os, math, warnings
warnings.filterwarnings('ignore')

# ---------- Config ----------
csv_path = '/content/xy_data.csv' 
output_dir = '/mnt/data/output_alternating'
os.makedirs(output_dir, exist_ok=True)

# Alternating algorithm parameters
n_outer_iters = 8          # outer alternating iterations
local_window = 6.0         # search window half-width around previous t (seconds)
minimize_opts = {'xatol':1e-4, 'maxiter':60}
theta_starts_deg = [10, 20, 28, 35, 45]   # multi-start initial theta guesses (degrees)
# Parameter bounds from assignment 
theta_min = np.deg2rad(0.0001)
theta_max = np.deg2rad(50.0 - 1e-6)
M_min, M_max = -0.05, 0.05
X_min, X_max = 0.0, 100.0

# ---------- Load data ----------
df = pd.read_csv(csv_path)
if not set(['x','y']).issubset(df.columns):
    raise RuntimeError("CSV must contain columns 'x' and 'y'")
x_obs = df['x'].values
y_obs = df['y'].values
n = len(x_obs)
# initial uniform t
t_init = np.linspace(6.0, 60.0, n)

# ---------- Model ----------
def model_xy(t, theta, M, X):
    term = np.exp(np.abs(t) * M) * np.sin(0.3 * t)
    x = t * np.cos(theta) - term * np.sin(theta) + X
    y = 42 + t * np.sin(theta) + term * np.cos(theta)
    return x, y

# residuals for least_squares (given t_vals)
def residuals_params(params, t_vals, x_target, y_target):
    theta, M, X = params
    xm, ym = model_xy(t_vals, theta, M, X)
    return np.concatenate([xm - x_target, ym - y_target])

# objective (sum of squared residuals) wrapper
def ssq_for_params(params, t_vals, x_target, y_target):
    r = residuals_params(params, t_vals, x_target, y_target)
    return np.sum(r**2)

# helper: find best t for a single observed point (bounded scalar minimization)
def find_best_t_for_point(xo, yo, theta, M, X, t_center):
    lo = max(6.0, t_center - local_window)
    hi = min(60.0, t_center + local_window)
    def d2(t):
        xm, ym = model_xy(np.array([t]), theta, M, X)
        return (xm[0] - xo)**2 + (ym[0] - yo)**2
    res = minimize_scalar(d2, bounds=(lo, hi), method='bounded', options=minimize_opts)
    return float(res.x)

# ---------- Alternating routine for one start ----------
def alternating_fit(theta0_rad, M0, X0, t_initial, n_outer=n_outer_iters):
    params = np.array([theta0_rad, M0, X0], dtype=float)
    t_vals = t_initial.copy()
    for outer in range(n_outer):
        # 1) update each t_i by projecting point to curve (local search)
        for i, (xo, yo) in enumerate(zip(x_obs, y_obs)):
            t_center = t_vals[i]
            t_vals[i] = find_best_t_for_point(xo, yo, params[0], params[1], params[2], t_center)
        # 2) enforce monotonic non-decreasing t (small numeric bumps to avoid equality)
        for i in range(1, n):
            if t_vals[i] < t_vals[i-1]:
                t_vals[i] = t_vals[i-1] + 1e-6
        # clip to domain
        t_vals = np.clip(t_vals, 6.0, 60.0)
        # 3) fit params (theta, M, X) with bounded least squares (robust loss)
        lower = [theta_min, M_min, X_min]
        upper = [theta_max, M_max, X_max]
        res_ls = least_squares(lambda p: residuals_params(p, t_vals, x_obs, y_obs),
                               params, bounds=(lower, upper), loss='soft_l1',
                               ftol=1e-9, xtol=1e-9, gtol=1e-9, max_nfev=2000)
        params = res_ls.x
        # optional: print progress
        xm, ym = model_xy(t_vals, params[0], params[1], params[2])
        L1_mean = np.mean(np.abs(xm - x_obs) + np.abs(ym - y_obs))
        RMSE = np.sqrt(np.mean((xm - x_obs)**2 + (ym - y_obs)**2))
        print(f"  outer {outer+1}: theta_deg={np.rad2deg(params[0]):.6f} M={params[1]:.8f} X={params[2]:.6f}  L1_mean={L1_mean:.4f} RMSE={RMSE:.4f}")
    # final metrics
    xm, ym = model_xy(t_vals, params[0], params[1], params[2])
    L1_total = np.sum(np.abs(xm - x_obs) + np.abs(ym - y_obs))
    L1_mean = np.mean(np.abs(xm - x_obs) + np.abs(ym - y_obs))
    RMSE = np.sqrt(np.mean((xm - x_obs)**2 + (ym - y_obs)**2))
    return params, t_vals, L1_total, L1_mean, RMSE

# ---------- Multi-starts ----------
best_result = None
for th_deg in theta_starts_deg:
    print(f"\nStarting alternating with initial theta = {th_deg}°")
    theta0 = np.deg2rad(th_deg)
    M0 = 0.0
    X0 = float(np.mean(x_obs))
    params_out, t_out, L1_tot, L1_mean, RMSE_val = alternating_fit(theta0, M0, X0, t_init, n_outer=n_outer_iters)
    print(f"--> Result: theta_deg={np.rad2deg(params_out[0]):.6f}, M={params_out[1]:.8f}, X={params_out[2]:.6f}, L1_total={L1_tot:.6f}, RMSE={RMSE_val:.6f}")
    if best_result is None or L1_tot < best_result[0]:
        best_result = (L1_tot, params_out.copy(), t_out.copy(), L1_mean, RMSE_val)

# ---------- Best overall ----------
L1_best, params_best, t_best, L1_mean_best, RMSE_best = best_result
theta_fit, M_fit, X_fit = params_best
print("\nBEST IN-BOUNDS RESULT (multi-start):")
print("theta (deg):", np.rad2deg(theta_fit))
print("M:", M_fit)
print("X:", X_fit)
print("L1 total:", L1_best, "L1 mean:", L1_mean_best, "RMSE:", RMSE_best)

# ---------- Save final outputs ----------
# 1) Save plots
x_fit, y_fit = model_xy(t_best, theta_fit, M_fit, X_fit)
plt.figure(figsize=(8,6))
plt.scatter(x_obs, y_obs, s=6, label='observed')
plt.plot(x_fit, y_fit, '-', lw=1.5, label='fitted (alternating)')
plt.legend(); plt.xlabel('x'); plt.ylabel('y'); plt.title('Observed vs fitted (alternating)'); plt.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fit_alternating.png'), dpi=150)

# residuals vs t
res_x = x_fit - x_obs
res_y = y_fit - y_obs
res_norm = np.sqrt(res_x**2 + res_y**2)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(t_best, res_x, '.-'); plt.title('residual x vs t'); plt.xlabel('t'); plt.grid(True)
plt.subplot(1,2,2)
plt.plot(t_best, res_y, '.-'); plt.title('residual y vs t'); plt.xlabel('t'); plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'residuals_vs_t.png'), dpi=150)

plt.figure(figsize=(6,4))
plt.hist(res_norm, bins=40); plt.title('hist residual magnitudes'); plt.xlabel('res mag'); plt.ylabel('count')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'residuals_hist.png'), dpi=150)

# 2) Save parameters file and desmos string
with open(os.path.join(output_dir, 'final_parameters.txt'), 'w') as f:
    f.write(f"theta_deg = {np.rad2deg(theta_fit)}\n")
    f.write(f"M = {M_fit}\n")
    f.write(f"X = {X_fit}\n")
    f.write(f"L1_total = {L1_best}\nL1_mean = {L1_mean_best}\nRMSE = {RMSE_best}\n")

desmos_x = f"t*cos({np.rad2deg(theta_fit):.12f}\\deg) - e^({M_fit:.12f}*abs(t))*sin(0.3t)*sin({np.rad2deg(theta_fit):.12f}\\deg) + {X_fit:.12f}"
desmos_y = f"42 + t*sin({np.rad2deg(theta_fit):.12f}\\deg) + e^({M_fit:.12f}*abs(t))*sin(0.3t)*cos({np.rad2deg(theta_fit):.12f}\\deg)"
with open(os.path.join(output_dir, 'desmos_equation.txt'), 'w') as f:
    f.write("x(t) = " + desmos_x + "\n")
    f.write("y(t) = " + desmos_y + "\n")
    f.write("Domain: 6 <= t <= 60\n")

print("\nSaved plots and final_parameters.txt and desmos_equation.txt to:", output_dir)
print("\nDESMOS parametric (copy-paste):")
print("x(t) =", desmos_x)
print("y(t) =", desmos_y)
print("\nDone.")
