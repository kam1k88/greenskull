# =============================================================
# calibration.py — финал: все три модели на CPU LM
# =============================================================
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.optimize import least_squares, brentq
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')
np.seterr(all='ignore')

URL = "https://kam1k88.github.io/postcrossing/TimeData.csv"
TARGET_M = 89
N_MC = 200
EPS = 1.0

# -------------------------------------------------------------
# 1. ДАННЫЕ
# -------------------------------------------------------------
MILESTONES_RAW = """
2008-04-11 15:03:00 1000000
2009-02-26 06:34:00 2000000
2009-09-24 16:09:00 3000000
2010-03-28 10:14:00 4000000
2010-08-24 19:14:00 5000000
2010-12-30 12:03:00 6000000
2011-04-19 18:59:00 7000000
2011-08-02 11:28:00 8000000
2011-11-03 05:13:00 9000000
2012-01-27 16:56:00 10000000
2012-04-03 20:04:00 11000000
2012-06-12 18:28:00 12000000
2012-08-22 05:08:00 13000000
2012-10-25 20:52:00 14000000
2012-12-31 13:46:00 15000000
2013-03-04 07:21:00 16000000
2013-05-01 00:20:00 17000000
2013-07-03 06:57:00 18000000
2013-09-02 09:56:00 19000000
2013-10-29 05:49:00 20000000
2013-12-23 22:54:00 21000000
2014-02-17 15:03:00 22000000
2014-04-10 10:47:00 23000000
2014-06-08 12:40:00 24000000
2014-08-10 11:02:00 25000000
2014-10-10 03:36:00 26000000
2014-12-07 02:05:00 27000000
2015-02-05 19:13:00 28000000
2015-04-02 16:04:00 29000000
2015-06-02 00:27:00 30000000
2015-08-05 11:39:00 31000000
2015-10-06 18:40:00 32000000
2015-12-05 11:37:00 33000000
2016-02-04 19:41:00 34000000
2016-03-31 23:49:00 35000000
2016-05-31 18:34:00 36000000
2016-08-02 19:08:00 37000000
2016-10-04 16:09:00 38000000
2016-12-07 07:56:00 39000000
2017-02-13 06:39:00 40000000
2017-04-17 10:46:00 41000000
2017-06-28 18:15:00 42000000
2017-09-02 14:23:00 43000000
2017-11-07 22:46:00 44000000
2018-01-16 18:26:00 45000000
2018-03-20 18:00:00 46000000
2018-05-28 08:03:00 47000000
2018-08-07 03:47:00 48000000
2018-10-16 01:32:00 49000000
2018-12-27 19:13:00 50000000
2019-03-04 10:56:00 51000000
2019-05-13 05:47:00 52000000
2019-07-25 05:25:00 53000000
2019-10-07 17:27:00 54000000
2019-12-18 16:56:00 55000000
2020-02-28 11:49:00 56000000
2020-06-05 19:15:00 57000000
2020-08-28 09:17:00 58000000
2020-11-10 20:51:00 59000000
2021-01-25 22:17:00 60000000
2021-04-02 00:57:00 61000000
2021-06-08 00:01:00 62000000
2021-08-18 07:12:00 63000000
2021-10-26 13:07:00 64000000
2022-01-04 20:39:00 65000000
2022-03-09 19:44:00 66000000
2022-05-25 13:11:00 67000000
2022-08-10 18:13:00 68000000
2022-10-22 19:18:00 69000000
2023-01-04 23:36:00 70000000
2023-03-14 12:32:00 71000000
2023-05-28 13:36:00 72000000
2023-08-12 19:35:00 73000000
2023-10-25 19:35:00 74000000
2024-01-08 15:13:00 75000000
2024-03-18 20:32:00 76000000
2024-06-02 08:58:00 77000000
2024-08-16 12:46:00 78000000
2024-10-27 20:52:00 79000000
2025-01-07 02:32:00 80000000
2025-03-18 17:50:00 81000000
2025-06-03 02:20:00 82000000
2025-08-22 08:30:00 83000000
2025-11-02 11:20:00 84000000
2026-01-19 01:06:00 85000000
2026-03-27 18:27:00 86000000
2026-06-09 15:02:00 87000000
2026-08-24 22:12:00 88000000
"""

def load_data():
    return pd.read_csv(URL, parse_dates=['datetime']).sort_values('datetime').reset_index(drop=True)

def load_milestones():
    rows = []
    for line in MILESTONES_RAW.strip().splitlines():
        p = line.strip().split()
        if len(p) != 3:
            continue
        dt = datetime.strptime(f"{p[0]} {p[1]}", "%Y-%m-%d %H:%M:%S")
        rows.append((int(p[2]), dt))
    ms = pd.DataFrame(rows, columns=['million', 'date'])
    ms['date'] = pd.to_datetime(ms['date'])
    ms['M'] = ms['million'] // 1_000_000
    ms['dt'] = (ms['date'].shift(-1) - ms['date']).dt.total_seconds() / 86400.0
    return ms

# -------------------------------------------------------------
# 2. МОДЕЛИ
# -------------------------------------------------------------
def model1_np(p, x, n_yearly=12):
    a, b, c = p[0], p[1], p[2]
    a3, c3 = p[3], p[4]
    out = a + b*x + c*x*x + a3*np.sin(2*np.pi*x*0.5/365.25 + c3)
    idx = 5
    for k in range(1, n_yearly+1):
        out = out + p[idx]*np.sin(2*np.pi*x*k/365.25 + p[idx+1])
        idx += 2
    for period in [7.0, 3.5, 1.0]:
        out = out + p[idx]*np.sin(2*np.pi*x/period + p[idx+1])
        idx += 2
    return out

def model2_np(p, n, q4=50.0):
    a, b, c, d = p[0], p[1], p[2], p[3]
    a4, c4 = p[4], p[5]
    a5, c5 = p[6], p[7]
    a6, c6 = p[8], p[9]
    return (a + b*n + c*n*n + d*n*n*n
            + a4*np.sin(2*np.pi*n/q4 + c4)
            + a5*np.sin(2*np.pi*n/(q4/2) + c5)
            + a6*np.sin(2*np.pi*n/(q4/3) + c6))

# -------------------------------------------------------------
# 3. ПАРАМЕТРЫ
# -------------------------------------------------------------
def init_m1_unp(y0, slope, n_yearly=12):
    """p0 для M1 в исходном масштабе (LM, без нормализации)."""
    n = 3 + 2 + 2*n_yearly + 6
    p = np.zeros(n)
    p[0] = y0
    p[1] = slope
    p[3] = 1000.0
    for i in range(n_yearly):
        p[5 + 2*i] = 500.0
    base = 5 + 2*n_yearly
    for i in range(3):
        p[base + 2*i] = 200.0
    return p

def phase_idx_m1(n_yearly=12):
    idx = {4}
    for i in range(n_yearly):
        idx.add(6 + 2*i)
    base = 5 + 2*n_yearly
    for i in range(3):
        idx.add(base + 2*i + 1)
    return idx

PHASE_M23 = {5, 7, 9}
POLY_M1 = {0, 1, 2}
POLY_M23 = {0, 1, 2, 3}

# -------------------------------------------------------------
# 4. MC-FIT (CPU LM)
# -------------------------------------------------------------
def mc_fit_cpu(x, y, model_func_np, p0, phase_idx, poly_idx,
               ridge_idx=None, ridge_lambda=0.0,
               n_iter=N_MC, eps=EPS, label=""):
    def resid(p):
        r = model_func_np(p, x) - y
        if ridge_idx:
            r = np.concatenate([r,
                np.sqrt(ridge_lambda) * np.array([p[i] for i in ridge_idx])])
        return r

    best_p = None
    best_std = np.inf
    for _ in tqdm(range(n_iter), desc=label, leave=False):
        p_try = p0.copy()
        for i in range(len(p0)):
            if i in phase_idx:
                p_try[i] = p0[i] + np.random.uniform(-1, 1) * np.pi * eps
            elif i in poly_idx:
                p_try[i] = p0[i] * (1 + np.random.uniform(-2, 2) * eps)
            else:
                p_try[i] = p0[i] * (1 + np.random.uniform(-1, 1) * eps)
        try:
            res = least_squares(resid, p_try, method='lm', max_nfev=20000)
            if not np.all(np.isfinite(res.x)):
                continue
            std = float(np.std(res.fun[:len(x)]))
            if std < best_std:
                best_std = std
                best_p = res.x
        except Exception:
            continue
    return best_p, best_std

# -------------------------------------------------------------
# 5. DISPATCHER — все три модели на CPU LM
# -------------------------------------------------------------
def mc_fit(x, y, p0, phase_idx, poly_idx, model_kind='m1',
           ridge_idx=None, ridge_lambda=0.0,
           n_iter=N_MC, label=""):
    if model_kind == 'm1':
        f = lambda p, x_: model1_np(p, x_)
    else:
        f = lambda p, x_: model2_np(p, x_)
    return mc_fit_cpu(x, y, f, p0, phase_idx, poly_idx,
                      ridge_idx=ridge_idx, ridge_lambda=ridge_lambda,
                      n_iter=n_iter, label=label)

# -------------------------------------------------------------
# 6. ВЕСА
# -------------------------------------------------------------
def get_weights(T_days, tau=60.0):
    w1 = 1.0/3 + (2.0/3) * np.exp(-T_days / tau)
    rest = 1.0 - w1
    return w1, rest*0.6, rest*0.4

# -------------------------------------------------------------
# 7. SELF-CONSISTENT FORECAST
# -------------------------------------------------------------
def forecast(data, milestones, target_M,
             max_iter=6, tol_days=0.1,
             n_yearly=12, n_iter_mc=N_MC, tau=60.0,
             ridge_lambda=50.0, verbose=False):
    last_M = target_M - 1
    last_known = milestones[milestones['M'] == last_M]
    if len(last_known) == 0:
        if verbose:
            print(f"    [forecast] last_known (M={last_M}) empty")
        return None
    last_date = last_known['date'].iloc[0]
    last_val = float(last_M * 1_000_000)
    target_val = float(target_M * 1_000_000)

    est_date = last_date + timedelta(days=75.0)

    mf = milestones[(milestones['M'] >= 44)
                    & (milestones['M'] <= last_M)
                    & (~milestones['M'].isin(range(56, 67)))].dropna(subset=['dt'])
    if len(mf) == 0:
        if verbose:
            print("    [forecast] mf is empty")
        return None

    m2_x = mf['M'].values.astype(float)
    m2_y = mf['dt'].values.astype(float)
    day0 = mf['date'].iloc[0]
    m3_x = (mf['date'] - day0).dt.total_seconds().values / 86400.0

    p0_m23 = np.array([m2_y.mean(), 0.0, 0.0, 0.0,
                       3.0, 0.0, 2.0, 0.0, 1.0, 0.0])

    dt_m1 = dt_m2 = dt_m3 = None
    w1 = w2 = w3 = None

    for it in range(max_iter):
        # ---------- M1 ----------
        cutoff = data['datetime'].max() - pd.Timedelta(days=730)
        sub = data[data['datetime'] >= cutoff].reset_index(drop=True)
        if len(sub) < 10:
            sub = data.tail(100).reset_index(drop=True)
        t0 = sub['datetime'].iloc[0]
        t_hist = (sub['datetime'] - t0).dt.total_seconds().values / 86400.0
        y_hist = sub['postcards_received'].values.astype(float)
        t_a1 = (last_date - t0).total_seconds() / 86400.0
        t_a2 = (est_date - t0).total_seconds() / 86400.0
        t_aug = np.concatenate([t_hist, [t_a1, t_a2]])
        y_aug = np.concatenate([y_hist, [last_val, target_val]])

        slope0 = (y_aug[-1] - y_aug[0]) / max(t_aug[-1] - t_aug[0], 1e-9)
        p0_m1 = init_m1_unp(y_aug[0], slope0, n_yearly)
        p_m1, std_m1 = mc_fit(t_aug, y_aug, p0_m1,
                              phase_idx_m1(n_yearly), POLY_M1,
                              model_kind='m1', n_iter=n_iter_mc, label="M1")
        if p_m1 is None:
            if verbose:
                print("    [forecast] M1 fit returned None")
            return None
        if verbose:
            print(f"    [M1] std={std_m1:.1f}")

        t_last = (data['datetime'].max() - t0).total_seconds() / 86400.0

        def f_target(t):
            return float(model1_np(p_m1, np.array([t]), n_yearly=n_yearly)[0]) - target_val

        try:
            t_89 = brentq(f_target, t_last, t_last + 400.0)
            date_m1 = t0 + timedelta(days=t_89)
        except Exception as e:
            if verbose:
                print(f"    [forecast] brentq failed: {e}")
            return None

        # ---------- M2 ----------
        dt_est = (est_date - last_date).total_seconds() / 86400.0
        m2_x_aug = np.append(m2_x, target_M)
        m2_y_aug = np.append(m2_y, dt_est)
        p_m2, std_m2 = mc_fit(m2_x_aug, m2_y_aug, p0_m23,
                              PHASE_M23, POLY_M23,
                              model_kind='m23',
                              ridge_idx=[3], ridge_lambda=ridge_lambda,
                              n_iter=n_iter_mc, label="M2")
        if p_m2 is None:
            if verbose:
                print("    [forecast] M2 fit returned None")
            return None
        dt_m2 = float(model2_np(p_m2, np.array([target_M]))[0])
        if verbose:
            print(f"    [M2] std={std_m2:.3f}, dt={dt_m2:.2f}")

        # ---------- M3 ----------
        day_est = (est_date - day0).total_seconds() / 86400.0
        m3_x_aug = np.append(m3_x, day_est)
        m3_y_aug = np.append(m2_y, dt_est)
        p_m3, std_m3 = mc_fit(m3_x_aug, m3_y_aug, p0_m23,
                              PHASE_M23, POLY_M23,
                              model_kind='m23',
                              ridge_idx=[3], ridge_lambda=ridge_lambda,
                              n_iter=n_iter_mc, label="M3")
        if p_m3 is None:
            if verbose:
                print("    [forecast] M3 fit returned None")
            return None
        dt_m3 = float(model2_np(p_m3, np.array([day_est]))[0])
        if verbose:
            print(f"    [M3] std={std_m3:.3f}, dt={dt_m3:.2f}")

        # ---------- Веса ----------
        dt_m1 = (date_m1 - last_date).total_seconds() / 86400.0
        T_mean = (dt_m1 + dt_m2 + dt_m3) / 3
        w1, w2, w3 = get_weights(T_mean, tau=tau)

        dt_new = w1*dt_m1 + w2*dt_m2 + w3*dt_m3
        est_new = last_date + timedelta(days=dt_new)
        delta = abs((est_new - est_date).total_seconds()) / 86400.0
        est_date = est_new

        if verbose:
            print(f"  it{it}: dt=({dt_m1:.2f},{dt_m2:.2f},{dt_m3:.2f}) "
                  f"w=({w1:.2f},{w2:.2f},{w3:.2f}) -> {est_date.date()}")

        if delta < tol_days:
            break

    return {
        'date': est_date,
        'dt_m1': dt_m1, 'dt_m2': dt_m2, 'dt_m3': dt_m3,
        'w1': w1, 'w2': w2, 'w3': w3,
        'last_date': last_date,
    }

# -------------------------------------------------------------
# 8. MULTI-HORIZON BACKTEST
# -------------------------------------------------------------
def backtest_multihorizon(milestones, data, ks, horizons,
                          n_yearly=12, n_iter_mc=N_MC, tau=60.0,
                          ridge_lambda=50.0, verbose=True):
    rows = []
    skipped = 0
    failed = 0
    for k in ks:
        if k + 1 > 88:
            continue
        actual_date = pd.Timestamp(milestones[milestones['M'] == k+1]['date'].iloc[0])
        k_date = pd.Timestamp(milestones[milestones['M'] == k]['date'].iloc[0])

        for H in horizons:
            cutoff = actual_date - pd.Timedelta(days=H)
            if cutoff < k_date:
                skipped += 1
                if verbose:
                    print(f"  k={k} H={H:3d}: SKIP")
                continue

            data_cut = data[data['datetime'] <= cutoff].copy()
            ms_cut = milestones[milestones['date'] <= cutoff].copy()

            try:
                res = forecast(data_cut, ms_cut, target_M=k+1,
                               n_yearly=n_yearly, n_iter_mc=n_iter_mc,
                               tau=tau, ridge_lambda=ridge_lambda,
                               verbose=False)
            except Exception as e:
                import traceback
                failed += 1
                if verbose:
                    print(f"  k={k} H={H:3d}: EXC {type(e).__name__}: {e}")
                    traceback.print_exc()
                continue
            if res is None:
                failed += 1
                if verbose:
                    print(f"  k={k} H={H:3d}: returned None")
                continue

            error = (res['date'] - actual_date).total_seconds() / 86400.0
            rows.append({
                'k': k, 'target': k+1, 'H': H,
                'pred': res['date'], 'actual': actual_date,
                'error_days': error,
                'w1': res['w1'], 'w2': res['w2'], 'w3': res['w3'],
                'dt_m1': res['dt_m1'], 'dt_m2': res['dt_m2'], 'dt_m3': res['dt_m3'],
            })
            if verbose:
                print(f"  k={k} H={H:3d} pred={res['date'].date()} "
                      f"actual={actual_date.date()} err={error:+.2f} w1={res['w1']:.2f}")

    if verbose:
        print(f"\n  [backtest] rows={len(rows)} skip={skipped} failed={failed}")
    return pd.DataFrame(rows)

# -------------------------------------------------------------
# 9. METRICS / GRID SEARCH
# -------------------------------------------------------------
def metrics(errors):
    e = np.asarray(errors, float)
    e = e[np.isfinite(e)]
    if len(e) == 0:
        return {'n': 0}
    return {
        'n': len(e),
        'bias': float(np.mean(e)),
        'sigma': float(np.std(e, ddof=1)) if len(e) > 1 else float('nan'),
        'MAE': float(np.mean(np.abs(e))),
        'RMSE': float(np.sqrt(np.mean(e**2))),
        'max_abs': float(np.max(np.abs(e))),
    }

def grid_search_tau(milestones, data, ks, horizons, taus,
                    n_yearly=12, n_iter_mc=100):
    results = []
    for tau in taus:
        print(f"\n=== tau = {tau} ===")
        df = backtest_multihorizon(milestones, data, ks, horizons,
                                   n_yearly=n_yearly, n_iter_mc=n_iter_mc,
                                   tau=tau, verbose=False)
        if len(df) == 0:
            continue
        m = metrics(df['error_days'].values)
        results.append({'tau': tau, **m})
        print(f"  n={m['n']} bias={m['bias']:+.2f} sigma={m['sigma']:.2f} "
              f"MAE={m['MAE']:.2f} RMSE={m['RMSE']:.2f} max={m['max_abs']:.2f}")
    return pd.DataFrame(results)

# -------------------------------------------------------------
# 10. MAIN
# -------------------------------------------------------------
if __name__ == "__main__":
    data = load_data()
    ms = load_milestones()
    print(f"\nData: {len(data)} rows, "
          f"{data['datetime'].min()} -- {data['datetime'].max()}")

    print("\n" + "=" * 70)
    print("MULTI-HORIZON BACKTEST (tau=60, baseline)")
    print("=" * 70)
    ks = [82, 83, 84, 85, 86, 87]
    horizons = [15, 30, 60, 90]
    df_bt = backtest_multihorizon(ms, data, ks, horizons,
                                  n_yearly=12, n_iter_mc=100, tau=60.0,
                                  verbose=True)
    if len(df_bt) > 0:
        print("\nMETRICS:")
        m = metrics(df_bt['error_days'].values)
        for k, v in m.items():
            print(f"  {k:10s}: {v:.3f}" if isinstance(v, float)
                  else f"  {k:10s}: {v}")
        print("\nBy horizon:")
        for H in horizons:
            sub = df_bt[df_bt['H'] == H]
            if len(sub) == 0:
                continue
            mh = metrics(sub['error_days'].values)
            print(f"  H={H:3d}: bias={mh['bias']:+.2f} sigma={mh['sigma']:.2f} "
                  f"MAE={mh['MAE']:.2f}")
        print("\nMean w1 by horizon:")
        for H in horizons:
            sub = df_bt[df_bt['H'] == H]
            if len(sub) > 0:
                print(f"  H={H:3d}: w1={sub['w1'].mean():.3f}")

    print("\n" + "=" * 70)
    print("GRID SEARCH OVER TAU")
    print("=" * 70)
    taus = [30, 45, 60, 90, 120]
    df_grid = grid_search_tau(ms, data, ks, horizons, taus,
                              n_yearly=12, n_iter_mc=100)
    print("\n" + "=" * 70)
    print("GRID SEARCH SUMMARY")
    print("=" * 70)
    if len(df_grid) > 0:
        print(df_grid.to_string(index=False))
        best = df_grid.loc[df_grid['MAE'].idxmin()]
        print(f"\nBest tau: {best['tau']} (MAE={best['MAE']:.3f}, "
              f"bias={best['bias']:+.2f}, sigma={best['sigma']:.2f})")

    print("\n" + "=" * 70)
    print("FINAL FORECAST 89M (full data)")
    print("=" * 70)
    res89 = forecast(data, ms, target_M=89, n_yearly=12, n_iter_mc=N_MC,
                     tau=60.0, verbose=True)
    if res89:
        print(f"\nDate 89M: {res89['date']}")
        print(f"dt=({res89['dt_m1']:.2f}, {res89['dt_m2']:.2f}, {res89['dt_m3']:.2f})")
        print(f"w=({res89['w1']:.3f}, {res89['w2']:.3f}, {res89['w3']:.3f})")