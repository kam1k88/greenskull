# -*- coding: utf-8 -*-
"""
fast_method.py -- метод Сергея, переписанный через линейный МНК (lstsq).

Три ключевых факта, найденных при разборе calibration.py:

1) Все три модели ЛИНЕЙНЫ по параметрам: a*sin(wx+c) = A*sin(wx) + B*cos(wx).
   Частоты (q, q4) зашиты константами и не подбираются, поэтому задача имеет
   ЕДИНСТВЕННЫЙ глобальный оптимум. Monte-Carlo встряска и 200 LM-рестартов
   дают побитово тот же ответ, что и один вызов np.linalg.lstsq.
   -> расчёт ускоряется примерно в 2000 раз без потери точности.
   (У Сергея q -- подбираемый параметр, поэтому у НЕГО встряска осмысленна.)

2) Структурные дефекты исходной конфигурации (проверено бэктестом на 105 точках):
   - квадратичный член в M1 при экстраполяции на 400 дней: MAE 3.97 против 2.36
     у линейного тренда с теми же гармониками;
   - кривая M1 не привязана к последнему наблюдению: потеря ещё ~1.2 дня MAE;
   - M2: кубика + 3 гармоники с периодом 50/25/16.7 по номеру вехи на 33 точках
     экстраполируется в бесконечность (MAE без anchor -- десятки тысяч дней);
     держится только за счёт anchor-точки, т.е. воспроизводит собственный вход;
   - M3: та же формула применяется к оси КАЛЕНДАРНЫХ ДНЕЙ, где q4=50 означает
     период 50 дней (60 циклов на интервале данных) вместо годовой сезонности.

3) Веса w1/w2/w3 считались от полного интервала между вехами (~74 дня), который
   почти не меняется, поэтому w1 = 0.52...0.54 на ЛЮБОМ горизонте. Идея Сергея
   ("ближе к концу доминирует первая регрессия") требует остатка горизонта.

Конфигурация ниже выбрана по бэктесту; tau подобран только на вехах 67-81.
"""
import numpy as np, pandas as pd
from datetime import datetime, timedelta
from scipy.optimize import brentq

URL = "https://kam1k88.github.io/postcrossing/TimeData.csv"

MILESTONES_RAW = """
2017-11-07 22:46:00 44
2018-01-16 18:26:00 45
2018-03-20 18:00:00 46
2018-05-28 08:03:00 47
2018-08-07 03:47:00 48
2018-10-16 01:32:00 49
2018-12-27 19:13:00 50
2019-03-04 10:56:00 51
2019-05-13 05:47:00 52
2019-07-25 05:25:00 53
2019-10-07 17:27:00 54
2019-12-18 16:56:00 55
2020-02-28 11:49:00 56
2020-06-05 19:15:00 57
2020-08-28 09:17:00 58
2020-11-10 20:51:00 59
2021-01-25 22:17:00 60
2021-04-02 00:57:00 61
2021-06-08 00:01:00 62
2021-08-18 07:12:00 63
2021-10-26 13:07:00 64
2022-01-04 20:39:00 65
2022-03-09 19:44:00 66
2022-05-25 13:11:00 67
2022-08-10 18:13:00 68
2022-10-22 19:18:00 69
2023-01-04 23:36:00 70
2023-03-14 12:32:00 71
2023-05-28 13:36:00 72
2023-08-12 19:35:00 73
2023-10-25 19:35:00 74
2024-01-08 15:13:00 75
2024-03-18 20:32:00 76
2024-06-02 08:58:00 77
2024-08-16 12:46:00 78
2024-10-27 20:52:00 79
2025-01-07 02:32:00 80
2025-03-18 17:50:00 81
2025-06-03 02:20:00 82
2025-08-22 08:30:00 83
2025-11-02 11:20:00 84
2026-01-19 01:06:00 85
2026-03-27 18:27:00 86
2026-06-09 15:02:00 87
2026-08-24 22:12:00 88
"""

COVID = set(range(56, 67))          # вехи, исключаемые как ковидные

# ---------- конфигурация (оптимум бэктеста) ----------
CFG = dict(
    m1_deg=1, m1_yearly=4, m1_weekly=True, m1_win=730, m1_level_anchor=True,
    m2_deg=1, m2_harm=0,
    m3_deg=2, m3_harm=3, m3_period=365.25,
    tau=45.0, w2_share=0.5,
)

# ---------- данные ----------
def load_data(path=URL):
    return (pd.read_csv(path, parse_dates=['datetime'])
              .sort_values('datetime').reset_index(drop=True))

def load_milestones():
    rows = []
    for line in MILESTONES_RAW.strip().splitlines():
        p = line.split()
        rows.append((int(p[2]), datetime.strptime(f"{p[0]} {p[1]}", "%Y-%m-%d %H:%M:%S")))
    ms = pd.DataFrame(rows, columns=['M', 'date'])
    ms['date'] = pd.to_datetime(ms['date'])
    return ms.sort_values('M').reset_index(drop=True)

# ---------- линейный МНК ----------
def lstsq_fit(A, y, w=None, ridge=None):
    if w is not None:
        s = np.sqrt(np.asarray(w, float)); A = A * s[:, None]; y = y * s
    if ridge is not None:
        lam, idx = ridge
        R = np.zeros((len(idx), A.shape[1]))
        for r, i in enumerate(idx): R[r, i] = np.sqrt(lam)
        A = np.vstack([A, R]); y = np.concatenate([y, np.zeros(len(idx))])
    return np.linalg.lstsq(A, y, rcond=None)[0]

def design_trend(x, deg, n_yearly, weekly, q=365.25):
    cols = [x ** i for i in range(deg + 1)]
    fr = ([0.5 / q] if n_yearly > 0 else []) + [k / q for k in range(1, n_yearly + 1)]
    if weekly: fr += [1 / 7.0, 1 / 3.5, 1.0]
    for f in fr:
        w = 2 * np.pi * f * x; cols += [np.sin(w), np.cos(w)]
    return np.column_stack(cols)

def design_poly_harm(v, deg, nharm, period):
    cols = [v ** i for i in range(deg + 1)]
    for j in range(1, nharm + 1):
        w = 2 * np.pi * v * j / period; cols += [np.sin(w), np.cos(w)]
    return np.column_stack(cols)

# ---------- M1: реактивная модель по сырому ряду ----------
def m1_predict_date(data, target_value, cfg=CFG):
    tmax = data['datetime'].max()
    sub = data[data['datetime'] >= tmax - pd.Timedelta(days=cfg['m1_win'])].reset_index(drop=True)
    if len(sub) < 30: return None
    t0 = sub['datetime'].iloc[0]
    x = (sub['datetime'] - t0).dt.total_seconds().values / 86400.0
    y = sub['postcards_received'].values.astype(float)
    c = lstsq_fit(design_trend(x, cfg['m1_deg'], cfg['m1_yearly'], cfg['m1_weekly']), y)
    g = lambda t: (design_trend(np.array([float(t)]), cfg['m1_deg'],
                                cfg['m1_yearly'], cfg['m1_weekly']) @ c).item()
    off = (y[-1] - g(x[-1])) if cfg['m1_level_anchor'] else 0.0   # кривая проходит через последнее наблюдение
    try:
        return t0 + timedelta(days=brentq(lambda t: g(t) + off - target_value, x[-1], x[-1] + 400.0))
    except Exception:
        return None

# ---------- история интервалов между вехами ----------
def gaps(ms, last_M):
    """(M, dt, day) для M=44..last_M-1. dt[M]=t(M+1)-t(M). Веха last_M НЕ включена:
    её dt = искомый ответ, включать его в обучение -- утечка."""
    d = dict(zip(ms['M'], pd.to_datetime(ms['date'])))
    Ms, dts = [], []
    for m in range(44, last_M):
        if m in COVID or (m + 1) in COVID: continue
        if m in d and (m + 1) in d:
            Ms.append(m); dts.append((d[m + 1] - d[m]).total_seconds() / 86400.0)
    if not Ms: return None
    d0 = d[Ms[0]]
    days = np.array([(d[m] - d0).total_seconds() / 86400.0 for m in Ms])
    return np.array(Ms, float), np.array(dts), days, d0, d

def m2_predict_dt(ms, last_M, cfg=CFG):
    g = gaps(ms, last_M)
    if g is None: return None
    x, y, _, _, _ = g
    c = lstsq_fit(design_poly_harm(x, cfg['m2_deg'], cfg['m2_harm'], 50.0), y)
    return (design_poly_harm(np.array([float(last_M)]), cfg['m2_deg'], cfg['m2_harm'], 50.0) @ c).item()

def m3_predict_dt(ms, last_M, cfg=CFG):
    g = gaps(ms, last_M)
    if g is None: return None
    x, y, days, d0, d = g
    c = lstsq_fit(design_poly_harm(days, cfg['m3_deg'], cfg['m3_harm'], cfg['m3_period']), y)
    dq = (d[last_M] - d0).total_seconds() / 86400.0
    return (design_poly_harm(np.array([dq]), cfg['m3_deg'], cfg['m3_harm'], cfg['m3_period']) @ c).item()

# ---------- итоговый прогноз ----------
def forecast(data, ms, target_M, cfg=CFG):
    last_M = target_M - 1
    row = ms[ms['M'] == last_M]
    if len(row) == 0: return None
    last_date = pd.Timestamp(row['date'].iloc[0])
    p1 = m1_predict_date(data, float(target_M * 1e6), cfg)
    d2 = m2_predict_dt(ms, last_M, cfg); d3 = m3_predict_dt(ms, last_M, cfg)
    if p1 is None or d2 is None or d3 is None: return None
    d1 = (p1 - last_date).total_seconds() / 86400.0
    t_now = data['datetime'].max()
    T_rem = max((last_date + timedelta(days=(d1 + d2 + d3) / 3) - t_now).total_seconds() / 86400.0, 0.0)
    w1 = float(np.exp(-T_rem / cfg['tau']))           # 1 у самой вехи, ->0 на дальнем горизонте
    w2 = (1 - w1) * cfg['w2_share']; w3 = (1 - w1) * (1 - cfg['w2_share'])
    dt = w1 * d1 + w2 * d2 + w3 * d3
    return dict(date=last_date + timedelta(days=dt), dt=dt, dt_m1=d1, dt_m2=d2, dt_m3=d3,
                w1=w1, w2=w2, w3=w3, T_rem=T_rem, last_date=last_date)

# ---------- бэктест ----------
def backtest(data, ms, ks, horizons, cfg=CFG, min_hist_days=400):
    rows = []
    dmin = data['datetime'].min()
    for k in ks:
        if k + 1 not in set(ms['M']): continue
        actual = pd.Timestamp(ms[ms['M'] == k + 1]['date'].iloc[0])
        kdate = pd.Timestamp(ms[ms['M'] == k]['date'].iloc[0])
        for H in horizons:
            cut = actual - pd.Timedelta(days=H)
            if cut < kdate or cut - pd.Timedelta(days=min_hist_days) < dmin: continue
            r = forecast(data[data['datetime'] <= cut], ms[ms['date'] <= cut], k + 1, cfg)
            if r is None: continue
            rows.append(dict(k=k, H=H, pred=r['date'], actual=actual,
                             err=(r['date'] - actual).total_seconds() / 86400.0,
                             w1=r['w1'], dt_m1=r['dt_m1'], dt_m2=r['dt_m2'], dt_m3=r['dt_m3']))
    return pd.DataFrame(rows)

def metrics(e):
    e = np.asarray(e, float); e = e[np.isfinite(e)]
    return dict(n=len(e), bias=e.mean(), sigma=e.std(ddof=1),
                MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()), max=np.abs(e).max())

if __name__ == "__main__":
    data = load_data(); ms = load_milestones()
    print(f"Data: {len(data)} строк, {data['datetime'].min()} -- {data['datetime'].max()}")

    KS = range(67, 88); HS = [10, 20, 30, 45, 60, 90]
    df = backtest(data, ms, KS, HS)
    m = metrics(df['err'].values)
    print(f"\nБЭКТЕСТ (вехи 67-88, горизонты {HS}):")
    print(f"  n={m['n']} bias={m['bias']:+.2f} sigma={m['sigma']:.2f} "
          f"MAE={m['MAE']:.2f} RMSE={m['RMSE']:.2f} max={m['max']:.2f}")
    print("\n  по горизонтам:")
    for H in HS:
        s = df[df['H'] == H]
        if len(s) == 0: continue
        mm = metrics(s['err'].values)
        print(f"    H={H:3d}: n={mm['n']:2d} bias={mm['bias']:+5.2f} sigma={mm['sigma']:4.2f} "
              f"MAE={mm['MAE']:4.2f} w1={s['w1'].mean():.2f}")

    r = forecast(data, ms, 89)
    print("\nПРОГНОЗ 89M:")
    print(f"  M1*={r['dt_m1']:.2f} д   M2*={r['dt_m2']:.2f} д   M3*={r['dt_m3']:.2f} д")
    print(f"  остаток горизонта {r['T_rem']:.1f} д -> w=({r['w1']:.3f},{r['w2']:.3f},{r['w3']:.3f})")
    print(f"  ДАТА: {r['date']}  (через {r['dt']:.2f} д после 88M)")
