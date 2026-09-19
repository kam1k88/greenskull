# -*- coding: utf-8 -*-
"""
sergei.py -- метод Сергея (greenskull), реализованный по его описанию.

Отличия от calibration.py, каждое -- по цитатам автора:

1) q -- ПОДБИРАЕМЫЙ параметр, а не константа.
   Цитата: "q = (random.uniform(0, (365.25-q)*epsilon))+q" -- q стоит в списке
   встряхиваемых параметров. В calibration.py q=365.25 и q4=50 зашиты жёстко,
   из-за чего модель становится линейной по всем параметрам, имеет единственный
   оптимум, а встряска и 300 итераций не делают ничего (проверено: n_iter=1 и
   n_iter=30 дают побитово одинаковый ответ).

2) Две БУДУЩИЕ точки в M1.
   Цитата: "первую регрессию я делаю не по прошлым точкам. А по прошлым и двум
   будущим. Чтобы её не уводило. А две других по прошлым и одной будущей."
   В calibration.py якоря -- это (88M, прошлое) и (89M, будущее), т.е. одна
   будущая. Здесь: две будущие -- оценки 89M и 90M.

3) Веса меняются вдоль пути.
   Цитата: "среднее между ними с весами, которые меняются от равных в начале
   пути и доминировании первой регрессии ближе к концу."
   В calibration.py вес считается от ПОЛНОГО интервала между вехами (~74 дня,
   почти константа), поэтому w1=0.52..0.54 на любом горизонте -- механизм
   выключен. Здесь вес считается от положения на пути от вехи N к вехе N+1.

4) Индексация уровней: g(n) = t(n) - t(n-1), "сколько дней заняло взять
   уровень n". Тогда цель -- g(89), последняя известная точка -- g(88), якорь
   "одна будущая" -- ровно в n=89. В calibration.py данные размечены как
   t(n+1)-t(n), а модель читается в n=89, т.е. экстраполяция на два шага.

Как решается. Модель ЛИНЕЙНА по всем амплитудам/фазам при фиксированном q
(a*sin(wx+c) = A*sin(wx)+B*cos(wx)), поэтому вместо 300 случайных стартов LM
по 36 параметрам применяется variable projection: q сканируется, остальное
решается точным МНК. Это тот же самый глобальный оптимум, что искала встряска,
но находится он гарантированно и в ~1000 раз быстрее. Литеральная MC+LM версия
оставлена в mc_lm_fit() для сверки.
"""
import numpy as np, pandas as pd
from datetime import datetime, timedelta
from scipy.optimize import brentq, minimize_scalar, least_squares

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
COVID = set(range(56, 67))     # красные уровни у Сергея -- игнор

CFG = dict(
    # --- Model 1: тренд + сезонность по сырому ряду ---
    m1_win=730.0,              # "выборка длиной два года"
    m1_deg=2,                  # a + b*x + c*x^2
    m1_nyearly=12,             # гармоники k=1..12 ("ничего короче 1/12 года")
    m1_half=True,              # a3*sin(2pi*x*0.5/q + c3)
    m1_weekly=(7.0, 3.5, 1.0), # "оставить неделю, полнедели, 1 день"
    m1_q0=365.25, m1_q_lo=300.0, m1_q_hi=430.0,
    # --- Model 2/3: дни между уровнями ---
    m23_deg=3,                 # a + b*n + c*n^2 + d*n^3
    m23_nharm=3,               # q4, q4/2, q4/3
    m2_q0=50.0, m2_q_lo=4.0,  m2_q_hi=200.0,
    m3_q0=50.0, m3_q_lo=20.0, m3_q_hi=4000.0,
    m23_ridge=50.0,            # штраф на кубический коэффициент
    # --- якоря ---
    anchor_w=1.0,              # вес одной будущей точки в сумме квадратов
    # --- сегментация уровней по цветам Сергея ---
    min_level=44,              # белые (1-43) по умолчанию не берутся вовсе
    seg_green_n=8,             # последние N уровней -- "зелёные" (свежие)
    w_green=1.0, w_yellow=1.0, w_white=0.0,
    two_future=True,           # M1 по прошлым и ДВУМ будущим
    m1_level_anchor=False,     # ДОБАВКА (не из описания Сергея): сдвигать кривую
                               # M1 так, чтобы она прошла через последнее наблюдение
    # --- веса смеси ---
    weight_mode='progress',    # 'progress' | 'remaining' | 'fixed'
    tau=45.0, gamma=1.0, w2_share=0.6,
    max_iter=8, tol=0.05,
)


# Конфигурация "как описал Сергей" -- CFG выше.
# Ниже -- улучшенная, каждый пункт подтверждён парным бэктестом на 105 точках
# (вехи 67-88, горизонты 10..60 дней); подбор проверен на отложенных вехах 82-87.
CFG_IMPROVED = dict(CFG)
CFG_IMPROVED.update(
    m1_q_lo=365.25, m1_q_hi=365.25,   # годовой период НЕ подбирать: свободный q
                                      # уходит на 372-409 д и переобучается
                                      # (MAE 2.62 -> 2.20 на бэктесте)
    m1_nyearly=4,                     # его же сомнение про "избыток гармоник":
                                      # 4 гармоники не хуже 12 (MAE 2.20 -> 2.05)
    m3_q_lo=300.0, m3_q_hi=430.0,     # в M3 период искать рядом с годом: свободный
                                      # уходит на 1400+ д (MAE 2.05 -> 1.99)
    m1_level_anchor=True,             # ДОБАВКА сверх описания: кривая M1 проходит
                                      # через последнее наблюдение (MAE 1.99 -> 1.62)
)

# ----------------------------------------------------------------- данные
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

# ------------------------------------------------- дизайн-матрицы моделей
def design_m1(x, q, cfg):
    cols = [x ** i for i in range(cfg['m1_deg'] + 1)]
    fr = ([0.5 / q] if cfg['m1_half'] else []) + [k / q for k in range(1, cfg['m1_nyearly'] + 1)]
    fr = list(fr) + [1.0 / p for p in cfg['m1_weekly']]
    for f in fr:
        w = 2 * np.pi * f * x; cols += [np.sin(w), np.cos(w)]
    return np.column_stack(cols)

def design_m23(v, q4, cfg):
    cols = [v ** i for i in range(cfg['m23_deg'] + 1)]
    for j in range(1, cfg['m23_nharm'] + 1):
        w = 2 * np.pi * v * j / (q4 / j if False else q4 / j)  # периоды q4, q4/2, q4/3
        cols += [np.sin(w), np.cos(w)]
    return np.column_stack(cols)

# --------------------------------------- МНК с весами и ridge + профиль q
def _wls(A, y, w, ridge=None):
    s = np.sqrt(w); Aw = A * s[:, None]; yw = y * s
    if ridge is not None:
        lam, idx = ridge
        R = np.zeros((len(idx), A.shape[1]))
        for r, i in enumerate(idx):
            R[r, i] = np.sqrt(lam)
        Aw = np.vstack([Aw, R]); yw = np.concatenate([yw, np.zeros(len(idx))])
    beta, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    return beta, float(np.sum((Aw @ beta - yw) ** 2))

def fit_profile(x, y, w, design, q0, q_lo, q_hi, ridge=None, n_coarse=400, n_ref=5):
    """Variable projection: сканируем единственный нелинейный параметр q,
    остальные решаются точным взвешенным МНК.
    Ландшафт по q многомодальный (именно поэтому у Сергея встряска q осмысленна),
    поэтому сетка плотная, а затем уточняются n_ref лучших локальных минимумов."""
    def cost(q):
        try:
            return _wls(design(x, q), y, w, ridge)[1]
        except Exception:
            return np.inf
    grid = np.unique(np.concatenate([np.linspace(q_lo, q_hi, n_coarse), [q0]]))
    vals = np.array([cost(q) for q in grid])
    # локальные минимумы сетки
    loc = [i for i in range(len(grid))
           if (i == 0 or vals[i] <= vals[i-1]) and (i == len(grid)-1 or vals[i] <= vals[i+1])]
    loc.sort(key=lambda i: vals[i])
    best_q, best_c = float(grid[int(np.argmin(vals))]), float(np.min(vals))
    for i in loc[:n_ref]:
        lo = grid[max(i - 1, 0)]; hi = grid[min(i + 1, len(grid) - 1)]
        if hi <= lo:
            continue
        r = minimize_scalar(cost, bounds=(lo, hi), method='bounded', options=dict(xatol=1e-4))
        if r.fun < best_c:
            best_c, best_q = float(r.fun), float(r.x)
    beta, rss = _wls(design(x, best_q), y, w, ridge)
    return beta, best_q, rss

def mc_lm_fit(x, y, w, design, q0, n_iter=300, eps=1.0, ridge=None, seed=0):
    """Литеральная версия Сергея: 300 стартов LM со встряской параметров.
    Оставлена для сверки с fit_profile."""
    rng = np.random.default_rng(seed)
    n_lin = design(np.array([0.0]), q0).shape[1]
    p0 = np.concatenate([_wls(design(x, q0), y, w, ridge)[0], [q0]])
    best = (None, np.inf)
    for _ in range(n_iter):
        p = p0.copy()
        p[:n_lin] *= (1 + rng.uniform(-1, 1, n_lin) * eps)
        p[-1] = p0[-1] + rng.uniform(0, (365.25 - p0[-1]) * eps)
        def resid(pp):
            r = (design(x, pp[-1]) @ pp[:n_lin] - y) * np.sqrt(w)
            if ridge is not None:
                lam, idx = ridge
                r = np.concatenate([r, np.sqrt(lam) * pp[list(idx)]])
            return r
        try:
            res = least_squares(resid, p, method='lm', max_nfev=20000)
        except Exception:
            continue
        c = float(np.sum(res.fun ** 2))
        if np.isfinite(c) and c < best[1]:
            best = (res.x, c)
    p = best[0]
    return p[:n_lin], float(p[-1]), best[1]

# ------------------------------------------------- быстрый решатель M1
class M1Cache:
    """Кэш для M1: история ряда не меняется между самосогласованными итерациями,
    меняются только две якорные точки. Поэтому для каждого q один раз считаются
    нормальные уравнения истории, а на каждой итерации к ним добавляются
    2 строки якорей. Математически -- тот же взвешенный МНК."""
    def __init__(self, data, cfg, n_grid=40):
        tmax = data['datetime'].max()
        sub = data[data['datetime'] >= tmax - pd.Timedelta(days=cfg['m1_win'])].reset_index(drop=True)
        self.ok = len(sub) >= 30
        if not self.ok:
            return
        self.cfg = cfg
        self.t0 = sub['datetime'].iloc[0]
        self.x = (sub['datetime'] - self.t0).dt.total_seconds().values / 86400.0
        self.y = sub['postcards_received'].values.astype(float)
        self.xl = (tmax - self.t0).total_seconds() / 86400.0
        self.yty = float(self.y @ self.y)
        self.qs = np.unique(np.concatenate([
            np.linspace(cfg['m1_q_lo'], cfg['m1_q_hi'], n_grid), [cfg['m1_q0']]]))
        self.cache = {}
        for q in self.qs:
            self.cache[q] = self._prep(q)

    def _prep(self, q):
        A = design_m1(self.x, q, self.cfg)
        s = np.linalg.norm(A, axis=0); s[s == 0] = 1.0
        An = A / s
        return s, An.T @ An, An.T @ self.y

    def _solve(self, q, xa, ya, wa):
        if q in self.cache:
            s, G, b = self.cache[q]
        else:
            s, G, b = self._prep(q)
        if len(xa):
            Aa = design_m1(np.asarray(xa, float), q, self.cfg) / s
            G = G + (Aa * wa[:, None]).T @ Aa
            b = b + Aa.T @ (wa * ya)
        try:
            beta = np.linalg.solve(G + 1e-12 * np.eye(len(b)), b)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(G, b, rcond=None)[0]
        rss = self.yty + float(wa @ (ya ** 2)) - 2 * float(b @ beta) + float(beta @ G @ beta)
        return beta / s, rss

    def fit(self, xa, ya, wa, n_ref=3):
        xa = np.asarray(xa, float); ya = np.asarray(ya, float); wa = np.asarray(wa, float)
        vals = np.array([self._solve(q, xa, ya, wa)[1] for q in self.qs])
        loc = [i for i in range(len(self.qs))
               if (i == 0 or vals[i] <= vals[i - 1]) and (i == len(self.qs) - 1 or vals[i] <= vals[i + 1])]
        loc.sort(key=lambda i: vals[i])
        bq, bc = float(self.qs[int(np.argmin(vals))]), float(np.min(vals))
        for i in loc[:n_ref]:
            lo = self.qs[max(i - 1, 0)]; hi = self.qs[min(i + 1, len(self.qs) - 1)]
            if hi <= lo:
                continue
            r = minimize_scalar(lambda q: self._solve(q, xa, ya, wa)[1],
                                bounds=(lo, hi), method='bounded', options=dict(xatol=1e-3))
            if r.fun < bc:
                bc, bq = float(r.fun), float(r.x)
        beta, _ = self._solve(bq, xa, ya, wa)
        return beta, bq

# ------------------------------------------- история "дней между уровнями"
def level_gaps(ms, last_M, cfg=None):
    """g(n) = t(n) - t(n-1). Возвращает n=min_level+1..last_M без ковидных уровней
    и веса уровней по цветовой сегментации Сергея.
    Точка n=last_M известна; прогнозируется g(last_M+1) -- утечки нет."""
    cfg = cfg or CFG
    d = dict(zip(ms['M'], pd.to_datetime(ms['date'])))
    lo = cfg['min_level'] if cfg['w_white'] <= 0 else 2
    ns, gs = [], []
    for n in range(lo + 1, last_M + 1):
        if n in COVID or (n - 1) in COVID:
            continue
        if n in d and (n - 1) in d:
            ns.append(n); gs.append((d[n] - d[n - 1]).total_seconds() / 86400.0)
    if len(ns) < 8:
        return None
    ns = np.array(ns, float); gs = np.array(gs)
    w = np.where(ns < cfg['min_level'], cfg['w_white'], cfg['w_yellow'])
    w = np.where(ns > last_M - cfg['seg_green_n'], cfg['w_green'], w)
    return ns, gs, d, w

# --------------------------------------------------- три регрессии Сергея
def m1_dates(cache, targets, est_dates, cfg):
    """M1: 2 года сырого ряда + ДВЕ будущие якорные точки (оценки N+1 и N+2)."""
    if not cache.ok:
        return None, None
    k = 2 if cfg['two_future'] else 1
    xa = [(ed - cache.t0).total_seconds() / 86400.0 for ed in est_dates[:k]]
    ya = list(targets[:k]); wa = [cfg['anchor_w']] * k
    beta, q = cache.fit(xa, ya, wa)
    g0 = lambda t: (design_m1(np.array([float(t)]), q, cfg) @ beta).item()
    off = (cache.y[-1] - g0(cache.xl)) if cfg.get('m1_level_anchor') else 0.0
    g = lambda t: g0(t) + off
    try:
        return cache.t0 + timedelta(days=brentq(lambda t: g(t) - targets[0],
                                                cache.xl, cache.xl + 400.0)), q
    except Exception:
        return None, q

def m23_predict(ns, gs, days, wlev, anchor_n, anchor_day, anchor_g, cfg):
    """M2 (аргумент -- номер уровня) и M3 (аргумент -- календарный день).
    Обе -- по прошлым точкам и ОДНОЙ будущей."""
    ridge = (cfg['m23_ridge'], [cfg['m23_deg']])
    out = {}
    for tag, xx, xq, q0, lo, hi in (
            ('m2', ns,   anchor_n,   cfg['m2_q0'], cfg['m2_q_lo'], cfg['m2_q_hi']),
            ('m3', days, anchor_day, cfg['m3_q0'], cfg['m3_q_lo'], cfg['m3_q_hi'])):
        X = np.append(xx, xq); Y = np.append(gs, anchor_g)
        W = np.append(wlev, cfg['anchor_w'])
        beta, q4, _ = fit_profile(X, Y, W, lambda v, q: design_m23(v, q, cfg),
                                  q0, lo, hi, ridge=ridge)
        out[tag] = ((design_m23(np.array([float(xq)]), q4, cfg) @ beta).item(), q4)
    return out

# ------------------------------------------------ самосогласованный расчёт
def mix_weights(t_now, last_date, est_date, cfg):
    """Равные веса в начале пути -> доминирование M1 ближе к концу."""
    if cfg['weight_mode'] == 'fixed':
        w1 = 1.0 / 3
    else:
        total = max((est_date - last_date).total_seconds() / 86400.0, 1e-6)
        rem = max((est_date - t_now).total_seconds() / 86400.0, 0.0)
        if cfg['weight_mode'] == 'progress':
            p = min(max(1.0 - rem / total, 0.0), 1.0) ** cfg['gamma']
            w1 = 1.0 / 3 + (2.0 / 3) * p
        else:                                   # 'remaining'
            w1 = 1.0 / 3 + (2.0 / 3) * np.exp(-rem / cfg['tau'])
    rest = 1.0 - w1
    return float(w1), float(rest * cfg['w2_share']), float(rest * (1 - cfg['w2_share']))

def forecast(data, ms, target_M, cfg=CFG, verbose=False):
    last_M = target_M - 1
    row = ms[ms['M'] == last_M]
    if len(row) == 0:
        return None
    last_date = pd.Timestamp(row['date'].iloc[0])
    lg = level_gaps(ms, last_M, cfg)
    if lg is None:
        return None
    ns, gs, d, wlev = lg
    day0 = d[int(ns[0])]
    days = np.array([(d[int(n)] - day0).total_seconds() / 86400.0 for n in ns])
    t_now = data['datetime'].max()
    cache = M1Cache(data, cfg)
    if not cache.ok:
        return None

    est = last_date + timedelta(days=float(np.mean(gs[-3:])))
    est_next = est + timedelta(days=float(np.mean(gs[-3:])))
    g1 = g2 = g3 = w1 = w2 = w3 = q1 = q2 = q3 = None

    for it in range(cfg['max_iter']):
        d1_date, q1 = m1_dates(cache, [target_M * 1e6, (target_M + 1) * 1e6],
                               [est, est_next], cfg)
        if d1_date is None:
            return None
        g1 = (d1_date - last_date).total_seconds() / 86400.0

        anchor_day = (est - day0).total_seconds() / 86400.0
        anchor_g = (est - last_date).total_seconds() / 86400.0
        r = m23_predict(ns, gs, days, wlev, float(target_M), anchor_day, anchor_g, cfg)
        (g2, q2), (g3, q3) = r['m2'], r['m3']

        w1, w2, w3 = mix_weights(t_now, last_date, est, cfg)
        new = last_date + timedelta(days=w1 * g1 + w2 * g2 + w3 * g3)
        delta = abs((new - est).total_seconds()) / 86400.0
        est = new
        est_next = est + timedelta(days=float(np.mean(gs[-3:])))
        if verbose:
            print(f"  it{it}: g=({g1:.2f},{g2:.2f},{g3:.2f}) "
                  f"q=({q1:.1f},{q2:.1f},{q3:.1f}) w1={w1:.3f} -> {est}")
        if delta < cfg['tol']:
            break

    return dict(date=est, g1=g1, g2=g2, g3=g3, q1=q1, q2=q2, q3=q3,
                w1=w1, w2=w2, w3=w3, last_date=last_date,
                dt=(est - last_date).total_seconds() / 86400.0)

# ------------------------------------------------------------- бэктест
def backtest(data, ms, ks, horizons, cfg=CFG, min_hist_days=400, verbose=False):
    rows = []; dmin = data['datetime'].min()
    for k in ks:
        if k + 1 not in set(ms['M']):
            continue
        actual = pd.Timestamp(ms[ms['M'] == k + 1]['date'].iloc[0])
        kdate = pd.Timestamp(ms[ms['M'] == k]['date'].iloc[0])
        for H in horizons:
            cut = actual - pd.Timedelta(days=H)
            if cut < kdate or cut - pd.Timedelta(days=min_hist_days) < dmin:
                continue
            r = forecast(data[data['datetime'] <= cut], ms[ms['date'] <= cut], k + 1, cfg)
            if r is None:
                continue
            e = (r['date'] - actual).total_seconds() / 86400.0
            rows.append(dict(k=k, H=H, pred=r['date'], actual=actual, err=e,
                             w1=r['w1'], g1=r['g1'], g2=r['g2'], g3=r['g3'],
                             q1=r['q1'], q2=r['q2'], q3=r['q3']))
            if verbose:
                print(f"  k={k} H={H:3d} err={e:+6.2f} w1={r['w1']:.2f} "
                      f"q=({r['q1']:.0f},{r['q2']:.0f},{r['q3']:.0f})")
    return pd.DataFrame(rows)

def metrics(e):
    e = np.asarray(e, float); e = e[np.isfinite(e)]
    if len(e) == 0:
        return dict(n=0)
    return dict(n=len(e), bias=e.mean(), sigma=e.std(ddof=1) if len(e) > 1 else np.nan,
                MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()), max=np.abs(e).max())

def report(name, e):
    m = metrics(e)
    print(f"  {name:40s} n={m['n']:3d} bias={m['bias']:+5.2f} sigma={m['sigma']:4.2f} "
          f"MAE={m['MAE']:4.2f} max={m['max']:5.2f}")
    return m

if __name__ == "__main__":
    data = load_data(); ms = load_milestones()
    print(f"Data: {len(data)} строк, {data['datetime'].min()} -- {data['datetime'].max()}")
    KS = range(67, 88); HS = [10, 20, 30, 45, 60]

    print("\nБЭКТЕСТ (вехи 67-88, горизонты 10..60 дней):")
    for tag, cfg in [("Сергей как описано", CFG), ("Сергей улучшенный", CFG_IMPROVED)]:
        df = backtest(data, ms, KS, HS, cfg=cfg)
        report(tag, df['err'])

    print("\nПРОГНОЗ 89M:")
    for tag, cfg in [("Сергей как описано", CFG), ("Сергей улучшенный", CFG_IMPROVED)]:
        r = forecast(data, ms, 89, cfg=cfg)
        print(f"  {tag:20s}: {r['date']}")
        print(f"{'':22s}  M1={r['g1']:.2f} д  M2={r['g2']:.2f} д  M3={r['g3']:.2f} д  "
              f"w=({r['w1']:.2f},{r['w2']:.2f},{r['w3']:.2f})")
        print(f"{'':22s}  подобранные периоды: q1={r['q1']:.1f} д, q2={r['q2']:.1f} ур., q3={r['q3']:.1f} д")
