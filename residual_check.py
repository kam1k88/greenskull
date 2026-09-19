# -*- coding: utf-8 -*-
"""
residual_check.py -- есть ли в остатках M1 систематика по дню года?

Логика: если годовые гармоники (k=1..4) не ловят реальную форму сезонности,
остаток модели будет не шумом, а повторяющейся из года в год кривой.
Проверяем это на СКОРОСТИ (карточек в день), а не на кумуляте: в кумуляте
остаток -- это интеграл, он автокоррелирован и его форма обманчива.
"""
import numpy as np, pandas as pd
import sergei as S

def daily_rate(data):
    """Суточная скорость по кумулятивному ряду, приведённая к календарным суткам."""
    d = data.sort_values('datetime').reset_index(drop=True)
    t = d['datetime'].values.astype('datetime64[s]').astype(float) / 86400.0
    y = d['postcards_received'].values.astype(float)
    dt = np.diff(t); dy = np.diff(y)
    ok = (dt > 0.2) & (dt < 3.0)            # выбрасываем слишком частые и дырки
    mid = (t[:-1] + t[1:]) / 2.0
    return pd.DataFrame(dict(t=mid[ok], rate=(dy / dt)[ok])).reset_index(drop=True)

def fit_and_residuals(data, cfg, win=730.0):
    """Фитим M1 на окне, возвращаем остатки скорости (наблюдение - модель)."""
    tmax = data['datetime'].max()
    sub = data[data['datetime'] >= tmax - pd.Timedelta(days=win)].reset_index(drop=True)
    t0 = sub['datetime'].iloc[0]
    x = (sub['datetime'] - t0).dt.total_seconds().values / 86400.0
    y = sub['postcards_received'].values.astype(float)
    cache = S.M1Cache(data, cfg)
    beta, q = cache.fit([], [], [])
    A = S.design_m1(x, q, cfg)
    fit = A @ beta
    # скорость модели и скорость данных на тех же серединах интервалов
    dt = np.diff(x); ok = (dt > 0.2) & (dt < 3.0)
    mid_t = ((x[:-1] + x[1:]) / 2.0)[ok]
    r_obs = (np.diff(y) / dt)[ok]
    r_mod = (np.diff(fit) / dt)[ok]
    dates = t0 + pd.to_timedelta(mid_t, unit='D')
    return pd.DataFrame(dict(date=dates, doy=dates.dayofyear,
                             obs=r_obs, mod=r_mod, res=r_obs - r_mod))

def circular_smooth(doy, val, bw=10.0):
    """Сглаживание по дню года с учётом того, что 365 и 1 -- соседи."""
    grid = np.arange(1, 366)
    out = np.zeros(len(grid))
    for i, g in enumerate(grid):
        d = np.abs(doy - g); d = np.minimum(d, 365 - d)
        w = np.exp(-0.5 * (d / bw) ** 2)
        out[i] = np.sum(w * val) / max(np.sum(w), 1e-9)
    return grid, out

if __name__ == "__main__":
    import sys
    data = S.load_data(sys.argv[1] if len(sys.argv)>1 else S.URL); ms = S.load_milestones()
    for tag, cfg in [("K=4 (CFG_IMPROVED)", S.CFG_IMPROVED), ("K=12 (как у Сергея)", S.CFG)]:
        r = fit_and_residuals(data, cfg)
        grid, sm = circular_smooth(r['doy'].values, r['res'].values)
        amp = (sm.max() - sm.min()) / 2
        # сколько дней сдвига даёт такая систематика на горизонте 60 дней
        mean_rate = r['obs'].mean()
        print(f"\n{tag}: остаток скорости, n={len(r)}")
        print(f"  СКО остатка            : {r['res'].std():8.0f} карточек/день")
        print(f"  амплитуда сезонной части: {amp:8.0f} карточек/день "
              f"({amp/mean_rate:.1%} от средней скорости {mean_rate:.0f})")
        print(f"  эквивалент на 60 днях  : {amp*60/mean_rate:8.2f} дня сдвига прогноза")
        worst = grid[np.argsort(sm)[:3]]; best = grid[np.argsort(sm)[-3:]]
        f = lambda dd: ", ".join(pd.Timestamp('2025-01-01').__add__(pd.Timedelta(days=int(d)-1)).strftime('%d %b') for d in dd)
        print(f"  модель СПЕШИТ (остаток<0) около: {f(worst)}")
        print(f"  модель ОТСТАЁТ (остаток>0) около: {f(best)}")
        # проверка на воспроизводимость: делим окно пополам по годам
        yrs = r['date'].dt.year
        halves = [r[yrs == y] for y in sorted(yrs.unique()) if (yrs == y).sum() > 60]
        if len(halves) >= 2:
            cs = [circular_smooth(h['doy'].values, h['res'].values)[1] for h in halves]
            cc = np.corrcoef(cs[0], cs[1])[0, 1]
            print(f"  корреляция формы между годами: {cc:+.2f}  "
                  f"({'систематика' if cc > 0.4 else 'скорее шум'})")
