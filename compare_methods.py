# -*- coding: utf-8 -*-
"""
compare_methods.py -- честное сравнение старого и нового метода на одном
наборе точек. Требует: calibration.py (старый), fastcal.py (его точная
lstsq-копия с флагами), fast_method.py (новый).

Важно: в старом бэктесте была УТЕЧКА. ms['dt'] считается в load_milestones()
на ПОЛНОМ списке вех, а backtest_multihorizon только фильтрует строки. Поэтому
в обучающей выборке M2/M3 остаётся строка M=k со значением dt = t(k+1)-t(k) --
ровно тот интервал, который прогнозируется. Здесь для старого метода берётся
вариант leak='drop' (dt пересчитывается по обрезанному списку).
"""
import numpy as np, pandas as pd
import fastcal as F           # старый метод через lstsq (идентичен LM-версии)
import fast_method as N       # новый метод

KS = range(67, 88)
HS = [10, 20, 30, 45, 60, 90]

data = N.load_data()  # при локальном запуске: N.load_data('TimeData.csv')
ms_new = N.load_milestones()

def rep(name, e):
    e = np.asarray(e, float); e = e[np.isfinite(e)]
    print(f"  {name:36s} n={len(e):3d} bias={e.mean():+5.2f} sigma={e.std(ddof=1):4.2f} "
          f"MAE={np.abs(e).mean():4.2f} max={np.abs(e).max():5.2f}")

def naive(dc, tv, win=365):
    te = dc['datetime'].iloc[-1]; ye = dc['postcards_received'].iloc[-1]
    w = dc[dc['datetime'] >= te - pd.Timedelta(days=win)]
    b = np.polyfit((w['datetime'] - te).dt.total_seconds().values / 86400.0,
                   w['postcards_received'].values.astype(float), 1)[0]
    return te + pd.Timedelta(days=(tv - ye) / b)

print("СРАВНЕНИЕ НА ОДНИХ И ТЕХ ЖЕ ТОЧКАХ (вехи 67-88, горизонты 10-90 дней)\n")
old_leak = F.backtest(data, KS, HS, tau=60.0, leak='keep')
old      = F.backtest(data, KS, HS, tau=60.0, leak='drop')
new      = N.backtest(data, ms_new, KS, HS)
nv = []
for _, r in old.iterrows():
    cut = r['actual'] - pd.Timedelta(days=r['H'])
    nv.append((naive(data[data['datetime'] <= cut], float((r['k'] + 1) * 1e6)) - r['actual']).total_seconds() / 86400.0)

rep("старый метод (как есть, с утечкой)", old_leak['err'])
rep("старый метод (утечка убрана)",       old['err'])
rep("наивный лин. тренд 365 дней",        nv)
rep("НОВЫЙ метод",                        new['err'])

print("\nПо горизонтам (MAE, дни):")
print(f"{'H':>4} {'старый':>9} {'наивный':>9} {'новый':>9}")
for H in HS:
    o = old[old['H'] == H]['err'].values; n = new[new['H'] == H]['err'].values
    nn = np.array([v for v, (_, r) in zip(nv, old.iterrows()) if r['H'] == H])
    if len(o) == 0: continue
    print(f"{H:>4} {np.abs(o).mean():9.2f} {np.abs(nn).mean():9.2f} {np.abs(n).mean():9.2f}")

print("\nПРОГНОЗ 89M:")
r = N.forecast(data, ms_new, 89)
print(f"  новый метод: {r['date']}  (w1={r['w1']:.3f}, dt={r['dt']:.2f} д)")
