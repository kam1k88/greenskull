# -*- coding: utf-8 -*-
"""Где предел точности: насколько будущую скорость вообще можно знать заранее."""
import numpy as np, pandas as pd
import sergei as S
data=S.load_data('TimeData.csv')
d=data[data['datetime']>=pd.Timestamp('2023-01-01')].reset_index(drop=True)
t=d['datetime'].values.astype('datetime64[s]').astype(float)/86400.
y=d['postcards_received'].values.astype(float)
def val(tq): return np.interp(tq,t,y)

print("Относительная ошибка оценки БУДУЩЕЙ средней скорости по истории")
print("(плотная эпоха 2023-2026, шаг среза 3 дня)\n")
print(f"{'H':>4} " + " ".join(f"{'окно '+str(w)+'д':>13}" for w in [30,60,90,180,365]))
rows={}
for H in [10,20,30,45,60]:
    line=f"{H:>4} "
    for w in [30,60,90,180,365]:
        errs=[]
        for tc in np.arange(t[0]+w, t[-1]-H, 3.0):
            rf=(val(tc+H)-val(tc))/H
            rh=(val(tc)-val(tc-w))/w
            errs.append(H*(rf/rh-1.0))
        e=np.array(errs); rows[(H,w)]=e
        line+=f"{np.abs(e).mean():6.2f}/{e.std():5.2f} "
    print(line+"   <- MAE/sigma в ДНЯХ сдвига прогноза")
print("\nЛучшее окно на каждом горизонте (по sigma):")
for H in [10,20,30,45,60]:
    best=min([30,60,90,180,365],key=lambda w: rows[(H,w)].std())
    e=rows[(H,best)]
    print(f"  H={H:3d}: окно {best:3d} д -> MAE={np.abs(e).mean():.2f} д, sigma={e.std():.2f} д  (n={len(e)})")
print("\nДля сравнения, наш бэктест по вехам:")
bt=pd.read_csv('bt_best.csv')
for H in [10,20,30,45,60]:
    s=bt[bt['H']==H]['err']
    print(f"  H={H:3d}: MAE={s.abs().mean():.2f} д, sigma={s.std(ddof=1):.2f} д  (n={len(s)})")
