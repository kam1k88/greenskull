# =============================================================
# fastcal.py - та же модель Сергея, но через lstsq (модели линейны
# по всем параметрам после замены a*sin(wx+c) -> A*sin(wx)+B*cos(wx)).
# Плюс флаги для проверки исправлений.
# =============================================================
import numpy as np, pandas as pd
from datetime import timedelta
from scipy.optimize import brentq
from calibration import load_data, MILESTONES_RAW
from datetime import datetime

def load_milestones_raw():
    rows=[]
    for line in MILESTONES_RAW.strip().splitlines():
        p=line.strip().split()
        if len(p)!=3: continue
        rows.append((int(p[2])//1_000_000, datetime.strptime(f"{p[0]} {p[1]}","%Y-%m-%d %H:%M:%S")))
    ms=pd.DataFrame(rows,columns=['M','date']); ms['date']=pd.to_datetime(ms['date'])
    return ms.sort_values('M').reset_index(drop=True)

MS = load_milestones_raw()

# ---------- дизайн-матрицы ----------
def design_m1(x,q=365.25,n_yearly=12,weekly=True):
    cols=[np.ones_like(x),x,x*x]
    freqs=[0.5/q]+[k/q for k in range(1,n_yearly+1)]
    if weekly: freqs += [1/7.0,1/3.5,1.0]
    for f in freqs:
        w=2*np.pi*f*x; cols += [np.sin(w),np.cos(w)]
    return np.column_stack(cols)

def design_m2(n,q4=50.0,deg=3,nharm=3):
    cols=[n**k for k in range(deg+1)]
    for j in range(1,nharm+1):
        w=2*np.pi*n*j/q4; cols += [np.sin(w),np.cos(w)]
    return np.column_stack(cols)

def lstsq_fit(A,y,w=None,ridge=None):
    if w is not None:
        s=np.sqrt(np.asarray(w,float)); A=A*s[:,None]; y=y*s
    if ridge is not None:
        lam,idx=ridge
        R=np.zeros((len(idx),A.shape[1]))
        for r,i in enumerate(idx): R[r,i]=np.sqrt(lam)
        A=np.vstack([A,R]); y=np.concatenate([y,np.zeros(len(idx))])
    return np.linalg.lstsq(A,y,rcond=None)[0]

# ---------- прогноз ----------
def forecast(data, ms, target_M, tau=60.0,
             leak='keep',        # 'keep' = как сейчас, 'drop' = пересчитать dt по обрезанным вехам
             idx='orig',         # 'orig' = как сейчас, 'shift' = f(M)=t(M)-t(M-1)
             wmode='total',      # 'total' = как сейчас, 'remaining' = вес по остатку горизонта
             q4_m3=50.0, q4_m2=50.0,
             n_yearly=12, ridge_lambda=50.0, win_days=730.0,
             tau_w=np.inf, max_iter=8, tol=0.05, verbose=False):
    last_M=target_M-1
    row=ms[ms['M']==last_M]
    if len(row)==0: return None
    last_date=pd.Timestamp(row['date'].iloc[0])
    last_val=float(last_M*1e6); target_val=float(target_M*1e6)

    m=ms[(ms['M']>=44)&(ms['M']<=last_M)&(~ms['M'].isin(range(56,67)))].copy()
    if leak=='drop':
        m['dt']=(m['date'].shift(-1)-m['date']).dt.total_seconds()/86400.0
        # разрыв там, где соседняя веха выпала (ковид) - тоже убираем
        m.loc[m['M'].diff(-1)!=-1,'dt']=np.nan
    else:
        full=MS.copy(); full['dt']=(full['date'].shift(-1)-full['date']).dt.total_seconds()/86400.0
        m=m.merge(full[['M','dt']],on='M',how='left')
    mf=m.dropna(subset=['dt']).reset_index(drop=True)
    if len(mf)<10: return None

    shift = 1 if idx=='shift' else 0
    m2_x=mf['M'].values.astype(float)+shift
    m2_y=mf['dt'].values.astype(float)
    day0=pd.Timestamp(mf['date'].iloc[0])
    # для 'shift' аргумент M3 - дата ДОСТИЖЕНИЯ вехи M+1, т.е. конец интервала
    if shift:
        m3_x=((mf['date']+pd.to_timedelta(mf['dt'],unit='D'))-day0).dt.total_seconds().values/86400.0
    else:
        m3_x=(mf['date']-day0).dt.total_seconds().values/86400.0

    w_ms = np.ones(len(mf)) if np.isinf(tau_w) else np.exp(-(last_M-mf['M'].values)/tau_w)

    t_data_max=data['datetime'].max()
    cutoff=t_data_max-pd.Timedelta(days=win_days)
    sub=data[data['datetime']>=cutoff].reset_index(drop=True)
    if len(sub)<10: sub=data.tail(100).reset_index(drop=True)
    t0=sub['datetime'].iloc[0]
    t_hist=(sub['datetime']-t0).dt.total_seconds().values/86400.0
    y_hist=sub['postcards_received'].values.astype(float)
    t_a1=(last_date-t0).total_seconds()/86400.0
    t_last=(t_data_max-t0).total_seconds()/86400.0

    est_date=last_date+timedelta(days=75.0)
    dt_m1=dt_m2=dt_m3=w1=w2=w3=None
    for it in range(max_iter):
        t_a2=(est_date-t0).total_seconds()/86400.0
        X=np.concatenate([t_hist,[t_a1,t_a2]]); Y=np.concatenate([y_hist,[last_val,target_val]])
        c1=lstsq_fit(design_m1(X,n_yearly=n_yearly),Y)
        f1=lambda t: (design_m1(np.array([float(t)]),n_yearly=n_yearly)@c1).item()-target_val
        try: date_m1=t0+timedelta(days=brentq(f1,t_last,t_last+400.0))
        except Exception: return None

        dt_est=(est_date-last_date).total_seconds()/86400.0
        xa=np.append(m2_x,target_M); ya=np.append(m2_y,dt_est); wa=np.append(w_ms,1.0)
        c2=lstsq_fit(design_m2(xa,q4=q4_m2),ya,w=wa,ridge=(ridge_lambda,[3]))
        dt_m2=(design_m2(np.array([float(target_M)]),q4=q4_m2)@c2).item()

        day_est=(est_date-day0).total_seconds()/86400.0
        xb=np.append(m3_x,day_est)
        c3=lstsq_fit(design_m2(xb,q4=q4_m3),ya,w=wa,ridge=(ridge_lambda,[3]))
        dt_m3=(design_m2(np.array([float(day_est)]),q4=q4_m3)@c3).item()

        dt_m1=(date_m1-last_date).total_seconds()/86400.0
        if wmode=='total':
            T=(dt_m1+dt_m2+dt_m3)/3.0
        else:  # остаток горизонта: сколько дней от последних данных до цели
            T=max(((last_date+timedelta(days=(dt_m1+dt_m2+dt_m3)/3.0))-t_data_max).total_seconds()/86400.0, 0.0)
        w1=1.0/3+(2.0/3)*np.exp(-T/tau); rest=1-w1; w2,w3=rest*0.6,rest*0.4
        est_new=last_date+timedelta(days=w1*dt_m1+w2*dt_m2+w3*dt_m3)
        d=abs((est_new-est_date).total_seconds())/86400.0
        est_date=est_new
        if verbose: print(f"  it{it}: dt=({dt_m1:.2f},{dt_m2:.2f},{dt_m3:.2f}) w1={w1:.3f} -> {est_date}")
        if d<tol: break
    return dict(date=est_date,dt_m1=dt_m1,dt_m2=dt_m2,dt_m3=dt_m3,w1=w1,w2=w2,w3=w3)

# ---------- бэктест ----------
def backtest(data, ks, horizons, min_hist_days=400, **kw):
    ms_full=MS
    dmin=data['datetime'].min(); rows=[]
    for k in ks:
        if k+1 not in set(ms_full['M']): continue
        actual=pd.Timestamp(ms_full[ms_full['M']==k+1]['date'].iloc[0])
        kdate=pd.Timestamp(ms_full[ms_full['M']==k]['date'].iloc[0])
        for H in horizons:
            cut=actual-pd.Timedelta(days=H)
            if cut<kdate or cut-pd.Timedelta(days=min_hist_days)<dmin: continue
            dc=data[data['datetime']<=cut]
            mc=ms_full[ms_full['date']<=cut]
            try: r=forecast(dc,mc,target_M=k+1,**kw)
            except Exception: r=None
            if r is None: continue
            rows.append(dict(k=k,H=H,pred=r['date'],actual=actual,
                             err=(r['date']-actual).total_seconds()/86400.0,
                             w1=r['w1'],dt_m1=r['dt_m1'],dt_m2=r['dt_m2'],dt_m3=r['dt_m3']))
    return pd.DataFrame(rows)

def met(e):
    e=np.asarray(e,float); e=e[np.isfinite(e)]
    return dict(n=len(e),bias=e.mean(),sigma=e.std(ddof=1),MAE=np.abs(e).mean(),
                RMSE=np.sqrt((e**2).mean()),max=np.abs(e).max())

def show(name,df):
    m=met(df['err'].values)
    print(f"{name:34s} n={m['n']:3d} bias={m['bias']:+5.2f} sigma={m['sigma']:4.2f} "
          f"MAE={m['MAE']:4.2f} RMSE={m['RMSE']:4.2f} max={m['max']:5.2f}")
    return m
