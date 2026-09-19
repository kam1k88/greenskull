# reverse_split.py
import numpy as np
import fast_method as N

data = N.load_data()
ms = N.load_milestones()

HS = [10, 20, 30, 45, 60, 90]

# Прямой сплит (уже сделали): train 67-81, test 82-87
train_forward = range(67, 82)
test_forward  = range(82, 88)

# Обратный сплит: train 82-87, test 67-81
train_reverse = range(82, 88)
test_reverse  = range(67, 82)

for name, train_ks, test_ks in [
    ('forward  67-81 → 82-87', train_forward, test_forward),
    ('reverse  82-87 → 67-81', train_reverse, test_reverse),
]:
    # Подбор tau на train
    best_mae = np.inf
    best_tau = None
    for tau in [30, 40, 45, 55, 70, 90]:
        cfg = dict(N.CFG); cfg['tau'] = tau
        df = N.backtest(data, ms, train_ks, HS, cfg=cfg)
        if len(df) == 0: continue
        mae = np.abs(df['err'].values).mean()
        if mae < best_mae:
            best_mae = mae
            best_tau = tau
    
    # Оценка на test с лучшим tau
    cfg = dict(N.CFG); cfg['tau'] = best_tau
    df_test = N.backtest(data, ms, test_ks, HS, cfg=cfg)
    test_mae = np.abs(df_test['err'].values).mean()
    
    print(f"{name}: train MAE={best_mae:.2f} (tau={best_tau}), "
          f"test MAE={test_mae:.2f}, n_train={len(df)}, n_test={len(df_test)}")