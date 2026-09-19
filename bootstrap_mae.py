# bootstrap_mae.py
import numpy as np
import fast_method as N
import pandas as pd

data = N.load_data()
ms = N.load_milestones()

KS = range(67, 88)
HS = [10, 20, 30, 45, 60, 90]
df = N.backtest(data, ms, KS, HS)

# Train/test split
train_ks = range(67, 82)
test_ks  = range(82, 88)

train = df[df['k'].isin(train_ks)]['err'].values
test  = df[df['k'].isin(test_ks)]['err'].values

# Bootstrap CI для MAE
def boot_mae(e, n_boot=2000):
    rng = np.random.default_rng(42)
    maes = []
    for _ in range(n_boot):
        sample = rng.choice(e, size=len(e), replace=True)
        maes.append(np.abs(sample).mean())
    return np.percentile(maes, [2.5, 50, 97.5])

print(f"Train MAE: {np.abs(train).mean():.2f}  95% CI: {boot_mae(train)}")
print(f"Test  MAE: {np.abs(test).mean():.2f}  95% CI: {boot_mae(test)}")