from calibration import load_data, load_milestones, forecast
data = load_data()
ms = load_milestones()
res = forecast(data, ms, target_M=89, n_iter_mc=200, tau=60.0, verbose=True)
print(res['date'] if res else None)