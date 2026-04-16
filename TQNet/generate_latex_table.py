import pandas as pd
import numpy as np

df = pd.read_csv('electricity_results_summary.csv', keep_default_na=False)

df = df[df['mse'] != 'NaN']
df = df[df['mae'] != 'NaN']

df['mse'] = df['mse'].astype(float)
df['mae'] = df['mae'].astype(float)

grouped = df.groupby(['pred_len', 'loss']).agg({
    'mse': ['mean', 'std'], 
    'mae': ['mean', 'std']
}).reset_index()
grouped.columns = ['pred_len', 'loss', 'mse_mean', 'mse_std', 'mae_mean', 'mae_std']

def fmt(m, s):
    if pd.isna(m) or pd.isna(s): return "$-$"
    return "$%.3f_{\\pm %.3f}$" % (m, s)

res = {}
for _, row in grouped.iterrows():
    pl = int(row['pred_len'])
    loss = row['loss']
    if pl not in res: res[pl] = {}
    res[pl][loss] = {
        'mse': fmt(row['mse_mean'], row['mse_std']),
        'mae': fmt(row['mae_mean'], row['mae_std'])
    }

avg_df = df.groupby(['loss', 'seed']).agg({'mse': 'mean', 'mae': 'mean'}).reset_index()
avg_grouped = avg_df.groupby('loss').agg({'mse': ['mean', 'std'], 'mae': ['mean', 'std']}).reset_index()
avg_grouped.columns = ['loss', 'mse_mean', 'mse_std', 'mae_mean', 'mae_std']

avg_res = {}
for _, row in avg_grouped.iterrows():
    avg_res[row['loss']] = {
        'mse': fmt(row['mse_mean'], row['mse_std']),
        'mae': fmt(row['mae_mean'], row['mae_std'])
    }

def get_val(pl, loss_type, metric):
    return res.get(pl, {}).get(loss_type, {}).get(metric, '$-_{\pm -}$')

def get_avg(loss_type, metric):
    return avg_res.get(loss_type, {}).get(metric, '$-_{\pm -}$')

latex_table = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{Experimental results ($\\text{{mean}}_{{\\pm \\text{{std}}}}$) with varying seeds (2020-2026) on ecl dataset.}}\\label{{tab:varying_seeds_ecl}}
\\renewcommand{{\\arraystretch}}{{1.2}} \\setlength{{\\tabcolsep}}{{12pt}}
\\begin{{tabular}}{{c | cc | cc}}
    \\toprule
    \\multirow{{2}}{{*}}{{Models}} & \\multicolumn{{2}}{{c|}}{{\\textbf{{CvLoss}}}} & \\multicolumn{{2}}{{c}}{{TQNet}} \\\\
    \\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}}
    Metrics & MSE & MAE & MSE & MAE \\\\
    \\midrule
    96  & {get_val(96, 'fcv', 'mse')} & {get_val(96, 'fcv', 'mae')} & {get_val(96, 'None', 'mse')} & {get_val(96, 'None', 'mae')} \\\\
    192 & {get_val(192, 'fcv', 'mse')} & {get_val(192, 'fcv', 'mae')} & {get_val(192, 'None', 'mse')} & {get_val(192, 'None', 'mae')} \\\\
    336 & {get_val(336, 'fcv', 'mse')} & {get_val(336, 'fcv', 'mae')} & {get_val(336, 'None', 'mse')} & {get_val(336, 'None', 'mae')} \\\\
    720 & {get_val(720, 'fcv', 'mse')} & {get_val(720, 'fcv', 'mae')} & {get_val(720, 'None', 'mse')} & {get_val(720, 'None', 'mae')} \\\\
    \\midrule
    Avg & {get_avg('fcv', 'mse')} & {get_avg('fcv', 'mae')} & {get_avg('None', 'mse')} & {get_avg('None', 'mae')} \\\\
    \\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
print(latex_table)
