#!/usr/bin/env bash
set -e

GPU_ID=4

# Prediction lengths with patchlen and beta.
configs=(
  "96 3 0.5"
  "192 3 0.5"
  "336 3 0.5"
  "720 3 0.2"
)

# Seeds and loss types.
seeds=(2020 2021 2022 2023 2024 2025 2026)
losses=("fcv" "None")

# Create result CSV.
CSV_FILE="electricity_results_summary.csv"
echo "pred_len,loss,seed,mse,mae" > $CSV_FILE

for loss in "${losses[@]}"; do
  for seed in "${seeds[@]}"; do
    for config in "${configs[@]}"; do
        read -r pred_len patchlen beta <<< "$config"
        
        alpha_add=$(awk "BEGIN {print 1.0 - $beta}")
        
        echo "======================================================"
        echo "Running Electricity: pred_len=${pred_len}, loss=${loss}, seed=${seed}, patchlen=${patchlen}, beta=${beta}"
        echo "======================================================"

        # Use a temp log.
        TMP_LOG="tmp_run_elec.log"

        CUDA_VISIBLE_DEVICES=$GPU_ID python -u run.py \
            --is_training 1 \
            --root_path ./dataset/ \
            --data_path electricity.csv \
            --model_id "Elec_96_${pred_len}_${loss}_patch${patchlen}_b${beta}_seed${seed}" \
            --model TQNet \
            --data custom \
            --features M \
            --seq_len 96 \
            --pred_len ${pred_len} \
            --enc_in 321 \
            --cycle 168 \
            --train_epochs 30 \
            --patience 5 \
            --itr 1 \
            --batch_size 32 \
            --learning_rate 0.003 \
            --random_seed ${seed} \
            --add_loss ${loss} \
            --loss_patchlen ${patchlen} \
            --alpha_add_loss ${alpha_add} \
            --beta_add_loss ${beta} | tee $TMP_LOG

        # Extract test MSE and MAE with Python regex.
        METRICS=$(python3 -c "
import re
import sys
try:
    with open('$TMP_LOG', 'r') as f:
        log = f.read().lower()
    matches = re.findall(r'mse\s*[:=]?\s*([0-9\.]+).*?mae\s*[:=]?\s*([0-9\.]+)', log)
    if matches:
        print(f'{matches[-1][0]},{matches[-1][1]}')
    else:
        print('NaN,NaN')
except Exception as e:
    print('NaN,NaN')
")
        
        MSE=$(echo $METRICS | cut -d',' -f1)
        MAE=$(echo $METRICS | cut -d',' -f2)
        
        echo "Extracted -> MSE: $MSE, MAE: $MAE"
        
        # Write CSV row.
        echo "${pred_len},${loss},${seed},${MSE},${MAE}" >> $CSV_FILE
        
        # Remove temp log.
        rm -f $TMP_LOG
    done
  done
done

cat << 'EOF' > generate_latex_table.py
import pandas as pd
import numpy as np

df = pd.read_csv('ecl_results_summary.csv', keep_default_na=False)

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
EOF

python3 generate_latex_table.py
