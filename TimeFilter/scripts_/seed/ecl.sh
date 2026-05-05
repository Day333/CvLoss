#!/usr/bin/env bash
set -e

GPU_ID=2
model_name=TimeFilter
patchlen=6

# Define test configurations.
pred_lens=(96 192 336 720)
seeds=(2020 2021 2022 2023 2024 2025 2026)
loss_modes=("fcv" "None")

# Create result CSV.
CSV_FILE="ecl_timefilter_seeds_results.csv"
echo "pred_len,loss,seed,mse,mae" > $CSV_FILE

seq_len=96

for loss in "${loss_modes[@]}"; do
  for seed in "${seeds[@]}"; do
    for pred_len in "${pred_lens[@]}"; do
        
        # Set beta and dropout by pred_len.
        if [[ "$pred_len" == "96" ]]; then
            beta=0.5
            dropout=0.5
        elif [[ "$pred_len" == "192" ]]; then
            beta=0.5
            dropout=0.4
        elif [[ "$pred_len" == "336" ]]; then
            beta=0.5
            dropout=0.4
        elif [[ "$pred_len" == "720" ]]; then
            beta=0.001
            dropout=0.4
        else
            echo "Unsupported pred_len=${pred_len}"
            exit 1
        fi

        # Compute alpha.
        alpha_add=$(awk "BEGIN {print 1.0 - $beta}")
        
        echo "======================================================"
        echo "Running ECL TimeFilter: pred_len=${pred_len}, loss=${loss}, seed=${seed}, patchlen=${patchlen}, beta=${beta}"
        echo "======================================================"

        # Set extra args by loss type.
        if [[ "${loss}" == "fcv" ]]; then
            extra_args=(
                --add_loss fcv
                --loss_patchlen "${patchlen}"
                --alpha_add_loss "${alpha_add}"
                --beta_add_loss "${beta}"
            )
        else
            extra_args=(
                --add_loss None
            )
        fi

        # Use a temp log for live output and metrics.
        TMP_LOG="tmp_run_ecl.log"
        model_id="ECL_${seq_len}_${pred_len}_${loss}_patch${patchlen}_b${beta}_seed${seed}"

        # Run training.
        CUDA_VISIBLE_DEVICES=$GPU_ID python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path ./data \
            --data_path electricity.csv \
            --model_id "${model_id}" \
            --model "${model_name}" \
            --data custom \
            --features M \
            --seq_len "${seq_len}" \
            --label_len 48 \
            --pred_len "${pred_len}" \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --enc_in 321 \
            --dec_in 321 \
            --c_out 321 \
            --patch_len 32 \
            --des Exp \
            --learning_rate 0.001 \
            --batch_size 16 \
            --train_epochs 15 \
            --d_model 512 \
            --d_ff 512 \
            --dropout ${dropout} \
            --random_seed ${seed} \
            --itr 1 \
            "${extra_args[@]}" 2>&1 | tee $TMP_LOG

        # Extract test MSE and MAE with Python regex.
        METRICS=$(python3 -c "
import re
try:
    with open('$TMP_LOG', 'r') as f:
        log = f.read().lower()
    matches = re.findall(r'mse\s*[:=]?\s*([0-9\.]+).*?mae\s*[:=]?\s*([0-9\.]+)', log)
    if matches:
        print(f'{matches[-1][0]},{matches[-1][1]}')
    else:
        print('NaN,NaN')
except:
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

echo "======================================================"
echo "All experiments finished! Generating LaTeX Table..."
echo "======================================================"

# Generate the Python script for mean/std and LaTeX output.
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
    \\multirow{{2}}{{*}}{{Models}} & \\multicolumn{{2}}{{c|}}{{\\textbf{{CvLoss}}}} & \\multicolumn{{2}}{{c}}{{TimeFilter}} \\\\
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
