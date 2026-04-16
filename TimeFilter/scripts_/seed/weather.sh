#!/usr/bin/env bash
set -e

GPU_ID=3
model_name=TimeFilter
patchlen=6

seq_lens=(96)
pred_lens=(96 192 336 720)
loss_modes=("fcv" "None")
# 1. 加入 seeds 列表
seeds=(2020 2021 2022 2023 2024 2025 2026)

# 创建记录文件夹
LOG_DIR=logs/weather_timefilter_grid
mkdir -p "${LOG_DIR}"

# 2. 初始化结果 CSV 文件，并加入 seed 列
RESULT_CSV="weather_timefilter_seeds_results.csv"
echo "seq_len,pred_len,loss,seed,mse,mae" > "${RESULT_CSV}"

for seq_len in "${seq_lens[@]}"; do
  for pred_len in "${pred_lens[@]}"; do

    # 根据预测长度自动设置 beta
    case ${pred_len} in
      96)  beta=1.0 ;;
      192) beta=1.0 ;;
      336) beta=0.5 ;;
      720) beta=1.0 ;;
      *)   echo "Unsupported pred_len=${pred_len}"; exit 1 ;;
    esac

    # 计算 alpha
    alpha_add=$(awk "BEGIN {print 1.0 - $beta}")

    for add_loss in "${loss_modes[@]}"; do
      # 3. 增加 seed 循环
      for seed in "${seeds[@]}"; do
      
        if [[ "${add_loss}" == "fcv" ]]; then
          table_model_name="CvLoss"
          extra_args=(
            --add_loss fcv
            --loss_patchlen "${patchlen}"
            --alpha_add_loss "${alpha_add}"
            --beta_add_loss "${beta}"
          )
        else
          table_model_name="TimeFilter"
          extra_args=(
            --add_loss None
          )
        fi

        # 4. 在 model_id 中加上 seed 以区分不同日志文件
        model_id="weather_${seq_len}_${pred_len}_${add_loss}_patch${patchlen}_b${beta}_seed${seed}"
        log_file="${LOG_DIR}/${model_id}.log"

        echo "======================================================"
        echo "Running | model=${model_name} | table_name=${table_model_name} | seq_len=${seq_len} | pred_len=${pred_len} | add_loss=${add_loss} | seed=${seed}"
        echo "======================================================"

        # 运行模型并将输出保存到 log_file
        # 5. 传入 --random_seed 参数
        CUDA_VISIBLE_DEVICES=${GPU_ID} python -u run.py \
          --task_name long_term_forecast \
          --is_training 2 \
          --root_path ./data \
          --data_path weather.csv \
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
          --enc_in 21 \
          --dec_in 21 \
          --c_out 21 \
          --patch_len 48 \
          --des Exp \
          --d_model 128 \
          --d_ff 256 \
          --dropout 0.3 \
          --learning_rate 0.0005 \
          --batch_size 32 \
          --random_seed "${seed}" \
          --itr 1 \
          "${extra_args[@]}" 2>&1 | tee "${log_file}"

        # 内嵌 Python 正则提取测试集 mse 和 mae
        METRICS=$(python3 -c "
import re
try:
    with open('${log_file}', 'r') as f:
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
        
        # 6. 将结果写入 CSV，补充 seed 字段 (seq_len, pred_len, loss, seed, mse, mae)
        echo "${seq_len},${pred_len},${add_loss},${seed},${MSE},${MAE}" >> "${RESULT_CSV}"
        
      done
    done
  done
done

echo "======================================================"
echo "All experiments finished! Generating LaTeX Table..."
echo "======================================================"

# 生成用于计算均值/方差并打印 LaTeX 表格的 Python 脚本
cat << EOF > generate_latex_table.py
import pandas as pd
import numpy as np

# 修复了读取的文件名，使其直接读取本脚本生成的 CSV 文件
df = pd.read_csv('${RESULT_CSV}', keep_default_na=False)

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
    return res.get(pl, {}).get(loss_type, {}).get(metric, '$-_{\\pm -}$')

def get_avg(loss_type, metric):
    return avg_res.get(loss_type, {}).get(metric, '$-_{\\pm -}$')

latex_table = f"""
\\\\begin{{table}}[htbp]
\\\\centering
\\\\caption{{Experimental results ($\\\\text{{mean}}_{{\\\\pm \\\\text{{std}}}}$) with varying seeds (2020-2026) on Weather dataset.}}\\\\label{{tab:varying_seeds_weather}}
\\\\renewcommand{{\\\\arraystretch}}{{1.2}} \\\\setlength{{\\\\tabcolsep}}{{12pt}}
\\\\begin{{tabular}}{{c | cc | cc}}
    \\\\toprule
    \\\\multirow{{2}}{{*}}{{Models}} & \\\\multicolumn{{2}}{{c|}}{{\\\\textbf{{CvLoss}}}} & \\\\multicolumn{{2}}{{c}}{{TimeFilter}} \\\\\\\\
    \\\\cmidrule(lr){{2-3}} \\\\cmidrule(lr){{4-5}}
    Metrics & MSE & MAE & MSE & MAE \\\\\\\\
    \\\\midrule
    96  & {get_val(96, 'fcv', 'mse')} & {get_val(96, 'fcv', 'mae')} & {get_val(96, 'None', 'mse')} & {get_val(96, 'None', 'mae')} \\\\\\\\
    192 & {get_val(192, 'fcv', 'mse')} & {get_val(192, 'fcv', 'mae')} & {get_val(192, 'None', 'mse')} & {get_val(192, 'None', 'mae')} \\\\\\\\
    336 & {get_val(336, 'fcv', 'mse')} & {get_val(336, 'fcv', 'mae')} & {get_val(336, 'None', 'mse')} & {get_val(336, 'None', 'mae')} \\\\\\\\
    720 & {get_val(720, 'fcv', 'mse')} & {get_val(720, 'fcv', 'mae')} & {get_val(720, 'None', 'mse')} & {get_val(720, 'None', 'mae')} \\\\\\\\
    \\\\midrule
    Avg & {get_avg('fcv', 'mse')} & {get_avg('fcv', 'mae')} & {get_avg('None', 'mse')} & {get_avg('None', 'mae')} \\\\\\\\
    \\\\bottomrule
\\\\end{{tabular}}
\\\\end{{table}}
"""
print(latex_table)
EOF

python3 generate_latex_table.py