#!/usr/bin/env bash
set -euo pipefail

GPU_ID=4

seq_lens=(96 192 336 720)

configs=(
  "96 3 1.0"
  "192 3 1.0"
  "336 3 0.5"
  "720 3 0.5"
)

# fcv   -> CvLoss
# None  -> TQNet
loss_modes=("fcv" "None")

LOG_DIR=logs/weather_grid
mkdir -p "${LOG_DIR}"

RESULT_CSV="${LOG_DIR}/weather_results.csv"
echo "model_name,add_loss,seq_len,pred_len,mse,mae,log_file" > "${RESULT_CSV}"

for seq_len in "${seq_lens[@]}"; do
  for config in "${configs[@]}"; do
    read -r pred_len patchlen beta <<< "$config"
    alpha_add=$(awk "BEGIN {print 1.0 - $beta}")

    for add_loss in "${loss_modes[@]}"; do
      if [[ "${add_loss}" == "fcv" ]]; then
        model_name="CvLoss"
        extra_args=(
          --add_loss fcv
          --loss_patchlen "${patchlen}"
          --alpha_add_loss "${alpha_add}"
          --beta_add_loss "${beta}"
        )
      else
        model_name="TQNet"
        extra_args=(
          --add_loss None
        )
      fi

      model_id="weather_${seq_len}_${pred_len}_${add_loss}_patch${patchlen}_b${beta}"
      log_file="${LOG_DIR}/${model_id}.log"

      echo "======================================================"
      echo "Running Weather | model=${model_name} | seq_len=${seq_len} | pred_len=${pred_len} | patchlen=${patchlen} | beta=${beta}"
      echo "======================================================"

      CUDA_VISIBLE_DEVICES=$GPU_ID python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path weather.csv \
        --model_id "${model_id}" \
        --model TQNet \
        --data custom \
        --features M \
        --seq_len "${seq_len}" \
        --pred_len "${pred_len}" \
        --enc_in 21 \
        --cycle 144 \
        --train_epochs 30 \
        --patience 5 \
        --dropout 0.5 \
        --itr 1 \
        --batch_size 64 \
        --learning_rate 0.001 \
        --random_seed 2024 \
        "${extra_args[@]}" 2>&1 | tee "${log_file}"

      python extract_metrics.py \
        --log_file "${log_file}" \
        --csv_file "${RESULT_CSV}" \
        --model_name "${model_name}" \
        --add_loss "${add_loss}" \
        --seq_len "${seq_len}" \
        --pred_len "${pred_len}"
    done
  done
done

echo "All Weather jobs finished."
echo "Collected results saved to: ${RESULT_CSV}"

python summarize_results.py --csv_file "${RESULT_CSV}"