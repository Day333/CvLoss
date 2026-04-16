#!/usr/bin/env bash
set -euo pipefail

GPU_ID=3
model_name=TimeFilter
patchlen=6

seq_lens=(96 192 336 720)
pred_lens=(96 192 336 720)
loss_modes=("fcv" "None")

LOG_DIR=logs/weather_timefilter_grid
mkdir -p "${LOG_DIR}"

RESULT_CSV="${LOG_DIR}/weather_timefilter_results.csv"
echo "model_name,add_loss,seq_len,pred_len,mse,mae,log_file" > "${RESULT_CSV}"

for seq_len in "${seq_lens[@]}"; do
  for pred_len in "${pred_lens[@]}"; do

    case ${pred_len} in
      96)  beta=1.0 ;;
      192) beta=1.0 ;;
      336) beta=0.5 ;;
      720) beta=1.0 ;;
      *)   echo "Unsupported pred_len=${pred_len}"; exit 1 ;;
    esac

    alpha_add=$(python - <<PY
b=float("${beta}")
a=1.0-b
print(f"{a:.6f}".rstrip('0').rstrip('.'))
PY
)

    for add_loss in "${loss_modes[@]}"; do
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

      model_id="weather_${seq_len}_${pred_len}_${add_loss}_patch${patchlen}_b${beta}"
      log_file="${LOG_DIR}/${model_id}.log"

      echo "======================================================"
      echo "Running | model=${model_name} | table_name=${table_model_name} | seq_len=${seq_len} | pred_len=${pred_len} | add_loss=${add_loss}"
      echo "======================================================"

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
        --itr 1 \
        "${extra_args[@]}" 2>&1 | tee "${log_file}"

      python extract_metrics.py \
        --log_file "${log_file}" \
        --csv_file "${RESULT_CSV}" \
        --model_name "${table_model_name}" \
        --add_loss "${add_loss}" \
        --seq_len "${seq_len}" \
        --pred_len "${pred_len}"
    done
  done
done

echo "All jobs finished."
echo "Collected results saved to: ${RESULT_CSV}"

python summarize_results.py --csv_file "${RESULT_CSV}"