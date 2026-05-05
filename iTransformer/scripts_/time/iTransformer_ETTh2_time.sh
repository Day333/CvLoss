#!/usr/bin/env bash
set -e

MAX_JOBS=1
TOTAL_GPUS=1
MAX_RETRIES=0

TEST_EPOCHS=10

SEMAPHORE=/tmp/gs_semaphore
mkfifo $SEMAPHORE
exec 9<>$SEMAPHORE
rm $SEMAPHORE
for ((i=0;i<${MAX_JOBS};i++)); do echo >&9; done

run_job() {
  local gpu_id=$1
  local cmd=$2
  local log_file=$3
  local model_id=$4
  local attempt=0

  while (( attempt <= MAX_RETRIES )); do
    echo "▶ [GPU $gpu_id][Try $((attempt+1))] $model_id"
    CUDA_VISIBLE_DEVICES=$gpu_id $cmd > "$log_file" 2>&1

    if [ $? -eq 0 ]; then
      echo "✅ [GPU $gpu_id] Success: $model_id"
      break
    else
      echo "❌ [GPU $gpu_id] Failed: $model_id (Attempt $((attempt+1)))"
      attempt=$((attempt + 1))
      if (( attempt > MAX_RETRIES )); then
        echo "$cmd" >> failures.txt
      fi
    fi
  done
  echo >&9
}

model_name=iTransformer
seq_len=96
pred_lens=(1024 720 512 336 256 192 128 96 64 48 32)

loss_modes=("fcv" "None")

root_path=./dataset/ETT-small/
data_path=ETTh2.csv

job_idx=0
mkdir -p logs_speed

CSV_FILE="speed_results.csv"
echo "T,loss_type,Forward_ms,Backward_ms" > "$CSV_FILE"

for pred_len in "${pred_lens[@]}"; do
  
  if (( pred_len >= 336 )); then
    patchlen=4
    beta=0.5
  else
    patchlen=4
    beta=1.0
  fi

  alpha=$(awk "BEGIN {print 1.0 - $beta}")

  for loss_type in "${loss_modes[@]}"; do

    read -u9  # semaphore token
    
    model_id="Speed_${seq_len}_${pred_len}_${loss_type}"
    log_file="logs_speed/${model_id}.log"

    {
      gpu_id=$((job_idx % TOTAL_GPUS))
      job_idx=$((job_idx + 1))

      cmd="python -u run.py \
        --is_training 1 \
        --root_path ${root_path} \
        --data_path ${data_path} \
        --model_id ${model_id} \
        --model ${model_name} \
        --data ETTh2 \
        --features M \
        --seq_len ${seq_len} \
        --pred_len ${pred_len} \
        --e_layers 2 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --d_model 128 \
        --d_ff 128 \
        --itr 1 \
        --train_epochs ${TEST_EPOCHS} \
        --add_loss ${loss_type} \
        --loss_patchlen ${patchlen} \
        --alpha_add_loss ${alpha} \
        --beta_add_loss ${beta}"

      run_job $gpu_id "$cmd" "$log_file" "$model_id"
      
      FWD=$(grep -oP 'Forward_Phase_Avg_ms:\s*\K[0-9.]+' "$log_file" | awk 'NR>1 {sum+=$1; cnt++} END {if(cnt>0) printf "%.4f", sum/cnt; else print "NaN"}')
      BWD=$(grep -oP 'Backward_Phase_Avg_ms:\s*\K[0-9.]+' "$log_file" | awk 'NR>1 {sum+=$1; cnt++} END {if(cnt>0) printf "%.4f", sum/cnt; else print "NaN"}')
      
      FWD=${FWD:-NaN}
      BWD=${BWD:-NaN}
      
      echo "${pred_len},${loss_type},${FWD},${BWD}" >> "$CSV_FILE"
      
    } &
  done
done

wait

echo "====================================================="
echo "All timing tasks finished. Tested ${TEST_EPOCHS} epochs and skipped warmup epoch 1."
echo "Data has been extracted to $CSV_FILE"
echo "====================================================="
cat "$CSV_FILE"
