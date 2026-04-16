#!/usr/bin/env bash
set -e

########################################
# 1. CONFIG
########################################

MAX_JOBS=2
AVAILABLE_GPUS=(5 6)
MAX_RETRIES=0
NUM_GPUS=${#AVAILABLE_GPUS[@]}

model_name="PatchTST"
seq_len=96
label_len=48
factor=3
itr=1

# datasets=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "Weather" "ECL" "Traffic")
datasets=("Traffic")

pred_lens=(96 192 336 720)
patchlens=(3)
betas=(0.5 0.6 0.7 0.8 0.9 1.0)

mkdir -p logs
gpu_ptr=0

########################################
# 2. SEMAPHORE
########################################

SEMAPHORE=/tmp/gs_semaphore_patchtst_fcv_search
mkfifo $SEMAPHORE
exec 9<>$SEMAPHORE
rm $SEMAPHORE

for ((i=0;i<${MAX_JOBS};i++)); do
  echo >&9
done

########################################
# 3. FUNCTIONS
########################################

run_job() {
  local gpu_id=$1
  local cmd=$2
  local log_file=$3
  local model_id=$4
  local attempt=0

  while (( attempt <= MAX_RETRIES )); do
    echo "▶ [GPU $gpu_id][Try $((attempt+1))] $model_id"
    
    if eval "CUDA_VISIBLE_DEVICES=$gpu_id $cmd > \"$log_file\" 2>&1"; then
      echo "✅ [GPU $gpu_id] Success: $model_id"
      break
    else
      echo "❌ [GPU $gpu_id] Failed: $model_id (Attempt $((attempt+1)))"
      attempt=$((attempt + 1))
      if (( attempt > MAX_RETRIES )); then
        echo "[$model_id] $cmd" >> failures.txt
      fi
    fi
  done

  echo >&9
}

is_finished() {
  local log_file="$1"
  grep -Eq 'mse:[[:space:]]*[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?,[[:space:]]*mae:[[:space:]]*[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?' "$log_file"
}

calc_alpha() {
  local beta="$1"
  python -c "print(f'{1.0 - float(\"$beta\"):.6f}'.rstrip('0').rstrip('.'))"
}

########################################
# 4. MAIN LOOP
########################################

echo "🚀 Starting PatchTST FCV hyperparameter search..."

for dataset in "${datasets[@]}"; do
  case "${dataset}" in
    ETTh1) root_path="./dataset/ETT-small/"; data_path="ETTh1.csv"; data_name="ETTh1"; enc_in=7;;
    ETTh2) root_path="./dataset/ETT-small/"; data_path="ETTh2.csv"; data_name="ETTh2"; enc_in=7;;
    ETTm1) root_path="./dataset/ETT-small/"; data_path="ETTm1.csv"; data_name="ETTm1"; enc_in=7;;
    ETTm2) root_path="./dataset/ETT-small/"; data_path="ETTm2.csv"; data_name="ETTm2"; enc_in=7;;
    Weather) root_path="./dataset/weather/"; data_path="weather.csv"; data_name="custom"; enc_in=21;;
    ECL) root_path="./dataset/electricity/"; data_path="electricity.csv"; data_name="custom"; enc_in=321;;
    Traffic) root_path="./dataset/traffic/"; data_path="traffic.csv"; data_name="custom"; enc_in=862;;
  esac

  for pred_len in "${pred_lens[@]}"; do
    
    e_layers=1; d_layers=1; n_heads=8; batch_size=32; extra_args=""
    
    if [ "$dataset" == "ETTh1" ]; then
        if [ "$pred_len" == "96" ]; then n_heads=2
        elif [ "$pred_len" == "192" ] || [ "$pred_len" == "336" ]; then n_heads=8
        elif [ "$pred_len" == "720" ]; then n_heads=16; fi
    elif [ "$dataset" == "ETTh2" ]; then
        e_layers=3; n_heads=4
    elif [ "$dataset" == "ETTm1" ]; then
        if [ "$pred_len" == "96" ]; then n_heads=2; batch_size=32
        elif [ "$pred_len" == "192" ]; then e_layers=3; n_heads=2; batch_size=128
        elif [ "$pred_len" == "336" ]; then n_heads=4; batch_size=128
        elif [ "$pred_len" == "720" ]; then e_layers=3; n_heads=4; batch_size=128; fi
    elif [ "$dataset" == "ETTm2" ]; then
        if [ "$pred_len" == "96" ]; then e_layers=3; n_heads=16; batch_size=32
        elif [ "$pred_len" == "192" ]; then e_layers=3; n_heads=2; batch_size=128
        elif [ "$pred_len" == "336" ]; then n_heads=4; batch_size=32
        elif [ "$pred_len" == "720" ]; then e_layers=3; n_heads=4; batch_size=128; fi
    elif [ "$dataset" == "Weather" ]; then
        e_layers=2; extra_args="--train_epochs 3"
        if [ "$pred_len" == "96" ]; then n_heads=4
        elif [ "$pred_len" == "192" ]; then n_heads=16
        elif [ "$pred_len" == "336" ] || [ "$pred_len" == "720" ]; then n_heads=4; batch_size=128; fi
    elif [ "$dataset" == "ECL" ]; then
        e_layers=2; batch_size=16
    elif [ "$dataset" == "Traffic" ]; then
        e_layers=2; batch_size=4; extra_args="--top_k 5 --d_model 512 --d_ff 512"
    fi

    for loss_patchlen in "${patchlens[@]}"; do
      for beta_add_loss in "${betas[@]}"; do
        
        read -u9 

        alpha_add_loss=$(calc_alpha "${beta_add_loss}")
        model_id="${dataset}_96_${pred_len}_fcv_p${loss_patchlen}_b${beta_add_loss}"
        log_file="logs/${model_name}_${model_id}.log"

        if [ -f "$log_file" ] && is_finished "$log_file"; then
          echo "⏭ Skip: $model_id"
          echo >&9 
          continue
        fi

        gpu_id=${AVAILABLE_GPUS[$gpu_ptr]}
        gpu_ptr=$(( (gpu_ptr + 1) % NUM_GPUS ))

        cmd="python -u run.py \
          --task_name long_term_forecast \
          --is_training 1 \
          --root_path \"${root_path}\" \
          --data_path \"${data_path}\" \
          --model_id \"${model_id}\" \
          --model \"${model_name}\" \
          --data \"${data_name}\" \
          --features M \
          --seq_len ${seq_len} \
          --label_len ${label_len} \
          --pred_len ${pred_len} \
          --e_layers ${e_layers} \
          --d_layers ${d_layers} \
          --factor ${factor} \
          --enc_in ${enc_in} \
          --dec_in ${enc_in} \
          --c_out ${enc_in} \
          --n_heads ${n_heads} \
          --batch_size ${batch_size} \
          --des Exp \
          --itr ${itr} \
          --add_loss fcv \
          --loss_patchlen ${loss_patchlen} \
          --alpha_add_loss ${alpha_add_loss} \
          --beta_add_loss ${beta_add_loss} \
          ${extra_args}"

        run_job $gpu_id "$cmd" "$log_file" "$model_id" &
        
      done
    done
  done
done

########################################
# 5. WAIT AND CLEANUP
########################################

wait
echo "🎉 All PatchTST FCV search jobs finished!"