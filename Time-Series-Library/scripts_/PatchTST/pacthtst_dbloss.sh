#!/bin/bash

# ==========================================================
# 1. Configuration parsing
# ==========================================================
GPUS="0,1,2,3,4,5,6"        # Default GPUs (can be multiple, e.g., "0,1,2,3")
MAX_JOBS=7      # Default maximum parallel jobs

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpus) GPUS="$2"; shift 2;;
        --max_jobs) MAX_JOBS="$2"; shift 2;;
        *) echo "Unknown parameter: $1"; exit 1;;
    esac
done

# Convert comma-separated GPU list to an array
IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "=========================================================="
echo "Using GPU list: [${GPU_ARRAY[@]}] ($NUM_GPUS available)"
echo "Max parallel jobs: $MAX_JOBS"
echo "=========================================================="

# Create logs directory
mkdir -p logs

# ==========================================================
# 2. Parallel Task Scheduler
# ==========================================================
running_pids=()
task_count=0

# Function: Wait until there is an available slot
wait_for_slot() {
    while [ ${#running_pids[@]} -ge $MAX_JOBS ]; do
        sleep 1 # Poll every second to save CPU
        local new_pids=()
        for pid in "${running_pids[@]}"; do
            # kill -0 checks if the process is still running
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
            fi
        done
        running_pids=("${new_pids[@]}")
    done
}

# ==========================================================
# 3. DBLoss Arguments Generator
# ==========================================================
get_dbloss_args() {
  local dataset_tag=$1
  local pred_len=$2
  
  local alpha=0.2
  local beta=0.5

  case "${dataset_tag}" in
    ETTh1)
      beta=0.9
      ;;
    ETTh2|ETTm1|ETTm2)
      beta=0.5
      ;;
    ECL|Traffic)
      beta=0.6
      ;;
    Weather)
      if [[ "${pred_len}" == "192" ]]; then 
        beta=0.5
      else 
        beta=0.2
      fi
      ;;
  esac

  echo "--alpha_DBLoss ${alpha} --beta_DBLoss ${beta} --loss DBLoss"
}

# ==========================================================
# 4. Core Execution Function
# ==========================================================
run_model() {
    wait_for_slot # Wait for an available parallel slot

    # Round-Robin GPU allocation
    local gpu_idx=$(( task_count % NUM_GPUS ))
    local current_gpu=${GPU_ARRAY[$gpu_idx]}
    task_count=$(( task_count + 1 ))

    # Extract model_id for log naming
    local model_id_val="task_$task_count"
    local args=("$@")
    for (( i=0; i<${#args[@]}; i++ )); do
        if [[ "${args[$i]}" == "--model_id" ]]; then
            model_id_val="${args[$((i+1))]}"
            break
        fi
    done

    echo "[$(date +'%H:%M:%S')] Starting task: Model ID [ $model_id_val ] => Assigned to GPU: $current_gpu"

    # Set environment variables and run python in background
    CUDA_VISIBLE_DEVICES=$current_gpu python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model PatchTST \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --d_layers 1 \
        --factor 3 \
        --des 'Exp' \
        --itr 1 \
        "$@" > "logs/${model_id_val}.log" 2>&1 &
        
    # Store background process PID for monitoring
    running_pids+=($!)
}

# ==========================================================
# 5. Submit Task Queue
# ==========================================================

# ----------------- ETTh1 -----------------
run_etth1() {
    local pred_len=$1; local e_layers=$2; local n_heads=$3
    local db_args=$(get_dbloss_args ETTh1 ${pred_len})
    run_model \
        --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --data ETTh1 \
        --model_id ETTh1_96_${pred_len} --pred_len ${pred_len} \
        --e_layers ${e_layers} --n_heads ${n_heads} \
        --enc_in 7 --dec_in 7 --c_out 7 ${db_args}
}
run_etth1 96 1 2
run_etth1 192 1 8
run_etth1 336 1 8
run_etth1 720 1 16

# ----------------- ETTh2 -----------------
for pred_len in 96 192 336 720; do
    db_args=$(get_dbloss_args ETTh2 ${pred_len})
    run_model \
        --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --data ETTh2 \
        --model_id ETTh2_96_${pred_len} --pred_len ${pred_len} \
        --e_layers 3 --n_heads 4 \
        --enc_in 7 --dec_in 7 --c_out 7 ${db_args}
done

# ----------------- ETTm1 -----------------
run_ettm1() {
    local pred_len=$1; local e_layers=$2; local n_heads=$3; local batch_size=$4
    local db_args=$(get_dbloss_args ETTm1 ${pred_len})
    run_model \
        --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --data ETTm1 \
        --model_id ETTm1_96_${pred_len} --pred_len ${pred_len} \
        --e_layers ${e_layers} --n_heads ${n_heads} --batch_size ${batch_size} \
        --enc_in 7 --dec_in 7 --c_out 7 ${db_args}
}
run_ettm1 96 1 2 32
run_ettm1 192 3 2 128
run_ettm1 336 1 4 128
run_ettm1 720 3 4 128

# ----------------- ETTm2 -----------------
run_ettm2() {
    local pred_len=$1; local e_layers=$2; local n_heads=$3; local batch_size=$4
    local db_args=$(get_dbloss_args ETTm2 ${pred_len})
    run_model \
        --root_path ./dataset/ETT-small/ --data_path ETTm2.csv --data ETTm2 \
        --model_id ETTm2_96_${pred_len} --pred_len ${pred_len} \
        --e_layers ${e_layers} --n_heads ${n_heads} --batch_size ${batch_size} \
        --enc_in 7 --dec_in 7 --c_out 7 ${db_args}
}
run_ettm2 96 3 16 32
run_ettm2 192 3 2 128
run_ettm2 336 1 4 32
run_ettm2 720 3 4 128

# ----------------- Weather -----------------
run_weather() {
    local pred_len=$1; local n_heads=$2
    shift 2
    local db_args=$(get_dbloss_args Weather ${pred_len})
    run_model \
        --root_path ./dataset/weather/ --data_path weather.csv --data custom \
        --model_id weather_96_${pred_len} --pred_len ${pred_len} \
        --e_layers 2 --n_heads ${n_heads} --train_epochs 3 \
        --enc_in 21 --dec_in 21 --c_out 21 ${db_args} "$@"
}
run_weather 96 4
run_weather 192 16
run_weather 336 4 --batch_size 128
run_weather 720 4 --batch_size 128

# ----------------- Electricity -----------------
for pred_len in 96 192 336 720; do
    db_args=$(get_dbloss_args ECL ${pred_len})
    run_model \
        --root_path ./dataset/electricity/ --data_path electricity.csv --data custom \
        --model_id ECL_96_${pred_len} --pred_len ${pred_len} \
        --e_layers 2 --batch_size 16 \
        --enc_in 321 --dec_in 321 --c_out 321 ${db_args}
done

# ----------------- Traffic -----------------
for pred_len in 96 192 336 720; do
    db_args=$(get_dbloss_args Traffic ${pred_len})
    run_model \
        --root_path ./dataset/traffic/ --data_path traffic.csv --data custom \
        --model_id traffic_96_${pred_len} --pred_len ${pred_len} \
        --e_layers 2 --batch_size 4 --top_k 5 \
        --enc_in 862 --dec_in 862 --c_out 862 \
        --d_model 512 --d_ff 512 ${db_args}
done

# ==========================================================
# 6. Wait and Cleanup
# ==========================================================
echo "All tasks pushed to queue, waiting for completion..."
wait
echo "🎉 All experiments finished! Logs are saved in the logs/ directory."