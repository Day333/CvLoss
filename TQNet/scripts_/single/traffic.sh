#!/usr/bin/env bash
set -e

GPU_ID=3

configs=(
  "96 3 0.5"
  "192 3 0.5"
  "336 3 0.5"
  "720 3 0.1"
)

for config in "${configs[@]}"; do
    read -r pred_len patchlen beta <<< "$config"
    
    alpha_add=$(awk "BEGIN {print 1.0 - $beta}")
    
    echo "======================================================"
    echo "Running Traffic: pred_len=${pred_len}, patchlen=${patchlen}, beta=${beta}"
    echo "======================================================"

    CUDA_VISIBLE_DEVICES=$GPU_ID python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path traffic.csv \
        --model_id "traffic_96_${pred_len}_fcv_patch${patchlen}_b${beta}" \
        --model TQNet \
        --data custom \
        --features M \
        --seq_len 96 \
        --pred_len ${pred_len} \
        --enc_in 862 \
        --cycle 168 \
        --train_epochs 30 \
        --patience 5 \
        --itr 1 \
        --batch_size 16 \
        --learning_rate 0.003 \
        --random_seed 2024 \
        --add_loss fcv \
        --loss_patchlen ${patchlen} \
        --alpha_add_loss ${alpha_add} \
        --beta_add_loss ${beta}
done

echo "All Traffic best parameter jobs finished."