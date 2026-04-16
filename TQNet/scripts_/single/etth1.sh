#!/usr/bin/env bash
set -e

GPU_ID=4

configs=(
  "96 48 0.8"
  "192 48 0.2"
  "336 3 0.8"
  "720 3 0.7"
)

for config in "${configs[@]}"; do
    read -r pred_len patchlen beta <<< "$config"
    
    alpha_add=$(awk "BEGIN {print 1.0 - $beta}")
    
    echo "======================================================"
    echo "Running ETTh1: pred_len=${pred_len}, patchlen=${patchlen}, beta=${beta}"
    echo "======================================================"

    CUDA_VISIBLE_DEVICES=$GPU_ID python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path ETTh1.csv \
        --model_id "ETTh1_96_${pred_len}_fcv_patch${patchlen}_b${beta}" \
        --model TQNet \
        --data ETTh1 \
        --features M \
        --seq_len 96 \
        --pred_len ${pred_len} \
        --enc_in 7 \
        --cycle 24 \
        --train_epochs 30 \
        --patience 5 \
        --dropout 0.5 \
        --itr 1 \
        --batch_size 256 \
        --learning_rate 0.001 \
        --random_seed 2024 \
        --add_loss fcv \
        --loss_patchlen ${patchlen} \
        --alpha_add_loss ${alpha_add} \
        --beta_add_loss ${beta}
done

echo "All best parameter jobs finished."