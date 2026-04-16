#!/usr/bin/env bash
set -e

GPU_ID=5


configs=(
  "12 6 0.1"
  "24 3 0.6"
  "48 3 0.6"
)

for config in "${configs[@]}"; do
    read -r pred_len patchlen beta <<< "$config"
    
    alpha_add=$(awk "BEGIN {print 1.0 - $beta}")
    
    echo "======================================================"
    echo "Running PEMS03: pred_len=${pred_len}, patchlen=${patchlen}, beta=${beta}"
    echo "======================================================"

    CUDA_VISIBLE_DEVICES=$GPU_ID python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path PEMS03.npz \
        --model_id "PEMS03_96_${pred_len}_fcv_patch${patchlen}_b${beta}" \
        --model TQNet \
        --data PEMS \
        --features M \
        --seq_len 96 \
        --pred_len ${pred_len} \
        --enc_in 358 \
        --cycle 288 \
        --train_epochs 30 \
        --patience 5 \
        --use_revin 0 \
        --itr 1 \
        --batch_size 32 \
        --learning_rate 0.003 \
        --random_seed 2024 \
        --add_loss fcv \
        --loss_patchlen ${patchlen} \
        --alpha_add_loss ${alpha_add} \
        --beta_add_loss ${beta}
done

echo "All PEMS03 best parameter jobs finished."