#!/usr/bin/env bash
set -e

GPU_ID=3

configs=(
  "12 6 0.6"
  "24 12 0.6"
  "48 12 0.5"
)

for config in "${configs[@]}"; do
    read -r pred_len patchlen beta <<< "$config"
    
    alpha_add=$(awk "BEGIN {print 1.0 - $beta}")
    
    echo "======================================================"
    echo "Running PEMS07: pred_len=${pred_len}, patchlen=${patchlen}, beta=${beta}"
    echo "======================================================"

    CUDA_VISIBLE_DEVICES=$GPU_ID python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path PEMS07.npz \
        --model_id "PEMS07_96_${pred_len}_fcv_patch${patchlen}_b${beta}" \
        --model TQNet \
        --data PEMS \
        --features M \
        --seq_len 96 \
        --pred_len ${pred_len} \
        --enc_in 883 \
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

echo "All PEMS07 best parameter jobs finished."