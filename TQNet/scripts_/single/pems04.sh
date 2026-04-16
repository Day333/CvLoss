#!/usr/bin/env bash
set -e

GPU_ID=3


configs=(
  "12 3 0.7"
  "24 6 0.6"
  "48 3 0.7"
)

for config in "${configs[@]}"; do
    read -r pred_len patchlen beta <<< "$config"
    
    alpha_add=$(awk "BEGIN {print 1.0 - $beta}")
    
    echo "======================================================"
    echo "Running PEMS04: pred_len=${pred_len}, patchlen=${patchlen}, beta=${beta}"
    echo "======================================================"

    CUDA_VISIBLE_DEVICES=$GPU_ID python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path PEMS04.npz \
        --model_id "PEMS04_96_${pred_len}_fcv_patch${patchlen}_b${beta}" \
        --model TQNet \
        --data PEMS \
        --features M \
        --seq_len 96 \
        --pred_len ${pred_len} \
        --enc_in 307 \
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

echo "All PEMS04 best parameter jobs finished."