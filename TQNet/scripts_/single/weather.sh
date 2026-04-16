#!/usr/bin/env bash
set -e

GPU_ID=3


configs=(
  "96 3 1.0"
  # "192 3 1.0"
  # "336 3 0.5"
  # "720 3 0.5"
)

for config in "${configs[@]}"; do
    read -r pred_len patchlen beta <<< "$config"
    
    alpha_add=$(awk "BEGIN {print 1.0 - $beta}")
    
    echo "======================================================"
    echo "Running Weather: pred_len=${pred_len}, patchlen=${patchlen}, beta=${beta}"
    echo "======================================================"

    CUDA_VISIBLE_DEVICES=$GPU_ID python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path weather.csv \
        --model_id "weather_96_${pred_len}_fcv_patch${patchlen}_b${beta}" \
        --model TQNet \
        --data custom \
        --features M \
        --seq_len 96 \
        --pred_len ${pred_len} \
        --enc_in 21 \
        --cycle 144 \
        --train_epochs 30 \
        --patience 5 \
        --dropout 0.5 \
        --itr 1 \
        --batch_size 64 \
        --learning_rate 0.001 \
        --random_seed 2024 \
        --add_loss fcv \
        --loss_patchlen ${patchlen} \
        --alpha_add_loss ${alpha_add} \
        --beta_add_loss ${beta}
done

echo "All Weather best parameter jobs finished."