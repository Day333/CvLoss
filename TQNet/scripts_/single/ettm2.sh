#!/usr/bin/env bash
set -e

GPU_ID=3

configs=(
  "96 24 0.7"
  "192 3 0.5"
  "336 3 0.9"
  "720 6 0.6"
)

for config in "${configs[@]}"; do
    read -r pred_len patchlen beta <<< "$config"
    
    alpha_add=$(python - <<PY
b=float("${beta}")
a=1.0-b
print(f"{a:.6f}".rstrip('0').rstrip('.'))
PY
)

    echo "======================================================"
    echo "Running ETTm2: pred_len=${pred_len}, patchlen=${patchlen}, beta=${beta}, alpha_add=${alpha_add}"
    echo "======================================================"

    CUDA_VISIBLE_DEVICES=$GPU_ID python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path ETTm2.csv \
        --model_id "ETTm2_96_${pred_len}_fcv_patch${patchlen}_b${beta}" \
        --model TQNet \
        --data ETTm2 \
        --features M \
        --seq_len 96 \
        --pred_len ${pred_len} \
        --enc_in 7 \
        --cycle 96 \
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

echo "All ETTm2 best parameter jobs finished."