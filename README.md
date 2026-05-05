# CvLoss

Cross-Variable Loss (CvLoss) is a plug-in structural regularizer for multivariate time series forecasting. It augments the standard direct forecasting objective with residual consistency constraints over cross-variable forecast patches, encouraging the predicted future variables to preserve synchronous and asynchronous relationships.

This repository is an anonymous research release. Author names, affiliations, and contact information are intentionally omitted from project files.

![CvLoss framework](Doc/framework.png)

## Implement CvLoss

Implement CvLoss by adapting the following script in your pipeline:

```python
loss_tmp = criterion(outputs, batch_y)
B, T, D = outputs.shape; device = outputs.device
patch_len = stride = 32
if (T - patch_len) % stride != 0:
    raise ValueError("(T - patch_len) % stride != 0")

out_p = outputs.unfold(1, patch_len, stride).permute(0, 1, 3, 2).contiguous()
y_p = batch_y.unfold(1, patch_len, stride).permute(0, 1, 3, 2).contiguous()
B, P, L, D = out_p.shape
out_nodes = out_p.permute(0, 1, 3, 2).reshape(B, P * D, L)
y_nodes = y_p.permute(0, 1, 3, 2).reshape(B, P * D, L)

N, num_pairs = P * D, (P * D) * (P * D - 1) // 2
idx_i = torch.randint(0, N, (num_pairs,), device=device)
idx_j = torch.randint(0, N, (num_pairs,), device=device)
patch_i, patch_j = idx_i // D, idx_j // D
var_i, var_j = idx_i % D, idx_j % D
mask = (idx_i < idx_j) & ~((var_i == var_j) & (patch_i != patch_j))
idx_i, idx_j = idx_i[mask], idx_j[mask]

pred_diff = out_nodes[:, idx_i] - out_nodes[:, idx_j]
true_diff = y_nodes[:, idx_i] - y_nodes[:, idx_j]
loss_add = (pred_diff - true_diff).abs().mean()
loss = 0.5 * loss_tmp + 0.5 * loss_add
```

## Repository Structure

```text
Doc/                    Paper draft and framework figure
Time-Series-Library/    Main experimental code with CvLoss-enabled objectives
iTransformer/           Adapted iTransformer experiments and scripts
TQNet/                  TQNet backbone experiments and ablations
PDF/                    PDF backbone experiments
CFPT/                   CFPT backbone experiments
TimeFilter/             TimeFilter backbone experiments
TimeBridge/             TimeBridge backbone experiments
baseline_results.py     Utility for summarizing recorded result files
search_config.py        Utility for parsing/searching add-loss experiments
*.log                   Recorded training logs
```

## Environment

The code follows the common PyTorch time-series forecasting stack. A minimal environment is:

```bash
conda create -n cvloss python=3.10
conda activate cvloss
pip install torch numpy pandas scikit-learn matplotlib einops tqdm PyWavelets
```

Some optional backbones may require additional packages. Install them only when running the corresponding model.

## Data

Datasets are not included in the repository. Place data under the dataset paths expected by the scripts, for example:

```text
Time-Series-Library/dataset/ETT-small/ETTh1.csv
Time-Series-Library/dataset/ETT-small/ETTh2.csv
Time-Series-Library/dataset/weather/weather.csv
Time-Series-Library/dataset/electricity/electricity.csv
Time-Series-Library/dataset/traffic/traffic.csv
```

The same layout is used by the adapted backbone folders when their scripts are run from that folder.

## Running CvLoss

The main CvLoss implementation is integrated into the long-term forecasting training loop in `Time-Series-Library/exp/exp_long_term_forecasting.py`. The key arguments are:

- `--add_loss`: `None`, `scv`, `stcv`, or `fcv`
- `--loss_patchlen`: patch granularity for the CvLoss graph
- `--alpha_add_loss`: weight for the direct forecasting loss
- `--beta_add_loss`: weight for CvLoss

Example single run:

```bash
cd Time-Series-Library
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96_fcv \
  --model iTransformer \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des Exp \
  --itr 1 \
  --add_loss fcv \
  --loss_patchlen 3 \
  --alpha_add_loss 0.5 \
  --beta_add_loss 0.5
```

## Experiment Scripts

Representative sweep scripts are provided under `Time-Series-Library/scripts_/`:

```bash
cd Time-Series-Library
bash scripts_/iTransformer/itransformer_cv.sh
bash scripts_/PatchTST/PatchTST_cv.sh
bash scripts_/DLinear/DLinear_cv.sh
```

Backbone-specific folders also contain scripts for their original and CvLoss-enhanced experiments.

## Results

Recorded metric files can be summarized with:

```bash
python baseline_results.py --file Time-Series-Library/cv_loss_iTransformer.txt --include-addloss
```

The parser expects result blocks in the format written by the training scripts:

```text
long_term_forecast_...
mse:<value>, mae:<value>
```
