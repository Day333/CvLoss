<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0F2027,50:2C5364,100:00C9FF&height=220&section=header&text=CvLoss&fontSize=72&fontColor=FFFFFF&animation=fadeIn&fontAlignY=38&desc=Cross-Variable%20Loss%20for%20Multivariate%20Time%20Series%20Forecasting&descAlignY=62&descSize=18" width="100%" alt="CvLoss header" />

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-8A2BE2?style=for-the-badge)](https://neurips.cc/)
[![arXiv](https://img.shields.io/badge/arXiv-2608.05742-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2608.05742)
[![PyTorch](https://img.shields.io/badge/PyTorch-Plug--in_Loss-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](#1-implement-cvloss)
[![Code](https://img.shields.io/badge/Code-Official-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Day333/CvLoss)

<br />

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=00C9FF&center=true&vCenter=true&width=820&lines=Forecast+together.+Preserve+structure.;A+plug-in+loss+for+multivariate+forecasting.;Accepted+at+NeurIPS+2026." alt="CvLoss typing animation" />

**One structural objective. Multiple forecasting backbones. Consistent multivariate futures.**

[📄 Paper](https://arxiv.org/abs/2608.05742) · [💻 Code](https://github.com/Day333/CvLoss) · [🧩 Plug-in Loss](#1-implement-cvloss)

</div>

---

# CvLoss

Cross-Variable Loss (CvLoss) is a plug-in structural regularizer for multivariate time series forecasting. It augments the standard direct forecasting objective with residual consistency constraints over cross-variable forecast patches, encouraging the predicted future variables to preserve synchronous and asynchronous relationships.

This repository is an anonymous research release. Author names, affiliations, and contact information are intentionally omitted from project files.

![CvLoss framework](Doc/framework.png)

## 1. Implement CvLoss

Implement CvLoss by adapting the following script in your pipeline:

```python
loss_tmp = criterion(outputs, batch_y)
B, T, D = outputs.shape; device = outputs.device
patch_len = stride = 16
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

## 2. Repository Structure

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

## 3. Environment

The code follows the common PyTorch time-series forecasting stack. A minimal environment is:

```bash
conda create -n cvloss python=3.10
conda activate cvloss
pip install torch numpy pandas scikit-learn matplotlib einops tqdm PyWavelets
```

Some optional backbones may require additional packages. Install them only when running the corresponding model.

## 4. Data

Datasets are not included in the repository. Place data under the dataset paths expected by the scripts, for example:

```text
Time-Series-Library/dataset/ETT-small/ETTh1.csv
Time-Series-Library/dataset/ETT-small/ETTh2.csv
Time-Series-Library/dataset/weather/weather.csv
Time-Series-Library/dataset/electricity/electricity.csv
Time-Series-Library/dataset/traffic/traffic.csv
```

The same layout is used by the adapted backbone folders when their scripts are run from that folder.

## 5. Running CvLoss

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

## 6. Experiment Scripts

Representative sweep scripts are provided under `Time-Series-Library/scripts_/`:

```bash
cd Time-Series-Library
bash scripts_/iTransformer/itransformer_cv.sh
bash scripts_/PatchTST/PatchTST_cv.sh
bash scripts_/DLinear/DLinear_cv.sh
```

Backbone-specific folders also contain scripts for their original and CvLoss-enhanced experiments.

## 7. Results

Recorded metric files can be summarized with:

```bash
python baseline_results.py --file Time-Series-Library/cv_loss_iTransformer.txt --include-addloss
```

The parser expects result blocks in the format written by the training scripts:

```text
long_term_forecast_...
mse:<value>, mae:<value>
```

## 8. Citation

If you find CvLoss useful in your research, please cite our paper:

```bibtex
@inproceedings{ding2026cvloss,
  title     = {Multivariate Time Series Forecasting needs Cross Variable Loss},
  author    = {Ding, Kuiye and Hu, Yifan and Wang, Hanchen and Xue, Hao},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://arxiv.org/abs/2608.05742}
}
```

## 9. Acknowledgements

We sincerely thank the authors of the following works for advancing learning objectives for time-series forecasting and inspiring this line of research:

- **FreDF: Learning to Forecast in the Frequency Domain** — [Paper](https://arxiv.org/abs/2402.02399) · [Code](https://github.com/Master-PLC/FreDF)
- **Time-o1: Time-Series Forecasting Needs Transformed Label Alignment** — [Paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/0cd62dea69635f4c5b569848267fe5a8-Paper-Conference.pdf) · [Code](https://github.com/Master-PLC/Time-o1)
- **DistDF: Time-Series Forecasting Needs Joint-Distribution Wasserstein Alignment** — [Paper](https://arxiv.org/abs/2510.24574) · [Code](https://github.com/Master-PLC/DistDF)
