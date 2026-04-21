python visualize_cvl_heatmaps.py \
  --base_dir ./results/long_term_forecast_ETTh2_96_96_iTransformer_ETTh2_addlossNone_numpairsmax_patchlen16_alpha1.0_beta0.1 \
  --cv_dir ./results/long_term_forecast_ETTh2_96_96_iTransformer_ETTh2_addlossfcv_numpairsmax_patchlen3_alpha0.1_beta0.9 \
  --save_dir ./heatmap_vis \
  --n_bins 8

python visualize_cvl_heatmaps.py \
  --base_dir ./results/long_term_forecast_weather_96_96_iTransformer_custom_addlossNone_numpairsmax_patchlen16_alpha1.0_beta0.1 \
  --cv_dir ./results/long_term_forecast_weather_96_96_iTransformer_custom_addlossfcv_numpairsmax_patchlen6_alpha0.5_beta0.5 \
  --save_dir ./heatmap_vis \
  --n_bins 8

python visualize_cvl_heatmaps.py \
  --base_dir ./results/long_term_forecast_ECL_96_96_iTransformer_custom_addlossNone_numpairsmax_patchlen16_alpha1.0_beta0.1 \
  --cv_dir ./results/long_term_forecast_ECL_96_96_iTransformer_custom_addlossfcv_numpairsmax_patchlen6_alpha0.5_beta0.5 \
  --save_dir ./heatmap_vis \
  --n_bins 8

python visualize_cvl_heatmaps.py \
  --base_dir ./results/long_term_forecast_ECL_96_720_iTransformer_custom_addlossNone_numpairsmax_patchlen16_alpha1.0_beta0.1 \
  --cv_dir ./results/long_term_forecast_ECL_96_720_iTransformer_custom_addlossfcv_numpairsmax_patchlen6_alpha0.5_beta0.5 \
  --save_dir ./heatmap_vis \
  --n_bins 8