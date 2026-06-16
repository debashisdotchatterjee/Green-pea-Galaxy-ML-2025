# Green Pea revised analysis pipeline v3

Use `green_pea_revision_colab_v3.py` or `green_pea_revision_colab_v3.ipynb` in Google Colab.
Upload `peas_tbl4.csv` when asked.

## Main fixes from v2

1. Removed `mean_squared_error(..., squared=False)` and replaced it with a version-safe RMSE function.
2. Removed fold-incompatible full-data Gaussian Process `alpha` array. GPR is optional and off by default.
3. Removed seaborn boxplot dependency for RMSE distribution; matplotlib is used instead.
4. Baselines do not receive `sample_weight` by default, avoiding estimator-specific `unexpected keyword` errors.
5. Weighted RMSE is still computed as a metric using object-specific metallicity errors.
6. The heteroscedastic Bayesian model keeps object-specific metallicity errors in the likelihood.
7. The script never produces [Fe/H] or negative-metallicity plots.
8. Default repeated CV is 5 repeats × 5 folds for stability; increase `N_REPEATS_CV` to 20 only after the first run succeeds.

## Scientific framing

If the target detected is `METALLICITY`, interpret results as empirical reproduction of the adopted strong-line metallicity calibration scale, not direct oxygen-abundance recovery.
