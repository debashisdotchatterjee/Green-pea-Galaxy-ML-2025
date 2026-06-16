````markdown
# Uncertainty-aware Empirical Modelling of Metallicity Calibration Scales in Green Pea Galaxies

This repository contains the Python/Colab code, reproducible analysis pipeline, figures, and tables for the revised study:

**Uncertainty-aware empirical modelling of metallicity calibration scales in Green Pea galaxies**

The work analyzes the Green Pea galaxy catalogue of Cardamone et al. (2009), with a specific focus on empirical modelling of the catalogue metallicity values, expressed as

\[
12+\log(\mathrm{O/H}).
\]

A central point of the revised analysis is that these metallicity values are treated as an **adopted strong-line metallicity calibration scale**, not as direct electron-temperature, or direct-\(T_e\), abundances. Therefore, predictive performance in this repository should be interpreted as reproducibility of the adopted catalogue scale, not as direct recovery of physically robust oxygen abundances.

---

## Repository purpose

This repository provides a transparent and reproducible workflow for:

- cleaning the Green Pea galaxy catalogue;
- identifying the metallicity response variable and uncertainty columns;
- converting metallicity upper/lower confidence limits into asymmetric uncertainty widths;
- constructing transformed predictors;
- performing exploratory data analysis;
- comparing baseline and flexible empirical regression models under repeated cross-validation;
- fitting a heteroscedastic Bayesian calibration model;
- producing posterior and residual diagnostics;
- saving all tables, plots, and reproducibility outputs automatically.

The analysis is deliberately conservative. It avoids claiming that a machine-learning model has discovered a new physical abundance law. Instead, it evaluates how well observed quantities such as redshift, [O III] equivalent width, star-formation rate, far-ultraviolet luminosity, and stellar mass reproduce the adopted metallicity calibration scale.

---

## Scientific framing

Green Pea galaxies are compact, extreme emission-line star-forming galaxies. Their metallicity estimates can be sensitive to the adopted abundance calibration, especially because these systems may have high ionization parameters, unusual abundance ratios, strong [O III] emission, and physical conditions that differ from ordinary local star-forming galaxies.

For this reason, the analysis distinguishes between:

1. **Empirical prediction of an adopted metallicity calibration scale**, which is what this repository does; and  
2. **Physical inference of true gas-phase oxygen abundance**, which would require direct-\(T_e\) measurements, tailored photoionization modelling, or independent physically calibrated abundance catalogues.

---

## Main analysis file

The main script is:

```text
green_pea_revision_colab_v3.py
````

A notebook version may also be provided:

```text
green_pea_revision_colab_v3_github_clean.ipynb
```

If GitHub gives an “Invalid Notebook” error, use the cleaned notebook version or run the `.py` script directly.

---

## Data

The expected input file is the Green Pea catalogue table, for example:

```text
peas_tbl4.csv
```

The code is designed to detect and normalize the following columns:

```text
DR7OBJID
RA
DEC
REDSHIFT
OIII_EW
L_FUV
L_FUV_ERR
SFR
SFR_ERR
METALLICITY
METALLICITY_ERR_UPPER
METALLICITY_ERR_LOWER
M_STELLAR
```

The target column is:

```text
METALLICITY
```

The uncertainty columns are:

```text
METALLICITY_ERR_UPPER
METALLICITY_ERR_LOWER
```

Important: in the catalogue, these are treated as **upper and lower confidence limits**, not direct error widths. The code converts them as:

```text
upper error width = METALLICITY_ERR_UPPER - METALLICITY
lower error width = METALLICITY - METALLICITY_ERR_LOWER
```

---

## Installation

The code is intended to run in Google Colab or a standard Python environment.

Install the required packages with:

```bash
pip install numpy pandas matplotlib scipy scikit-learn openpyxl pymc arviz
```

Optional packages:

```bash
pip install umap-learn
```

If `umap-learn` is not installed, the script will still run and will skip the UMAP plot.

---

## Running in Google Colab

Upload the Python script and the data file to Colab, then run:

```python
%run green_pea_revision_colab_v3.py
```

When prompted, upload:

```text
peas_tbl4.csv
```

The script will automatically create an output directory and a compressed output archive.

---

## Output directory

The script creates:

```text
green_pea_revised_outputs_v3/
```

and a zip archive:

```text
green_pea_revised_outputs_v3.zip
```

The output folder contains:

```text
figures/
tables/
models/
README/
```

Typical outputs include:

### Data-cleaning outputs

```text
column_name_mapping.csv
missingness_summary.csv
cleaned_modeling_data_head.csv
```

### Exploratory figures

```text
01_missingness_map.png
02_target_distribution.png
03_target_asymmetric_errors_corrected.png
04_spearman_correlation_heatmap_corrected.png
05_pairplot_core_variables.png
06_sfr_mass_relation.png
07_mzr_with_errors_corrected.png
08_fmr_alpha_diagnostic.png
```

### Cross-validation outputs

```text
repeated_cv_all_folds_baselines.csv
repeated_cv_model_summary_baselines.csv
out_of_fold_predictions_best_model.csv
09_cv_model_benchmark_rmse_baselines.png
10_cv_rmse_distribution_baselines.png
11_oof_observed_vs_predicted_best_baseline.png
```

### Residual and importance diagnostics

```text
12_residuals_vs_fitted_best_baseline.png
13_residuals_qq_best_baseline.png
14_residuals_vs_mstar_log.png
14_residuals_vs_log_sfr.png
14_residuals_vs_log_oiii_ew.png
15_permutation_importance_best_baseline.png
```

### Bayesian diagnostic outputs

```text
bayesian_posterior_summary.csv
16_corrected_posterior_coefficients.png
```

### Exploratory embedding outputs

```text
20_pca_embedding_exploratory.png
21_umap_embedding_exploratory.png
```

---

## Statistical models

The analysis compares several empirical models under repeated cross-validation, including:

* mean-only baseline;
* ordinary linear regression;
* Ridge regression;
* polynomial Ridge regression;
* Bayesian Ridge regression;
* robust Huber regression;
* Random Forest regression;
* Extra Trees regression;
* Gradient Boosting regression.

The Bayesian model is used as a heteroscedastic uncertainty diagnostic. It incorporates object-specific metallicity uncertainty widths and estimates an additional intrinsic scatter term.

---

## Main results from the current run

For the real Green Pea dataset used in the paper:

* raw sample size: 80 galaxies;
* supervised modelling sample: 66 galaxies;
* adopted metallicity mean: approximately 8.710;
* adopted metallicity standard deviation: approximately 0.104 dex;
* strongest empirical predictor: transformed [O III] equivalent width;
* best repeated-cross-validation model: Random Forest;
* Random Forest RMSE: approximately 0.0949 dex;
* Ridge/Bayesian Ridge/polynomial Ridge RMSE: approximately 0.099 dex;
* mean-only baseline RMSE: approximately 0.1029 dex;
* Bayesian intrinsic scatter estimate: approximately 0.057 dex.

These results should be interpreted as empirical reproducibility of the adopted metallicity calibration scale. They should not be interpreted as proof of direct physical abundance recovery.

---

## Important interpretation warning

This repository does **not** claim that machine learning recovers true Green Pea oxygen abundances.

The metallicity response used here is an adopted catalogue calibration value. Therefore:

* low RMSE means agreement with the adopted catalogue scale;
* low RMSE does not prove agreement with direct-(T_e) abundances;
* low RMSE does not prove a new physical chemical-evolution relation;
* model performance must be compared against simple baselines;
* dimensionality reduction is exploratory only.

A physically stronger version of the analysis would require direct-(T_e) abundances, photoionization-model abundances, or an independent Green Pea/extreme emission-line galaxy catalogue.

---

## Notebook rendering issue on GitHub

Colab notebooks sometimes fail to render on GitHub with an error such as:

```text
Invalid Notebook:
the 'state' key is missing from 'metadata.widgets'
```

This is a notebook metadata issue, not a code issue.

To clean the notebook, run:

```python
import json
from pathlib import Path

notebook_path = Path("green_pea_revision_colab_v3.ipynb")

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

nb.get("metadata", {}).pop("widgets", None)

for cell in nb.get("cells", []):
    cell.get("metadata", {}).pop("widgets", None)

clean_path = notebook_path.with_name(notebook_path.stem + "_github_clean.ipynb")

with open(clean_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Saved cleaned notebook: {clean_path}")
```

Upload the cleaned notebook to GitHub.

---

## Recommended citation

If using this code or analysis, please cite the associated manuscript and the original Green Pea catalogue:

```text
Cardamone et al. (2009), Monthly Notices of the Royal Astronomical Society.
```

A full citation for the present manuscript should be added here after submission or publication.

---

## Reproducibility notes

The code uses fixed random seeds where possible. Some small numerical differences may occur across Python, scikit-learn, PyMC, or Colab runtime versions.

Recommended environment:

```text
Python >= 3.10
numpy
pandas
matplotlib
scipy
scikit-learn
pymc
arviz
openpyxl
umap-learn optional
```

---

## Suggested repository structure

```text
Green-pea-Galaxy-ML-2025/
│
├── README.md
├── green_pea_revision_colab_v3.py
├── green_pea_revision_colab_v3_github_clean.ipynb
├── requirements.txt
│
├── data/
│   └── peas_tbl4.csv
│
├── outputs/
│   └── green_pea_revised_outputs_v3/
│
├── manuscript/
│   └── revised_manuscript.tex
│
└── figures/
    └── selected manuscript figures
```

---

## License

Please add the appropriate license for the repository. If the code is intended to be openly reusable, a standard choice is:

```text
MIT License
```

or

```text
GNU General Public License v3.0
```

The data should retain the citation and usage conditions of the original source.

---

## Contact

For questions about the code or analysis, contact:

```text
Dr. Debashis Chatterjee
Department of Statistics
Visva-Bharati University
Email: debashis.chatterjee@visva-bharati.ac.in
```

```
```
