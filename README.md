# Heart Disease Risk Prediction: does a model trained at one hospital work at another?

We trained Logistic Regression, XGBoost (tuned with Optuna) and a small MLP on the **Cleveland** cohort of the UCI Heart Disease data. Then we tested them unchanged on patients from two other sites, **Hungary** and **Switzerland**. Beyond ROC-AUC we looked at significance (DeLong), calibration, domain adaptation (CORAL) and SHAP explanations.

This is an academic project, not a medical tool.

## Results

ROC-AUC on data never used for training or model selection. From `outputs/tables/phase4_external_metrics.csv`.

| Model | Cleveland test (n=46) | Hungarian (n=294) | Swiss (n=123) |
| --- | ---: | ---: | ---: |
| Logistic Regression | **0.949** | 0.869 | 0.654 |
| XGBoost + Optuna | 0.935 | **0.876** | **0.720** |
| Small MLP | 0.935 | 0.833 | 0.607 |

- **The site gap is real.** The best model drops from 0.949 on Cleveland to 0.876 on Hungarian and 0.720 on Swiss.
- **Most differences aren't significant.** Only XGBoost over Logistic Regression on Swiss passes DeLong (p = 0.031), and with just 8 negative cases that's weak evidence.
- **Calibration helps a little.** Isotonic calibration cuts Cleveland ECE for Logistic Regression from 0.106 to 0.072, but Swiss ECE stays around 0.39–0.45.
- **CORAL didn't help.** It lowered XGBoost's Swiss ROC-AUC from 0.720 to 0.628.
- **The Swiss cohort is 93.5% positive** (115 of 123), so read its precision numbers with care.

<p align="center">
  <img src="outputs/figures/top_distribution_shifts.png" width="60%" alt="Top feature distribution shifts of the Hungarian and Swiss cohorts relative to Cleveland train"/>
  <img src="outputs/figures/phase4_shap_summary_xgboost.png" width="36%" alt="SHAP summary plot for the optimized XGBoost model"/>
</p>
<p align="center"><sub>Left: biggest feature shifts versus Cleveland train (chi-square / KS tests). Right: SHAP summary for XGBoost.</sub></p>

## How it works

1. **Data** (`01_dataset.ipynb`): load the three cohorts, turn the label into disease vs no disease, and split Cleveland 70/15/15 with stratification. Hungary and Switzerland are held out.
2. **EDA** (`02_eda.ipynb`): class balance, missingness and cross-site shift tests. Hungarian is missing `ca` in 291 of 294 rows.
3. **Modeling** (`03_modeling.ipynb`): one shared pipeline expands 13 features to 19 with clinical interaction terms, then imputes and scales. Models are tuned with 5-fold CV on Cleveland train and selected on validation.
4. **Evaluation** (`04_evaluation.ipynb`): the test sets are touched only here, at a fixed 0.5 threshold.

## Run it

```bash
git clone https://github.com/aexomir/heart-disease-prediction.git
cd heart-disease-prediction
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/
```

Run the notebooks in order. The raw UCI files are already in `data/raw/`.

`04_evaluation.ipynb` imports classes from `src/03_modeling.py`, which isn't in the repo right now, so it won't re-run as-is. All of its outputs are committed under `outputs/`.

## Limitations

Small data (the Cleveland test set is 46 rows), a very imbalanced Swiss cohort, heavy missingness at the external sites, and a fixed 0.5 threshold rather than a clinically chosen one. No prospective or clinical validation.
