# Credit Risk Scoring Model

An end-to-end machine learning pipeline to predict loan default probability, simulating a real-world credit risk assessment workflow used in banking.

## Results

| Model | AUC-ROC | KS Statistic | Gini Coefficient |
|---|---|---|---|
| Logistic Regression (baseline) | 0.835 | 0.527 | 0.670 |
| XGBoost (production) | 0.825 | 0.502 | 0.649 |

**Top 5 Predictors of Default (SHAP):**
1. Delinquency history (`delinq_2yrs`)
2. Loan grade (`loan_grade`)
3. Credit utilization (`credit_utilization`)
4. Debt-to-income ratio (`dti`)
5. Interest rate (`interest_rate`)

## Dataset

Synthetic dataset of **30,000 loan records** (~30% default rate) generated to mirror real Lending Club distributions. Features include borrower financials, credit history, and engineered risk indicators.

## Pipeline

```
Raw Data → EDA → Preprocessing → SMOTE → Model Training → Evaluation → SHAP
```

1. **EDA** — Distribution analysis, default rates by segment, correlation heatmap
2. **Preprocessing** — Ordinal encoding (loan grade), one-hot encoding (loan purpose), StandardScaler
3. **SMOTE** — Synthetic Minority Oversampling to fix 70/30 class imbalance (training set only)
4. **Models** — Logistic Regression (interpretable baseline) + XGBoost (production)
5. **Evaluation** — AUC-ROC, KS Statistic, Gini Coefficient, Confusion Matrix
6. **SHAP** — TreeExplainer for feature-level explainability (regulatory compliance)

## Why These Metrics?

- **AUC-ROC** — Measures discriminatory power across all decision thresholds
- **KS Statistic** — Maximum separation between default/non-default score distributions; standard in credit decisioning
- **Gini Coefficient** — `2 × AUC − 1`; common in Basel II/III model validation
- **SHAP** — Required for adverse action notices; explains *why* a loan was rejected

## Tech Stack

`Python` · `Pandas` · `Scikit-learn` · `XGBoost` · `imbalanced-learn (SMOTE)` · `SHAP` · `Matplotlib` · `Seaborn`

## Files

```
lending_data.csv          — Dataset (30k rows)
generate_data.py          — Data generation script
credit_risk_model.py      — Full ML pipeline
eda.png                   — EDA visualizations
evaluation.png            — ROC curve, KS chart, confusion matrix
shap.png                  — SHAP feature importance + beeswarm
```

## Run

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn shap matplotlib seaborn
python generate_data.py
python credit_risk_model.py
```
