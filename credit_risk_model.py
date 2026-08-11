"""
Credit Risk Scoring Model
=========================
Full ML pipeline: EDA → Preprocessing → SMOTE → Model Training → Evaluation → SHAP
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS — why each one?
# pandas   → tabular data manipulation (loading CSV, filtering, feature creation)
# numpy    → numerical operations and array math
# matplotlib/seaborn → plotting (EDA charts, confusion matrix, ROC curve)
# sklearn  → preprocessing (scaling, encoding, train/test split), metrics, Logistic Regression
# xgboost  → gradient boosted trees — state of the art for tabular credit data
# imblearn → SMOTE: fix class imbalance without losing data
# shap     → explain model predictions (regulatory requirement in banking)
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import StandardScaler, LabelEncoder
from sklearn.linear_model     import LogisticRegression
from sklearn.metrics          import (roc_auc_score, roc_curve,
                                       classification_report,
                                       confusion_matrix, ConfusionMatrixDisplay)
from xgboost                  import XGBClassifier
from imblearn.over_sampling   import SMOTE
import shap

OUTPUT = '/home/claude/credit_risk/'
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)

# ═════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1 — Loading Data")
print("=" * 60)

df = pd.read_csv(OUTPUT + 'lending_data.csv')
print(f"Shape: {df.shape}")
print(f"\nDefault rate: {df['loan_status'].mean()*100:.1f}%")
print(f"Non-default:  {(1 - df['loan_status'].mean())*100:.1f}%")
print("\nColumn types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())

# ═════════════════════════════════════════════════════════════════════════════
# 2. EDA — Exploratory Data Analysis
# Why: Before touching models, understand the data shape, distributions,
#      correlations. Catch issues early. Tells a story in interviews.
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2 — EDA")
print("=" * 60)

fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

# 2a. Class imbalance — the core challenge
ax0 = fig.add_subplot(gs[0, 0])
counts = df['loan_status'].value_counts()
ax0.bar(['Paid (0)', 'Default (1)'], counts.values, color=['#4CAF50','#F44336'], edgecolor='white')
ax0.set_title('Class Distribution\n(Imbalanced)', fontweight='bold')
ax0.set_ylabel('Count')
for i, v in enumerate(counts.values):
    ax0.text(i, v + 200, f'{v:,}', ha='center', fontweight='bold')

# 2b. Default rate by loan grade — shows grade is a strong predictor
ax1 = fig.add_subplot(gs[0, 1])
grade_default = df.groupby('loan_grade')['loan_status'].mean().reindex(['A','B','C','D','E','F','G'])
ax1.bar(grade_default.index, grade_default.values * 100, color='#2196F3', edgecolor='white')
ax1.set_title('Default Rate by\nLoan Grade', fontweight='bold')
ax1.set_ylabel('Default Rate (%)')
ax1.set_xlabel('Grade (A=Best, G=Riskiest)')

# 2c. DTI distribution split by outcome
ax2 = fig.add_subplot(gs[0, 2])
df[df['loan_status']==0]['dti'].clip(0, 40).hist(ax=ax2, bins=40, alpha=0.6, color='#4CAF50', label='Paid')
df[df['loan_status']==1]['dti'].clip(0, 40).hist(ax=ax2, bins=40, alpha=0.6, color='#F44336', label='Default')
ax2.set_title('DTI Distribution\nby Outcome', fontweight='bold')
ax2.set_xlabel('Debt-to-Income Ratio')
ax2.legend()

# 2d. Credit utilization by outcome
ax3 = fig.add_subplot(gs[1, 0])
df.boxplot(column='credit_utilization', by='loan_status', ax=ax3,
           boxprops=dict(color='#2196F3'),
           medianprops=dict(color='red', linewidth=2))
ax3.set_title('Credit Utilization\nvs Default', fontweight='bold')
ax3.set_xlabel('0=Paid, 1=Default')
ax3.set_ylabel('Credit Utilization')
plt.sca(ax3); plt.title('Credit Utilization vs Default', fontweight='bold')

# 2e. Interest rate distribution
ax4 = fig.add_subplot(gs[1, 1])
df[df['loan_status']==0]['interest_rate'].hist(ax=ax4, bins=40, alpha=0.6, color='#4CAF50', label='Paid')
df[df['loan_status']==1]['interest_rate'].hist(ax=ax4, bins=40, alpha=0.6, color='#F44336', label='Default')
ax4.set_title('Interest Rate\nby Outcome', fontweight='bold')
ax4.set_xlabel('Interest Rate (%)')
ax4.legend()

# 2f. Correlation heatmap — find which features move together
ax5 = fig.add_subplot(gs[1, 2])
num_cols = ['dti','credit_utilization','loan_to_income','interest_rate',
            'missed_payment_flag','employment_length','credit_history_years','loan_status']
corr = df[num_cols].corr()
sns.heatmap(corr, ax=ax5, annot=True, fmt='.2f', cmap='RdYlGn_r',
            center=0, linewidths=0.5, annot_kws={'size': 7})
ax5.set_title('Correlation Matrix', fontweight='bold')
ax5.tick_params(axis='x', rotation=45, labelsize=7)
ax5.tick_params(axis='y', rotation=0, labelsize=7)

# 2g. Default rate by missed payment
ax6 = fig.add_subplot(gs[2, 0])
mp_default = df.groupby('missed_payment_flag')['loan_status'].mean()
ax6.bar(['No Missed\nPayment', 'Missed\nPayment'], mp_default.values * 100,
        color=['#4CAF50','#F44336'], edgecolor='white')
ax6.set_title('Default Rate by\nMissed Payment History', fontweight='bold')
ax6.set_ylabel('Default Rate (%)')

# 2h. Annual income vs default (boxplot)
ax7 = fig.add_subplot(gs[2, 1])
df[df['loan_status']==0]['annual_income'].clip(0, 200000).hist(ax=ax7, bins=40, alpha=0.6, color='#4CAF50', label='Paid')
df[df['loan_status']==1]['annual_income'].clip(0, 200000).hist(ax=ax7, bins=40, alpha=0.6, color='#F44336', label='Default')
ax7.set_title('Annual Income\nby Outcome', fontweight='bold')
ax7.set_xlabel('Annual Income ($)')
ax7.legend()

# 2i. Loan purpose default rates
ax8 = fig.add_subplot(gs[2, 2])
purpose_default = df.groupby('loan_purpose')['loan_status'].mean().sort_values(ascending=True)
ax8.barh(purpose_default.index, purpose_default.values * 100, color='#9C27B0', edgecolor='white')
ax8.set_title('Default Rate by\nLoan Purpose', fontweight='bold')
ax8.set_xlabel('Default Rate (%)')

plt.suptitle('Credit Risk — Exploratory Data Analysis', fontsize=16, fontweight='bold', y=1.01)
plt.savefig(OUTPUT + 'eda.png', dpi=150, bbox_inches='tight')
plt.close()
print("EDA chart saved.")

# ═════════════════════════════════════════════════════════════════════════════
# 3. PREPROCESSING
# Why: Raw data has categoricals (grade, purpose) that models can't read.
#      Features are on different scales — income in 100ks, utilization 0–1.
#      Logistic Regression is scale-sensitive; StandardScaler fixes this.
#      XGBoost is scale-invariant but we scale anyway for consistency.
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3 — Preprocessing")
print("=" * 60)

df_model = df.copy()

# Encode loan_grade: A=0, B=1 ... G=6 (ordinal — grade has natural order)
grade_map = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6}
df_model['loan_grade'] = df_model['loan_grade'].map(grade_map)

# One-hot encode loan_purpose (no natural order — medical isn't "more" than home)
# drop_first=True to avoid dummy variable trap (multicollinearity)
df_model = pd.get_dummies(df_model, columns=['loan_purpose'], drop_first=True)

FEATURES = [
    'annual_income', 'loan_amount', 'interest_rate', 'employment_length',
    'credit_history_years', 'num_open_accounts', 'revolving_balance',
    'revolving_limit', 'delinq_2yrs', 'loan_grade', 'dti',
    'credit_utilization', 'loan_to_income', 'missed_payment_flag',
    'loan_purpose_home_improvement', 'loan_purpose_medical',
    'loan_purpose_other', 'loan_purpose_small_business'
]
TARGET = 'loan_status'

X = df_model[FEATURES]
y = df_model[TARGET]

print(f"Features: {X.shape[1]}")
print(f"Class balance before SMOTE: {y.value_counts().to_dict()}")

# Train/test split — 80/20, stratified to preserve class ratio in both sets
# Why stratify? Without it, by chance test set might have very few defaults
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# SMOTE — Synthetic Minority Oversampling Technique
# Why: 15% defaults means model trained naively will just predict "Paid" for
#      everyone and get 85% accuracy while being useless. SMOTE creates
#      synthetic default examples by interpolating between existing ones.
#      Applied ONLY on training data — test set must stay real-world ratio.
# ─────────────────────────────────────────────────────────────────────────────
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"\nAfter SMOTE — Class balance: {pd.Series(y_train_sm).value_counts().to_dict()}")

# StandardScaler — zero mean, unit variance
# Why: Logistic Regression uses gradient descent; large-scale features dominate
#      the gradient update and slow convergence. Scaling fixes this.
# Fit on training data only — never on test (data leakage prevention)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled  = scaler.transform(X_test)   # transform only, don't refit

# ═════════════════════════════════════════════════════════════════════════════
# 4. MODEL TRAINING
# Why two models?
# - Logistic Regression: interpretable baseline, regulators love it
# - XGBoost: best performance on tabular data, industry standard for credit
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4 — Model Training")
print("=" * 60)

# Logistic Regression
# C=0.1 → regularization (penalizes large weights to prevent overfitting)
# max_iter=1000 → enough iterations to converge
lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train_sm)
lr_probs = lr.predict_proba(X_test_scaled)[:, 1]  # probability of default
lr_auc   = roc_auc_score(y_test, lr_probs)
print(f"Logistic Regression AUC-ROC: {lr_auc:.4f}")

# XGBoost
# n_estimators=500 → 500 trees in the ensemble
# max_depth=5 → each tree can go 5 levels deep
# learning_rate=0.05 → small steps for better generalization
# subsample=0.8 → each tree trained on 80% of data (reduces overfitting)
# colsample_bytree=0.8 → each tree uses 80% of features (reduces overfitting)
# scale_pos_weight: since we used SMOTE, classes are balanced → set to 1
xgb = XGBClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=1, eval_metric='auc',
    random_state=42, verbosity=0
)
xgb.fit(X_train_sm, y_train_sm,   # XGBoost doesn't need scaling
        eval_set=[(X_test, y_test)],
        verbose=False)
xgb_probs = xgb.predict_proba(X_test)[:, 1]
xgb_auc   = roc_auc_score(y_test, xgb_probs)
print(f"XGBoost AUC-ROC:            {xgb_auc:.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# 5. EVALUATION METRICS
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5 — Evaluation")
print("=" * 60)

# KS Statistic — maximum separation between default and non-default score dists
# How to compute: for each threshold, compute TPR − FPR. Max value = KS.
def ks_statistic(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return np.max(tpr - fpr)

lr_ks  = ks_statistic(y_test, lr_probs)
xgb_ks = ks_statistic(y_test, xgb_probs)
print(f"Logistic Regression  → AUC: {lr_auc:.4f} | KS: {lr_ks:.4f} | Gini: {2*lr_auc-1:.4f}")
print(f"XGBoost              → AUC: {xgb_auc:.4f} | KS: {xgb_ks:.4f} | Gini: {2*xgb_auc-1:.4f}")

# Optimal threshold via KS (best separation point)
fpr_x, tpr_x, thresh_x = roc_curve(y_test, xgb_probs)
optimal_idx   = np.argmax(tpr_x - fpr_x)
optimal_thresh = thresh_x[optimal_idx]
print(f"\nXGBoost optimal threshold: {optimal_thresh:.3f}")

xgb_preds = (xgb_probs >= optimal_thresh).astype(int)
print("\nXGBoost Classification Report:")
print(classification_report(y_test, xgb_preds, target_names=['Paid','Default']))

# ═════════════════════════════════════════════════════════════════════════════
# 6. EVALUATION CHARTS
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 6a. ROC Curves — both models
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
ax = axes[0]
ax.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={lr_auc:.3f})', color='#2196F3', lw=2)
ax.plot(fpr_x,  tpr_x,  label=f'XGBoost (AUC={xgb_auc:.3f})',           color='#FF5722', lw=2)
ax.plot([0,1], [0,1], 'k--', lw=1, label='Random Baseline')
ax.scatter(fpr_x[optimal_idx], tpr_x[optimal_idx], color='red', s=100,
           zorder=5, label=f'Optimal Threshold ({optimal_thresh:.2f})')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves', fontweight='bold'); ax.legend(fontsize=9)

# 6b. KS Plot — score distribution separation
ax = axes[1]
default_scores  = xgb_probs[y_test == 1]
ndefault_scores = xgb_probs[y_test == 0]
thresholds = np.linspace(0, 1, 200)
ks_tpr = [np.mean(default_scores  <= t) for t in thresholds]
ks_fpr = [np.mean(ndefault_scores <= t) for t in thresholds]
ax.plot(thresholds, ks_tpr, label='Cumulative Defaults',    color='#F44336', lw=2)
ax.plot(thresholds, ks_fpr, label='Cumulative Non-Defaults',color='#4CAF50', lw=2)
ks_idx = np.argmax(np.abs(np.array(ks_tpr) - np.array(ks_fpr)))
ax.axvline(thresholds[ks_idx], color='navy', linestyle='--',
           label=f'KS={xgb_ks:.3f} at {thresholds[ks_idx]:.2f}')
ax.set_xlabel('Score Threshold'); ax.set_ylabel('Cumulative %')
ax.set_title('KS Chart (XGBoost)', fontweight='bold'); ax.legend(fontsize=9)

# 6c. Confusion Matrix
ax = axes[2]
cm = confusion_matrix(y_test, xgb_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=['Paid','Default'])
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title(f'Confusion Matrix\n(threshold={optimal_thresh:.2f})', fontweight='bold')

plt.suptitle('Model Evaluation — Credit Risk Scoring', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT + 'evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nEvaluation charts saved.")

# ═════════════════════════════════════════════════════════════════════════════
# 7. SHAP — Model Explainability
# Why: Banks can't legally use a black-box model. Regulators (OCC, Fed) require
#      "adverse action notices" — you must tell a rejected applicant WHY.
#      SHAP assigns each feature a contribution value for each prediction.
#      Positive SHAP = pushed score toward default. Negative = away from default.
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7 — SHAP Explainability")
print("=" * 60)

# TreeExplainer: optimized for tree-based models (XGBoost, Random Forest)
# Much faster than generic KernelExplainer
explainer   = shap.TreeExplainer(xgb)
# Compute SHAP on 2000 test samples (full 6000 is slow; 2000 representative)
sample_idx  = np.random.choice(len(X_test), size=2000, replace=False)
X_shap      = pd.DataFrame(X_test, columns=FEATURES).iloc[sample_idx]
shap_values = explainer.shap_values(X_shap)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 7a. Feature importance bar chart (mean |SHAP| across all samples)
mean_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURES).sort_values(ascending=True)
top_n = 12
mean_shap_top = mean_shap.tail(top_n)
axes[0].barh(mean_shap_top.index, mean_shap_top.values, color='#FF5722', edgecolor='white')
axes[0].set_title('Top Feature Importances\n(Mean |SHAP Value|)', fontweight='bold')
axes[0].set_xlabel('Mean |SHAP Value| (impact on default probability)')

# 7b. SHAP beeswarm — each dot = one prediction, color = feature value
# Shows not just WHICH features matter but HOW they affect the score
# (e.g., high credit_utilization → high SHAP → pushed toward default)
plt.sca(axes[1])
top_features = mean_shap.tail(10).index.tolist()
top_idx = [FEATURES.index(f) for f in top_features]
shap.summary_plot(
    shap_values[:, top_idx],
    X_shap[top_features],
    plot_type='dot',
    show=False,
    max_display=10
)
axes[1].set_title('SHAP Beeswarm Plot\n(Red=High Feature Value, Blue=Low)', fontweight='bold')

plt.suptitle('SHAP Explainability — XGBoost Credit Risk Model', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT + 'shap.png', dpi=150, bbox_inches='tight')
plt.close()
print("SHAP charts saved.")

# Top 5 predictors
print("\nTop 5 Predictors of Default (SHAP):")
for i, (feat, val) in enumerate(mean_shap.tail(5).iloc[::-1].items(), 1):
    print(f"  {i}. {feat:<30} mean |SHAP| = {val:.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# 8. SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Dataset:          30,000 loan records | {df['loan_status'].mean()*100:.1f}% default rate")
print(f"Class imbalance:  Fixed with SMOTE (k=5 neighbors)")
print(f"Baseline model:   Logistic Regression  → AUC {lr_auc:.4f} | Gini {2*lr_auc-1:.4f}")
print(f"Production model: XGBoost              → AUC {xgb_auc:.4f} | Gini {2*xgb_auc-1:.4f} | KS {xgb_ks:.4f}")
print(f"Explainability:   SHAP TreeExplainer (top predictors identified)")
print(f"\nAll outputs saved to: {OUTPUT}")
