# Model Card — Loan Approval Prediction System

## Overview
| | |
|---|---|
| **Model type** | XGBoost binary classifier |
| **Task** | Predict loan approval (Approved / Rejected) |
| **Training data** | 50,000 synthetic applicant records (mirrors real lending dataset distributions) |
| **Features** | 19 engineered features (12 core + derived ratios) |
| **Accuracy** | 89.9% |
| **AUC-ROC** | 0.899 |
| **Explainability** | SHAP (TreeExplainer) |
| **Deployment** | Flask REST API (`/predict`) |

---

## Intended Use
This model assists loan officers and compliance teams by providing a **preliminary decision recommendation with full explainability**. It is designed to be:
- **Human-readable**: every decision includes the top factors that drove it
- **Audit-ready**: SHAP values are logged per prediction
- **Non-final**: outputs are recommendations, not automated final decisions

## Out of Scope
- Not a substitute for regulatory fair-lending review
- Not trained on real customer data — synthetic data approximates real-world distributions for demonstration
- Should not be used as the sole basis for adverse action without human review

---

## Features Used

| Feature | Description |
|---------|-------------|
| `age` | Applicant age |
| `income` | Primary applicant monthly income |
| `coapplicant_income` | Co-applicant monthly income |
| `total_income` | income + coapplicant_income |
| `loan_amount` | Requested loan principal |
| `loan_term` | Loan term in months |
| `monthly_installment` | loan_amount / loan_term |
| `credit_score` | Credit score (300–850) |
| `credit_bucket` | Binned credit score (0–4) |
| `employment_yrs` | Years of employment |
| `existing_loans` | Number of existing loans |
| `num_dependents` | Number of dependents |
| `debt_to_income` | loan_amount / total_income |
| `loan_income_ratio` | loan_amount / total_income |
| `income_per_dependent` | total_income / (dependents + 1) |
| `self_employed` | 0/1 |
| `education_grad` | 0/1 — graduate education |
| `area_urban` | 0/1 |
| `area_semi_urban` | 0/1 |

---

## Performance

```
Accuracy  : 89.9%
AUC-ROC   : 0.899

              precision    recall  f1-score   support
    Rejected       0.92      0.97      0.94      8605
    Approved       0.70      0.48      0.57      1395
```

Full metrics in `evaluation_report.json`.

---

## Top Drivers (SHAP Global Importance)

1. `debt_to_income` — largest driver of rejection
2. `credit_score` — primary positive driver of approval
3. `existing_loans` — higher count increases rejection likelihood
4. `education_grad` — small positive effect
5. `credit_bucket` — reinforces credit_score signal

---

## Fairness & Compliance Notes

- The model does **not** use protected attributes (gender, religion, caste, etc.) as direct features
- `area_urban` / `area_semi_urban` (proxy for geography) is retained because property location is a standard underwriting factor, but should be monitored for disparate impact in production
- SHAP explanations are surfaced per-decision so compliance teams can audit individual outcomes, not just aggregate metrics
- Recommended: run periodic fairness audits (e.g., disparate impact ratio) segmented by demographic groups before production use

---

## Maintenance

- Retrain quarterly on updated applicant data
- Monitor for feature drift (especially `credit_score` distribution and `debt_to_income`)
- Re-run SHAP analysis after each retrain to confirm top drivers remain stable
