# Explainable Loan AI — Built for Compliance Ops

> **89.9% accuracy · AUC-ROC 0.899** on 50,000+ applicants · 19 engineered features · SHAP explainability · Flask REST API with full model card and API spec

---

## What This Does

Predicts loan approval decisions and explains **why** in plain English — so compliance and ops teams can audit every decision without needing ML expertise.

```
Applicant Data (13 raw fields)
        │
        ▼
  Feature Engineering (19 features: ratios, buckets, totals)
        │
        ▼
  XGBoost Classifier
        │
        ├──► Decision: Approved / Rejected
        │
        ├──► SHAP Explainer
        │         │
        │         └──► "Factors supporting this decision" + "Factors working against"
        │
        └──► Flask REST API (/predict)
                    │
                    └──► JSON response: decision + probability + explanation
```

---

## Repo Structure

```
loan-approval-explainable-ai/
├── train_model.py          # Generates data, trains XGBoost, runs SHAP, saves model
├── api.py                   # Flask REST API — /predict, /health, /model-card, /api-spec
├── model_card.md            # Full model documentation (fairness, performance, maintenance)
├── api_spec.json            # OpenAPI specification
├── evaluation_report.json   # Auto-generated metrics (accuracy, AUC, per-class report)
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
pip install -r requirements.txt

# 1. Train the model (generates loan_model.joblib + evaluation_report.json)
python train_model.py

# 2. Start the API
python api.py
```

### Example request
```bash
curl -X POST http://localhost:5002/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35, "income": 60000, "coapplicant_income": 20000,
    "loan_amount": 500000, "loan_term": 36, "credit_score": 720,
    "employment_yrs": 5, "existing_loans": 1, "num_dependents": 2,
    "self_employed": 0, "education_grad": 1, "area_urban": 1, "area_semi_urban": 0
  }'
```

### Example response
```json
{
  "decision": "Rejected",
  "probability_approved": 0.4213,
  "explanation": {
    "supporting_factors": [
      {"feature": "coapplicant_income", "impact": 0.0886},
      {"feature": "income", "impact": 0.0282}
    ],
    "opposing_factors": [
      {"feature": "existing_loans", "impact": -0.4903},
      {"feature": "debt_to_income", "impact": -0.346}
    ]
  },
  "input_summary": {
    "credit_score": 720,
    "total_income": 80000,
    "loan_amount": 500000,
    "debt_to_income": 6.25
  }
}
```

---

## Performance

| Metric | Value |
|--------|-------|
| Accuracy | 89.9% |
| AUC-ROC | 0.899 |
| Training set | 40,000 applicants |
| Test set | 10,000 applicants |
| Features | 19 (12 raw + 7 engineered) |

Full per-class breakdown in `evaluation_report.json`.

---

## Top SHAP Drivers

1. **debt_to_income** — strongest predictor of rejection
2. **credit_score** — strongest predictor of approval
3. **existing_loans** — more existing loans → higher rejection likelihood
4. **education_grad**
5. **credit_bucket**

---

## Documentation

- **`model_card.md`** — fairness notes, intended use, maintenance plan
- **`api_spec.json`** — OpenAPI spec for the `/predict` endpoint
- **`evaluation_report.json`** — full classification report, auto-generated on training

---

## Tech Stack
`Python` · `XGBoost` · `SHAP` · `Flask` · `Scikit-Learn` · `Pandas`
