---
title: Loan Approval Explainable AI
emoji: 🏦
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
app_port: 7860
pinned: false
---

# LoanSight — Explainable Loan Approval AI

> **89.9% accuracy · AUC-ROC 0.899** on 50,000+ applicants · XGBoost + SHAP · Interactive React-style frontend · Flask REST API

Not just a verdict — the reasons behind it, in plain language your compliance team can audit.

---

## Live Demo

Deployed on Hugging Face Spaces → [ShrutiMishra/loan-approval-explainable-ai](https://huggingface.co/spaces/ShrutiMishra/loan-approval-explainable-ai)

Fill in applicant details, click **Evaluate Application**, and see:
- ✅ / ❌ Decision with approval probability
- Animated probability bar
- SHAP-powered explanation: supporting vs opposing factors
- Input summary (credit score, total income, loan amount, debt-to-income)

---

## Architecture

```
Applicant Data (13 raw fields)
        │
        ▼
  Feature Engineering
  (19 features: ratios, credit buckets, income totals)
        │
        ▼
  XGBoost Classifier
  (trained on 50,000 applicants)
        │
        ├──► Decision: Approved / Rejected
        ├──► Approval Probability (0–100%)
        └──► SHAP Explainer
                  └──► Supporting factors + Opposing factors
                            │
                            ▼
                    Flask REST API (/predict)
                            │
                            ▼
                    Interactive Web UI (LoanSight)
```

---

## Repo Structure

```
loan-approval-explainable-ai/
├── app.py                    # HF Spaces entry point (port 7860)
├── api.py                    # Flask API — /predict, /health, /model-card, /api-spec
├── train_model.py            # Generates data, trains XGBoost, runs SHAP, saves model
├── frontend/
│   ├── templates/index.html  # LoanSight UI (Jinja2)
│   └── static/
│       ├── css/style.css     # Light corporate theme — white/gray + amber
│       └── js/app.js         # Form logic, API calls, SHAP rendering
├── Dockerfile                # For Hugging Face Spaces deployment
├── model_card.md             # Fairness notes, intended use, maintenance plan
├── api_spec.json             # OpenAPI specification
├── evaluation_report.json    # Auto-generated metrics
└── requirements.txt
```

---

## Quickstart (Local)

```bash
pip install -r requirements.txt

# Train the model (runs once — generates loan_model.joblib)
python train_model.py

# Start the server
python api.py
# → http://localhost:5002
```

---

## API

### `POST /predict`

```bash
curl -X POST http://localhost:5002/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35, "income": 600000, "coapplicant_income": 200000,
    "loan_amount": 500000, "loan_term": 36, "credit_score": 720,
    "employment_yrs": 5, "existing_loans": 1, "num_dependents": 2,
    "self_employed": 0, "education_grad": 1, "area_urban": 0, "area_semi_urban": 1
  }'
```

```json
{
  "decision": "Approved",
  "probability_approved": 0.87,
  "explanation": {
    "supporting_factors": [
      { "feature": "credit_score", "impact": 0.312 },
      { "feature": "debt_to_income", "impact": 0.198 }
    ],
    "opposing_factors": [
      { "feature": "existing_loans", "impact": -0.089 }
    ]
  },
  "input_summary": {
    "credit_score": 720,
    "total_income": 800000,
    "loan_amount": 500000,
    "debt_to_income": 0.63
  }
}
```

### Other endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Interactive UI |
| GET | `/health` | Service status |
| GET | `/model-card` | Model documentation |
| GET | `/api-spec` | OpenAPI specification |

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

| Rank | Feature | Direction |
|------|---------|-----------|
| 1 | `debt_to_income` | High ratio → rejection |
| 2 | `credit_score` | High score → approval |
| 3 | `existing_loans` | More loans → rejection |
| 4 | `education_grad` | Graduate → approval |
| 5 | `credit_bucket` | Better band → approval |

---

## Deployment

Runs on **Hugging Face Spaces** (Docker, free tier, no expiration).  
Model trains automatically on first startup — no pre-built joblib needed.

---

## Tech Stack

`Python` · `XGBoost` · `SHAP` · `Flask` · `Scikit-Learn` · `Pandas` · `Vanilla JS` · `Docker`
