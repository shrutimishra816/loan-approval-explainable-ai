"""
api.py — Loan Approval REST API + Frontend
-------------------------------------------
Flask app serving the trained XGBoost loan approval model with
SHAP-based human-readable explanations, plus an interactive web UI.

Endpoints:
  GET  /                → interactive frontend (LoanSight UI)
  POST /predict         → decision + probability + SHAP explanation
  GET  /health          → service health check
  GET  /model-card      → returns model_card.md content
  GET  /api-spec        → returns this API's OpenAPI-style spec

Setup:
  pip install -r requirements.txt
  python train_model.py     # generates loan_model.joblib (run once)
  python api.py
"""

import json
import os
import joblib
import pandas as pd
import shap
from flask import Flask, request, jsonify, render_template

# ── Point Flask at the frontend folder ───────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
FRONTEND   = os.path.join(BASE_DIR, "frontend")

app = Flask(
    __name__,
    template_folder=os.path.join(FRONTEND, "templates"),
    static_folder=os.path.join(FRONTEND, "static"),
)

FEATURE_COLS = [
    "age", "income", "coapplicant_income", "total_income",
    "loan_amount", "loan_term", "monthly_installment",
    "credit_score", "credit_bucket", "employment_yrs",
    "existing_loans", "num_dependents", "debt_to_income",
    "loan_income_ratio", "income_per_dependent",
    "self_employed", "education_grad", "area_urban", "area_semi_urban"
]

REQUIRED_RAW_FIELDS = [
    "age", "income", "coapplicant_income", "loan_amount", "loan_term",
    "credit_score", "employment_yrs", "existing_loans", "num_dependents",
    "self_employed", "education_grad", "area_urban", "area_semi_urban"
]

# ── Load model + explainer once at startup ───────────────────────────────────

MODEL_PATH = os.path.join(BASE_DIR, "loan_model.joblib")

if not os.path.exists(MODEL_PATH):
    print("[INFO] loan_model.joblib not found — training now, please wait...")
    import train_model
    print("[INFO] Training complete.")

try:
    model = joblib.load(MODEL_PATH)
    explainer = shap.TreeExplainer(model)
    MODEL_LOADED = True
except Exception as e:
    model, explainer = None, None
    MODEL_LOADED = False
    print(f"[WARN] Could not load model: {e}")


# ── Feature engineering (must mirror train_model.py) ─────────────────────────

def engineer_features(raw: dict) -> pd.DataFrame:
    total_income = raw["income"] + raw["coapplicant_income"]
    debt_to_income = raw["loan_amount"] / (total_income + 1)
    loan_income_ratio = raw["loan_amount"] / (total_income + 1)
    monthly_installment = raw["loan_amount"] / raw["loan_term"]
    income_per_dependent = total_income / (raw["num_dependents"] + 1)

    cs = raw["credit_score"]
    if cs <= 580:
        credit_bucket = 0
    elif cs <= 670:
        credit_bucket = 1
    elif cs <= 740:
        credit_bucket = 2
    elif cs <= 800:
        credit_bucket = 3
    else:
        credit_bucket = 4

    row = {
        "age": raw["age"],
        "income": raw["income"],
        "coapplicant_income": raw["coapplicant_income"],
        "total_income": total_income,
        "loan_amount": raw["loan_amount"],
        "loan_term": raw["loan_term"],
        "monthly_installment": monthly_installment,
        "credit_score": cs,
        "credit_bucket": credit_bucket,
        "employment_yrs": raw["employment_yrs"],
        "existing_loans": raw["existing_loans"],
        "num_dependents": raw["num_dependents"],
        "debt_to_income": debt_to_income,
        "loan_income_ratio": loan_income_ratio,
        "income_per_dependent": income_per_dependent,
        "self_employed": raw["self_employed"],
        "education_grad": raw["education_grad"],
        "area_urban": raw["area_urban"],
        "area_semi_urban": raw["area_semi_urban"],
    }
    return pd.DataFrame([row])[FEATURE_COLS]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """Serve the LoanSight interactive frontend."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL_LOADED})


@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    raw = request.get_json(force=True)
    missing = [f for f in REQUIRED_RAW_FIELDS if f not in raw]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    X = engineer_features(raw)

    decision = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    shap_vals = explainer.shap_values(X)[0]
    shap_series = pd.Series(shap_vals, index=FEATURE_COLS)
    if decision == 0:
        shap_series = -shap_series

    top_factors = shap_series.sort_values(ascending=False)
    supporting = [{"feature": k, "impact": round(float(v), 4)}
                   for k, v in top_factors.head(3).items() if v > 0]
    against = [{"feature": k, "impact": round(float(v), 4)}
               for k, v in top_factors.tail(3).items() if v < 0]

    return jsonify({
        "decision": "Approved" if decision == 1 else "Rejected",
        "probability_approved": round(probability, 4),
        "explanation": {
            "supporting_factors": supporting,
            "opposing_factors": against
        },
        "input_summary": {
            "credit_score": raw["credit_score"],
            "total_income": raw["income"] + raw["coapplicant_income"],
            "loan_amount": raw["loan_amount"],
            "debt_to_income": round(X.iloc[0]["debt_to_income"], 3)
        }
    })


@app.route("/model-card", methods=["GET"])
def model_card():
    try:
        with open(os.path.join(BASE_DIR, "model_card.md")) as f:
            return f.read(), 200, {"Content-Type": "text/markdown"}
    except FileNotFoundError:
        return jsonify({"error": "model_card.md not found"}), 404


@app.route("/api-spec", methods=["GET"])
def api_spec():
    try:
        with open(os.path.join(BASE_DIR, "api_spec.json")) as f:
            return json.load(f)
    except FileNotFoundError:
        return jsonify({"error": "api_spec.json not found"}), 404


if __name__ == "__main__":
    print("\n LoanSight API + UI running at http://localhost:5002")
    print(" GET  /            — interactive frontend")
    print(" POST /predict     — get decision + SHAP explanation")
    print(" GET  /health      — health check")
    print(" GET  /model-card  — model documentation")
    print(" GET  /api-spec    — API specification\n")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), debug=False)
