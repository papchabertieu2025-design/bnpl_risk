"""
BNPL Instalment Risk - Flask web app and API.

Predicts whether a Buy Now, Pay Later customer will miss their next instalment,
using the logistic-regression model trained by fintech.py (fintech.joblib).

Routes
  GET  /              HTML form
  POST /              form submit -> page with prediction and probability
  GET  /api/predict   JSON API using query-string parameters
  POST /api/predict   JSON API using a JSON body
  GET  /api/health    simple check that the app and model are loaded
"""

import os

import pandas as pd
from flask import Flask, jsonify, render_template, request
from joblib import load

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fintech.joblib")

FEATURES = ["number_of_active_plans", "monthly_income", "outstanding_bnpl_balance"]

# Allowed input ranges (match the synthetic training data)
LIMITS = {
    "number_of_active_plans": (1, 8),
    "monthly_income": (0, 20000),
    "outstanding_bnpl_balance": (0, 10000),
}

app = Flask(__name__)
model = load(MODEL_PATH)


def parse_inputs(source):
    """Read and validate the three model inputs from a dict-like object.

    Returns (values, errors). values is a dict of floats/ints; errors is a list of strings.
    """
    values, errors = {}, []
    for name in FEATURES:
        raw = source.get(name)
        if raw is None or str(raw).strip() == "":
            errors.append(f"'{name}' is required.")
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"'{name}' must be a number.")
            continue
        low, high = LIMITS[name]
        if not (low <= value <= high):
            errors.append(f"'{name}' must be between {low} and {high}.")
            continue
        if name == "number_of_active_plans":
            if value != int(value):
                errors.append("'number_of_active_plans' must be a whole number.")
                continue
            value = int(value)
        values[name] = value
    return values, errors


def risk_band(probability):
    if probability < 0.30:
        return "Low"
    if probability < 0.60:
        return "Medium"
    return "High"


def recommendation(band):
    return {
        "Low": "Approve new BNPL purchases as normal.",
        "Medium": "Approve with caution: send a payment reminder before the due date.",
        "High": "Pause new BNPL plans and offer a payment schedule or support.",
    }[band]


def predict(values):
    x = pd.DataFrame([values], columns=FEATURES)
    probability = float(model.predict_proba(x)[0, 1])
    prediction = int(model.predict(x)[0])
    band = risk_band(probability)
    return {
        "prediction": prediction,
        "prediction_label": "Will miss next instalment" if prediction == 1 else "Will pay on time",
        "probability_missed_instalment": round(probability, 4),
        "probability_pays_on_time": round(1 - probability, 4),
        "risk_band": band,
        "recommendation": recommendation(band),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result, errors, values = None, [], {}
    if request.method == "POST":
        values, errors = parse_inputs(request.form)
        if not errors:
            result = predict(values)
        values = {k: request.form.get(k, "") for k in FEATURES}
    return render_template("index.html", result=result, errors=errors, values=values)


@app.route("/api/predict", methods=["GET", "POST"])
def api_predict():
    if request.method == "POST":
        source = request.get_json(silent=True)
        if not isinstance(source, dict):
            return jsonify({"error": "Send a JSON object in the request body."}), 400
    else:
        source = request.args
    values, errors = parse_inputs(source)
    if errors:
        return jsonify({"error": "Invalid input.", "details": errors}), 400
    return jsonify({"input": values, **predict(values)})


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model": "LogisticRegression", "features": FEATURES})


if __name__ == "__main__":
    app.run(debug=True)
