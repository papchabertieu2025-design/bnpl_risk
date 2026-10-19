# BNPL Instalment Risk

**Fintech Technology with Impact — group project (credit risk)**

A web app and JSON API that predicts whether a **Buy Now, Pay Later (BNPL)** customer will **miss their next instalment**. It runs on Flask and is deployed on PythonAnywhere.

**Live app:** `https://<your-username>.pythonanywhere.com`

---

## 1. The problem

BNPL apps approve small loans in seconds and let customers stack several instalment plans at once. Customers can fall into arrears before the app notices. This model scores every active customer, so the app can act **before** a payment is missed. It can send a reminder, offer a payment schedule, or pause new plans for high-risk customers.

| Item | Value |
|---|---|
| Population | Adult customers (18+) of a BNPL app with at least one open instalment plan |
| ML task | Binary classification |
| Model | Logistic regression (scikit-learn) |
| Training data | 1,000 synthetic customers generated with NumPy (seed 42), 70/30 train/test split |
| Test accuracy | 76% |

## 2. Variables

| Variable | Type | Description | Allowed range |
|---|---|---|---|
| `number_of_active_plans` | Discrete | BNPL plans the customer has open right now | 1 – 8 |
| `monthly_income` | Continuous | Monthly net income in euros | 0 – 20,000 |
| `outstanding_bnpl_balance` | Continuous | Total amount still owed across all BNPL plans, in euros | 0 – 10,000 |
| **`missed_next_instalment`** | **Target (binary)** | 1 = misses the next instalment, 0 = pays on time | — |

More open plans and a higher balance increase the risk. A higher income lowers it.

**Risk bands** returned by the app: Low (< 30%), Medium (30–60%), High (≥ 60%).

## 3. Project structure

```
bnpl-risk/
├── fintech.py            # Generates synthetic data, trains and saves the model
├── fintech.xlsx          # Synthetic dataset (created by fintech.py)
├── fintech.joblib        # Trained model (created by fintech.py)
├── flask_app.py          # Flask web app + API
├── templates/
│   └── index.html        # HTML form
├── static/
│   └── style.css         # Page styling
├── requirements.txt
└── README.md
```

## 4. API

### `GET /api/predict` or `POST /api/predict`

Receives the three input variables, applies the trained model and returns the prediction and probability as JSON.

**Example request (GET):**

```
https://<your-username>.pythonanywhere.com/api/predict?number_of_active_plans=5&monthly_income=1200&outstanding_bnpl_balance=2200
```

**Example request (POST with curl):**

```bash
curl -X POST https://<your-username>.pythonanywhere.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{"number_of_active_plans": 5, "monthly_income": 1200, "outstanding_bnpl_balance": 2200}'
```

**Example request (Python):**

```python
import requests
r = requests.post(
    "https://<your-username>.pythonanywhere.com/api/predict",
    json={"number_of_active_plans": 5, "monthly_income": 1200, "outstanding_bnpl_balance": 2200},
)
print(r.json())
```

**Example response (200):**

```json
{
  "input": {
    "number_of_active_plans": 5,
    "monthly_income": 1200.0,
    "outstanding_bnpl_balance": 2200.0
  },
  "prediction": 1,
  "prediction_label": "Will miss next instalment",
  "probability_missed_instalment": 0.6921,
  "probability_pays_on_time": 0.3079,
  "risk_band": "High",
  "recommendation": "Pause new BNPL plans and offer a payment schedule or support."
}
```

**Error response (400)**, returned when an input is missing, not a number or out of range:

```json
{
  "error": "Invalid input.",
  "details": ["'number_of_active_plans' must be between 1 and 8."]
}
```

### `GET /api/health`

Returns `{"status": "ok", ...}` if the app and model are loaded.

### `GET /` and `POST /`

The HTML form. Enter the three values, press **Predict risk**, and the page shows the prediction, the probability and a suggested action.

## 5. Run locally

```bash
pip install -r requirements.txt
python fintech.py        # creates fintech.xlsx and fintech.joblib
python flask_app.py      # open http://127.0.0.1:5000
```

## 6. Deploy on PythonAnywhere

1. **Open a Bash console** (Consoles → Bash) and clone the repo:
   ```bash
   git clone https://github.com/<your-github-user>/bnpl-risk.git
   cd bnpl-risk
   ```
2. **Retrain the model on PythonAnywhere.** This builds `fintech.joblib` with the same scikit-learn version the server uses:
   ```bash
   python3 fintech.py
   ```
   (If a package is missing: `pip3 install --user -r requirements.txt`.)
3. **Create the web app:** Web → Add a new web app → Manual configuration → the same Python 3 version you used in step 2 (e.g. `python3 --version`).
4. **Edit the WSGI file** (link on the Web tab). Replace its content with:
   ```python
   import sys
   path = '/home/<your-username>/bnpl-risk'
   if path not in sys.path:
       sys.path.insert(0, path)
   from flask_app import app as application
   ```
5. **Static files** (Web tab, optional): URL `/static/` → directory `/home/<your-username>/bnpl-risk/static`.
6. Click **Reload** and open `https://<your-username>.pythonanywhere.com`.

If the page shows an error, check the **error log** link on the Web tab.

## 7. Notes

- The data is synthetic and was created for teaching. The model is not trained on real customers.
- This repository contains no passwords, API tokens or other credentials.
