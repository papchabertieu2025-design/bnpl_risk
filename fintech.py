"""
Fintech problem: predicting missed instalments in Buy Now, Pay Later (BNPL) apps.

Population: adult customers (18+) of a BNPL app who currently have at least one
open purchase split into instalments.
ML task: binary classification (logistic regression).
Target: missed_next_instalment (1 = misses the next instalment, 0 = pays on time).
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Fix the seed so all variables always get the same values
np.random.seed(42)

n_customers = 1000

# 1) Discrete: number of BNPL plans the customer has open right now (1 to 8)
number_of_active_plans = np.random.randint(1, 9, size=n_customers)

# 2) Continuous: monthly net income in euros
monthly_income = np.round(np.random.normal(loc=1800, scale=600, size=n_customers), 2)
monthly_income = np.clip(monthly_income, 500, None)

# 3) Continuous: total outstanding BNPL balance in euros
outstanding_bnpl_balance = np.round(np.random.uniform(50, 3000, size=n_customers), 2)

# Target: missed the next instalment (1) or paid on time (0).
# Risk rises with more open plans and a higher balance, and falls with income.
linear_score = (
    -1.5
    + 0.45 * number_of_active_plans
    - 0.0015 * monthly_income
    + 0.0009 * outstanding_bnpl_balance
)
probability_of_missing = 1 / (1 + np.exp(-linear_score))
missed_next_instalment = np.random.binomial(1, probability_of_missing)

df = pd.DataFrame({
    "number_of_active_plans": number_of_active_plans,
    "monthly_income": monthly_income,
    "outstanding_bnpl_balance": outstanding_bnpl_balance,
    "missed_next_instalment": missed_next_instalment,
})

# Save the data to Excel without the index
df.to_excel("fintech.xlsx", index=False)

# Split into train and test
x = df[["number_of_active_plans", "monthly_income", "outstanding_bnpl_balance"]]
y = df["missed_next_instalment"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42, stratify=y
)

# Train the logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Validate on train and test
print("Classification report - TRAIN")
print(classification_report(y_train, model.predict(x_train)))
print("Classification report - TEST")
print(classification_report(y_test, model.predict(x_test)))

# Dump the model with joblib
joblib.dump(model, "fintech.joblib")
print("Saved fintech.xlsx and fintech.joblib")
