import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# === Load Data ===
df = pd.read_csv("merged_transaction_with_upi_scores.csv")
df1 = pd.read_csv("synthetic_credit_score_calc.csv")

# Encode 'id' (instead of recipient)
le = LabelEncoder()
df['id'] = le.fit_transform(df['id'])  # make sure 'id' exists in df

# Features and Target
X = df[['id', 'amount_transacted', 'avg_upi_txn_amount']]
y = df['UPI_score']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = XGBRegressor()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"✅ Model trained. RMSE: {rmse:.2f} | R² Score: {r2:.2f}")

# === Prediction Function ===
def predict_score(user_id, amount, avg):
    try:
        encoded_id = int(le.transform([user_id])[0])
    except ValueError:
        return None  # ID not found in training data

    # Prepare input for model
    input_data = pd.DataFrame([{
        'id': encoded_id,
        'amount_transacted': amount,
        'avg_upi_txn_amount': avg
    }])

    input_data = input_data[X.columns]
    input_data = input_data.astype({
        'id': 'int64',
        'amount_transacted': 'float64',
        'avg_upi_txn_amount': 'float64'
    })

    # Base score from model
    base_score = model.predict(input_data)[0] * 0.2

    # Lookup additional weighted info
    row = df1[df1['id'].astype(str) == str(user_id)]
    if row.empty:
        return round(base_score, 4)

    row = row.iloc[0]

    # Final credit score with all weights
    total_score = base_score + (
        row['avg_upi_txn_amount'] * 0.2 +
        row['peer_trust_avg_score'] * 0.2 +
        row['num_verified_endorsers'] * 0.1 +
        row['repayment_success_rate'] * 0.15 +
        row['utility_bill_timeliness_pct'] * 0.15
    )

    return round(total_score, 4)
