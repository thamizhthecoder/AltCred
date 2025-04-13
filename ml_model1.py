import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import numpy as np


# Load data
df = pd.read_csv("merged_transaction_with_upi_scores.csv")
print("Initial columns:", df.columns)

# Encode 'recipient'
le = LabelEncoder()
df['recipient'] = le.fit_transform(df['recipient'])

# Define features and target
X = df[['recipient', 'amount_transacted', 'avg_upi_txn_amount']]
y = df['UPI_score']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBRegressor()
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred)  # Using root mean squared error
r2 = r2_score(y_test, y_pred)


# === Manual Input Test ===
# Use a valid recipient name from the list above
sample_recipient_name = le.classes_[0]  # or use a specific name from the list
encoded_recipient = le.transform([sample_recipient_name])[0]

    
def predict_score(recipient,amount,avg):
    manual_input = pd.DataFrame([{
    'recipient': encoded_recipient,
    'amount_transacted': amount,
    'avg_upi_txn_amount': avg
    }])



    manual_input = manual_input[X.columns]  # ensure feature order matches

    manual_pred = model.predict(manual_input)

    return round(manual_pred[0], 4)