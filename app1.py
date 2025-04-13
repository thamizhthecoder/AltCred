from flask import Flask, render_template, request, redirect, url_for
import firebase_admin
from firebase_admin import credentials, firestore
from ml_model import predict_score
import pandas as pd
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load CSVs
df_credit = pd.read_csv("synthetic_credit_score_calc.csv")
df_txn = pd.read_csv("transaction_details.csv")

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

@app.route('/')
def main():
    return render_template('login.html')

@app.route('/register', methods=['POST'])
def register():
    name = request.form['name']
    account_no = request.form['account_no']
    password = request.form['password']
    db.collection('users').document(account_no).set({
        'name': name,
        'password': password
    })
    return render_template("login.html")

@app.route('/login', methods=['POST'])
def login():
    account_no = request.form['account_no']
    password = request.form['password']
    doc = db.collection('users').document(account_no).get()

    if doc.exists and doc.to_dict()['password'] == password:
        return render_template("index.html")
    elif doc.exists:
        return render_template("login.html", error="Wrong password! Try again")
    else:
        return render_template("login.html", error="No user found! Signup")

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/credit', methods=['GET', 'POST'])
def credit():
    if request.method == 'POST':
        try:
            user_id = request.form['id']
            amount = float(request.form['amount'])
            avg_amount = float(request.form['avg_amount'])

            score = predict_score(user_id, amount, avg_amount)

            return render_template('results.html', user_id=user_id, score=score)

        except Exception as e:
            return render_template('results.html', user_id="Unknown", score=0, error=str(e))

    return render_template('credit.html')


@app.route('/community', methods=['GET', 'POST'])
def community():
    if request.method == 'POST':
        current_user = request.form['currentUserId']
        endorsed_user = request.form['targetUserId']

        # Save screenshots (optional)
        sent_screenshot = request.files['sentScreenshot']
        received_screenshot = request.files['receivedScreenshot']
        if sent_screenshot:
            sent_screenshot.save(os.path.join(app.config['UPLOAD_FOLDER'], sent_screenshot.filename))
        if received_screenshot:
            received_screenshot.save(os.path.join(app.config['UPLOAD_FOLDER'], received_screenshot.filename))

        # Match both forward and reverse transactions
        forward_match = (
            (df_txn['sender'] == current_user) &
            (df_txn['recipient'] == endorsed_user)
        ).any()

        reverse_match = (
            (df_txn['sender'] == endorsed_user) &
            (df_txn['recipient'] == current_user)
        ).any()

        if forward_match and reverse_match:
            # Update num_verified_endorsers in synthetic_credit_score_calc.csv
            credit_df = pd.read_csv("synthetic_credit_score_calc.csv")
            if endorsed_user in credit_df['user_id'].values:
                idx = credit_df[credit_df['user_id'] == endorsed_user].index[0]
                credit_df.at[idx, 'num_verified_endorsers'] += 1
                credit_df.to_csv("synthetic_credit_score_calc.csv", index=False)
                message = f"Endorsement successful! {endorsed_user} has been verified."
            else:
                message = f"Endorsed user ID '{endorsed_user}' not found in credit score data."
        else:
            message = "Verification failed. Matching transactions not found in transaction log."

        return render_template('endorsement_result.html', message=message)

    return render_template('community.html')


if __name__ == '__main__':
    app.run(debug=True)
