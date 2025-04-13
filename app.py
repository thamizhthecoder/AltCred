import pandas as pd

from flask import Flask, render_template, request, redirect, url_for
import firebase_admin
from firebase_admin import credentials, firestore
from ml_model import predict_score  # ✅ import the function
app = Flask(__name__)

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


df1 = pd.read_csv("synthetic_credit_score_calc.csv")

@app.route('/')
def main():
    return render_template('login.html')

@app.route('/register', methods=['POST'])
def register():
    name = request.form['name']
    account_no = request.form['account_no']
    password = request.form['password']

    # Save to Firebase DB
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
    if doc.exists:
        return render_template("login.html",error="Wrong password! Try again")
    else:
        return render_template("login.html",error="No user found!  Signup")

@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/credit',methods=['GET','POST'])
def credit():
    if request.method == 'POST':
        recipient = request.form['recipient']
        amount = float(request.form['amount'])
        avg = float(request.form['avg_amount'])

        score = predict_score(recipient,amount,avg)

        message = f"Credit score:{score}"
    
    return render_template('credit.html')


if __name__ == '__main__':
    app.run(debug=True)
