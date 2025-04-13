from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import firebase_admin
from firebase_admin import credentials, firestore
from ml_model import predict_score
import pandas as pd
import easyocr
import re

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load credit-related CSV
df = pd.read_csv("synthetic_credit_score_calc.csv")

# Firebase Init
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# EasyOCR Init
reader = easyocr.Reader(['en'])

def clean_email(text):
    return text.replace(" ", "").lower()

def process_image(image_path):
    result = reader.readtext(image_path)
    lines = [detection[1].strip() for detection in result]

    # Extract integer from line 3
    line_3_value = None
    if len(lines) >= 3:
        digits_in_line3 = re.findall(r'\d+', lines[2])
        if digits_in_line3:
            try:
                line_3_value = int(digits_in_line3[0][1:])
            except:
                pass

    # Extract emails
    seen = set()
    email_lines = []
    for line in lines:
        if '@' in line:
            cleaned = clean_email(line)
            if cleaned not in seen:
                seen.add(cleaned)
                email_lines.append(cleaned)

    # Extract transaction ID
    transaction_id = None
    for idx, line in enumerate(lines):
        if 'upi transaction id' in line.lower():
            if idx + 1 < len(lines):
                transaction_id = lines[idx + 1].strip()

    return {
        'line3_int': line_3_value,
        'emails': email_lines,
        'transaction_id': transaction_id
    }

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
    result = None

    if request.method == 'POST':
        current_user_id = request.form.get('currentUserId')
        target_user_id = request.form.get('targetUserId')
        sent_file = request.files.get('sentScreenshot')
        received_file = request.files.get('receivedScreenshot')

        if not (sent_file and received_file):
            result = "❌ Please upload both screenshots."
            return render_template("community.html", result=result)

        sent_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{current_user_id}_sent.png")
        received_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{current_user_id}_received.png")
        sent_file.save(sent_path)
        received_file.save(received_path)

        data1 = process_image(sent_path)
        data2 = process_image(received_path)

        results = []

        if data1['line3_int'] and data2['line3_int']:
            if data1['line3_int'] == data2['line3_int']:
                results.append(f"✅ Amount Match: ₹{data1['line3_int']}")
            else:
                results.append(f"❌ Amount Mismatch: ₹{data1['line3_int']} vs ₹{data2['line3_int']}")
        else:
            results.append("❌ Amount not detected.")

        if len(data1['emails']) >= 2 and len(data2['emails']) >= 2:
            if data1['emails'][1] == data2['emails'][0] and data1['emails'][0] == data2['emails'][1]:
                results.append("✅ Emails match.")
            else:
                results.append("❌ Emails do not match.")
        else:
            results.append("❌ Insufficient email data.")

        if data1['transaction_id'] == data2['transaction_id']:
            results.append("✅ UPI Transaction ID matched.")
        else:
            results.append("❌ UPI Transaction ID mismatch.")

        result = "<br>".join(results)

    return render_template("community.html", result=result)

if __name__ == '__main__':
    app.run(debug=True)
