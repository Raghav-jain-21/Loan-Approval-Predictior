from flask import Flask, request, jsonify, render_template, redirect, session, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import pandas as pd
import numpy as np
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import requests
import logging

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'your_secure_secret_key'  # Change this in production
db = SQLAlchemy(app)

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

with app.app_context():
    db.create_all()

# Initialize SQLite database for predictions
def init_db():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  age INTEGER,
                  income INTEGER,
                  emp_exp INTEGER,
                  loan_amnt INTEGER,
                  int_rate REAL,
                  credit_score INTEGER,
                  defaults TEXT,
                  gender TEXT,
                  education TEXT,
                  home_ownership TEXT,
                  loan_intent TEXT,
                  prediction TEXT)''')
    conn.commit()
    conn.close()

init_db()

def is_duplicate_prediction(age, income, emp_exp, loan_amnt, int_rate, credit_score, defaults, gender, education, home_ownership, loan_intent, time_threshold=60):
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute("SELECT timestamp FROM predictions WHERE age=? AND income=? AND emp_exp=? AND loan_amnt=? AND int_rate=? AND credit_score=? AND defaults=? AND gender=? AND education=? AND home_ownership=? AND loan_intent=? ORDER BY timestamp DESC LIMIT 1",
              (age, income, emp_exp, loan_amnt, int_rate, credit_score, defaults, gender, education, home_ownership, loan_intent))
    result = c.fetchone()
    conn.close()
    if result:
        last_timestamp = datetime.strptime(result[0], '%Y-%m-%d %H:%M:%S')
        return (datetime.now() - last_timestamp).total_seconds() < time_threshold
    return False

# Load the trained model
try:
    model = joblib.load('loan_model.pkl')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            return redirect(url_for('home'))
        return render_template('login.html', error="Invalid email or password")
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        if password != confirm_password:
            return render_template('signup.html', error="Passwords do not match")
        if User.query.filter_by(email=email).first():
            return render_template('signup.html', error="Email already exists")
        hashed_password = generate_password_hash(password)
        new_user = User(email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('home'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
def predict():
    response = {}
    try:
        person_age = int(request.form['person_age'])
        person_income = int(request.form['person_income'])
        person_emp_exp = int(request.form['person_emp_exp'])
        loan_amnt = int(request.form['loan_amnt'])
        loan_int_rate = float(request.form['loan_int_rate'])
        credit_score = int(request.form['credit_score'])
        previous_loan_defaults = request.form['previous_loan_defaults']
        person_gender = request.form['person_gender']
        person_education = request.form['person_education']
        person_home_ownership = request.form['person_home_ownership']
        loan_intent = request.form['loan_intent']

        if not (18 <= person_age <= 100):
            raise ValueError("Age must be between 18 and 100")
        if person_income <= 0:
            raise ValueError("Income must be positive")
        if not (300 <= credit_score <= 850):
            raise ValueError("Credit score must be between 300 and 850")
        if loan_amnt <= 0:
            raise ValueError("Loan amount must be positive")
        if not (0 <= loan_int_rate <= 100):
            raise ValueError("Interest rate must be between 0 and 100%")
        if person_emp_exp < 0:
            raise ValueError("Employment experience cannot be negative")

        education_map = {'High School': [1, 0, 0, 0], 'Bachelor': [0, 1, 0, 0], 'Master': [0, 0, 1, 0], 'Doctorate': [0, 0, 0, 1]}
        home_ownership_map = {'Own': [1, 0, 0], 'Rent': [0, 1, 0], 'Other': [0, 0, 1]}
        intent_map = {'Personal': [1, 0, 0, 0, 0], 'Education': [0, 1, 0, 0, 0], 'Medical': [0, 0, 1, 0, 0],
                      'HomeImprovement': [0, 0, 0, 1, 0], 'Venture': [0, 0, 0, 0, 1]}

        education_encoded = education_map[person_education]
        home_ownership_encoded = home_ownership_map[person_home_ownership]
        intent_encoded = intent_map[loan_intent]

        loan_percent_income = loan_amnt / person_income
        cred_hist_length = 3
        input_data = [
            person_age, person_income, person_emp_exp, loan_amnt, loan_int_rate,
            loan_percent_income, cred_hist_length, credit_score, 1 if previous_loan_defaults == 'No' else 0,
            1 if person_gender == 'Male' else 0,
            *education_encoded,
            *home_ownership_encoded,
            *intent_encoded
        ]
        if len(input_data) != 22:
            raise ValueError(f"Expected 22 columns, got {len(input_data)}: {input_data}")

        input_df = pd.DataFrame([input_data], columns=model.feature_names_in_)
        print(f"Input DataFrame columns: {input_df.columns.tolist()}")
        print(f"Input Data: {input_data}")

        probabilities = model.predict_proba(input_df)[0]
        model_prediction = 1 if probabilities[1] > 0.4 else 0
        print(f"Probabilities: {probabilities}, Model Prediction: {model_prediction}")

        final_prediction = model_prediction
        if credit_score < 600 or loan_percent_income > 0.5:
            final_prediction = 0

        major_conditions = []
        if final_prediction == 0:
            if previous_loan_defaults == 'Yes':
                major_conditions.append("Previous loan defaults on file (Yes)")
            if credit_score < 700:
                major_conditions.append("Low credit score (<700)")
            if loan_percent_income > 0.3:
                major_conditions.append("High loan-to-income ratio (>30%)")
        elif final_prediction == 1:
            if previous_loan_defaults == 'No':
                major_conditions.append("No previous loan defaults on file")
            if credit_score >= 700:
                major_conditions.append("Good credit score (>=700)")
            if loan_percent_income <= 0.3:
                major_conditions.append("Low loan-to-income ratio (<=30%)")

        credit_tips = []
        if final_prediction == 0:
            if credit_score < 700:
                credit_tips.append("Increase your credit score by paying down debts.")
            if loan_percent_income > 0.3:
                credit_tips.append("Reduce your loan-to-income ratio by lowering the loan amount or increasing income.")
            if previous_loan_defaults == 'Yes':
                credit_tips.append("Avoid future defaults and maintain a clean credit history.")
        response['credit_tips'] = credit_tips if credit_tips else ["No specific tips at this time."]

        result = 'Approved' if final_prediction == 1 else 'Rejected'

        if not is_duplicate_prediction(person_age, person_income, person_emp_exp, loan_amnt, loan_int_rate, credit_score, previous_loan_defaults, person_gender, person_education, person_home_ownership, loan_intent):
            conn = sqlite3.connect('predictions.db')
            c = conn.cursor()
            c.execute("INSERT INTO predictions (timestamp, age, income, emp_exp, loan_amnt, int_rate, credit_score, defaults, gender, education, home_ownership, loan_intent, prediction) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), person_age, person_income, person_emp_exp, loan_amnt, loan_int_rate, credit_score, previous_loan_defaults, person_gender, person_education, person_home_ownership, loan_intent, result))
            conn.commit()
            conn.close()

        response = {
            'prediction': result,
            'major_conditions': major_conditions if major_conditions else [],
            'credit_tips': credit_tips if credit_tips else ["No specific tips at this time."],
            'error': None
        }
        return jsonify(response)

    except Exception as e:
        print(f"Error in predict: {str(e)}")
        response = {
            'prediction': None,
            'major_conditions': [],
            'credit_tips': ["An error occurred. Please check your input and try again."],
            'error': str(e)
        }
        return jsonify(response), 400

@app.route('/dashboard')
def dashboard():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute("SELECT timestamp, age, income, emp_exp, loan_amnt, int_rate, credit_score, defaults, gender, education, home_ownership, loan_intent, prediction FROM predictions ORDER BY timestamp DESC LIMIT 10")
    predictions = c.fetchall()
    conn.close()

    feature_importance = model.feature_importances_
    feature_names = model.feature_names_in_
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_names, feature_importance)
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance in Loan Approval Prediction')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.subplots_adjust(left=0.3)

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    approved_count = sum(1 for pred in predictions if pred[12] == 'Approved')
    total_count = len(predictions)
    approval_rate = (approved_count / total_count * 100) if total_count > 0 else 0

    return render_template('dashboard.html', predictions=predictions, plot_url=plot_url, approval_rate=approval_rate)

@app.route('/news', methods=['GET'])
def get_news():
    api_key = '5c1c2554935a422cadf2a8bf043909ec' 
    url = f'https://newsapi.org/v2/top-headlines?category=business&language=en&apiKey={api_key}'
    limit = request.args.get('limit', default=20, type=int)
    if limit > 20:
        limit = 20
    try:
        response = requests.get(url, timeout=10)
        print(f"News API Response Status: {response.status_code}")
        print(f"News API Response Text: {response.text}")
        if response.status_code == 200:
            news_data = response.json()
            articles = news_data['articles'][:limit]
            return jsonify({'articles': [{'title': a['title'], 'url': a['url'], 'urlToImage': a.get('urlToImage', 'https://via.placeholder.com/300x200?text=No+Image'), 'publishedAt': a.get('publishedAt', datetime.now().isoformat())} for a in articles]})
        else:
            error_msg = f'Failed to fetch news. Status: {response.status_code}. Details: {response.text}'
            logging.error(error_msg)
            return jsonify({'error': error_msg}), response.status_code
    except requests.RequestException as e:
        error_msg = f'Network error fetching news: {str(e)}'
        logging.error(error_msg)
        return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
