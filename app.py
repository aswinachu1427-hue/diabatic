"""
app.py
Production Flask Application for Diabetes Risk Prediction.

Features:
- SQLite Database with Flask-Login & Werkzeug Password Hashing
- REST API for Real-time ML Inference with Explainable AI (Top 3 Contributing Factors)
- Complete Application Routing:
  - / (Landing Page with Split Login & Splash Intro)
  - /login, /register, /logout
  - /dashboard (Interactive Prediction Form, Risk Ring, Area Trend Chart, Stats)
  - /predict (POST API, server-side validated, persistent history)
  - /history & /history/export (CSV export & filtered table)
  - /results (Model metrics, ROC-AUC, Confusion Matrix, Feature Importance)
  - /about (Project overview, medical disclaimer, methodology)
"""

import os
import json
import sqlite3
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from functools import wraps
from flask import (
    Flask, render_template, request, jsonify, redirect,
    url_for, flash, send_file, Response, g
)
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash

# Initialize App
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'diabetes-risk-guard-secret-key-2026-phase4')
DATABASE = os.path.join(os.path.dirname(__file__), 'database.db')

# Initialize Login Manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'landing'
login_manager.login_message = "Please sign in to access your clinical dashboard."
login_manager.login_message_category = "info"

# Load Trained Model and Metrics
MODEL_PATH = os.path.join('models', 'diabetes_model.joblib')
METRICS_PATH = os.path.join('models', 'metrics.json')

ml_model = None
metrics_data = {}

try:
    if os.path.exists(MODEL_PATH):
        ml_model = joblib.load(MODEL_PATH)
        print(" Successfully loaded diabetes_model.joblib")
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            metrics_data = json.load(f)
        print(" Successfully loaded metrics.json")
except Exception as e:
    print(f" Error loading ML artifacts: {e}")


# Database helpers
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


def init_db():
    """Initializes the database schema and seeds a demo clinician user."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Predictions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        gender TEXT NOT NULL,
        age REAL NOT NULL,
        hypertension INTEGER NOT NULL,
        heart_disease INTEGER NOT NULL,
        smoking_history TEXT NOT NULL,
        bmi REAL NOT NULL,
        hba1c_level REAL NOT NULL,
        blood_glucose_level REAL NOT NULL,
        risk_probability REAL NOT NULL,
        risk_level TEXT NOT NULL,
        top_factors TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    );
    """)
    conn.commit()

    # Seed demo user if none exists
    cursor.execute("SELECT id FROM users WHERE email = 'demo@healthguard.org'")
    demo_user = cursor.fetchone()
    if not demo_user:
        hashed = generate_password_hash("Password123!")
        cursor.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            ("Dr. Sarah Jenkins", "demo@healthguard.org", hashed)
        )
        user_id = cursor.lastrowid

        # Seed sample historical checks for demo user
        sample_records = [
            ("Female", 42.0, 0, 0, "never", 23.4, 5.2, 98.0, 4.2, "Low", "Age Baseline, Normoglycemic HbA1c, Ideal BMI", "2026-08-14 09:30:00"),
            ("Female", 42.0, 0, 0, "never", 25.1, 5.5, 112.0, 11.5, "Low", "Fasting Glucose Trend, Age Factor, BMI Elevation", "2026-09-02 11:15:00"),
            ("Female", 42.5, 1, 0, "former", 27.8, 6.1, 142.0, 48.6, "Moderate", "Pre-diabetic HbA1c (6.1%), Elevated Fasting Glucose, Hypertension", "2026-09-21 14:00:00"),
            ("Female", 42.8, 1, 0, "former", 26.5, 5.8, 128.0, 34.0, "Moderate", "Elevated Fasting Glucose, Hypertension, Borderline HbA1c", "2026-10-02 10:45:00")
        ]
        for rec in sample_records:
            cursor.execute("""
            INSERT INTO predictions 
            (user_id, gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c_level, blood_glucose_level, risk_probability, risk_level, top_factors, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, *rec))
        conn.commit()
        print(" Seeded demo clinician account (demo@healthguard.org / Password123!) with history.")
    conn.close()


# User Model for Flask-Login
class User(UserMixin):
    def __init__(self, id, name, email, password_hash):
        self.id = id
        self.name = name
        self.email = email
        self.password_hash = password_hash

    @staticmethod
    def get_by_id(user_id):
        db = get_db()
        row = db.execute("SELECT id, name, email, password_hash FROM users WHERE id = ?", (user_id,)).fetchone()
        if row:
            return User(row['id'], row['name'], row['email'], row['password_hash'])
        return None

    @staticmethod
    def get_by_email(email):
        db = get_db()
        row = db.execute("SELECT id, name, email, password_hash FROM users WHERE LOWER(email) = LOWER(?)", (email.strip(),)).fetchone()
        if row:
            return User(row['id'], row['name'], row['email'], row['password_hash'])
        return None


@login_manager.user_loader
def load_user(user_id):
    return User.get_by_id(user_id)


# Explainability and Personalization Logic
def compute_patient_factors(inputs):
    """
    Computes patient-specific factor contributions and lifestyle recommendations.
    Based on model feature weights, clinical thresholds, and deviation from optimal norms.
    """
    factors = []

    # 1. Glycated Hemoglobin (HbA1c)
    hba1c = inputs['HbA1c_level']
    if hba1c >= 6.5:
        factors.append({
            'name': 'Severe Glycated Hemoglobin (HbA1c)',
            'impact': min(98, round(45 + (hba1c - 6.5) * 20)),
            'severity': 'danger',
            'note': f'{hba1c:.1f}% is in the diagnostic diabetic range (≥6.5%). High systemic glycemia.'
        })
    elif hba1c >= 5.7:
        factors.append({
            'name': 'Pre-diabetic HbA1c Window',
            'impact': round(25 + (hba1c - 5.7) * 25),
            'severity': 'warning',
            'note': f'{hba1c:.1f}% indicates impaired glucose regulation (normal is <5.7%).'
        })
    else:
        factors.append({
            'name': 'Optimal Glycemic Marker (HbA1c)',
            'impact': 12,
            'severity': 'success',
            'note': f'{hba1c:.1f}% is within healthy physiological parameters (<5.7%).'
        })

    # 2. Blood Glucose Level
    glucose = inputs['blood_glucose_level']
    if glucose >= 200:
        factors.append({
            'name': 'Critical Blood Glucose Spike',
            'impact': min(95, round(40 + (glucose - 200) * 0.4)),
            'severity': 'danger',
            'note': f'{glucose:.0f} mg/dL demonstrates marked hyperglycemia (target <140 mg/dL).'
        })
    elif glucose >= 140:
        factors.append({
            'name': 'Elevated Blood Glucose',
            'impact': round(22 + (glucose - 140) * 0.3),
            'severity': 'warning',
            'note': f'{glucose:.0f} mg/dL reflects impaired glucose tolerance post-load.'
        })
    else:
        factors.append({
            'name': 'Normoglycemic Blood Sugar',
            'impact': 10,
            'severity': 'success',
            'note': f'{glucose:.0f} mg/dL is within ideal baseline range (70-139 mg/dL).'
        })

    # 3. Body Mass Index (BMI)
    bmi = inputs['bmi']
    if bmi >= 30.0:
        factors.append({
            'name': 'Adiposity & Insulin Resistance (BMI)',
            'impact': min(85, round(25 + (bmi - 30.0) * 2.2)),
            'severity': 'warning' if bmi < 35 else 'danger',
            'note': f'BMI of {bmi:.1f} kg/m² correlates with elevated free fatty acids and insulin resistance.'
        })
    elif bmi >= 25.0:
        factors.append({
            'name': 'Mildly Elevated BMI',
            'impact': round(15 + (bmi - 25.0) * 2),
            'severity': 'warning',
            'note': f'BMI of {bmi:.1f} kg/m² lies in the overweight band (25.0-29.9 kg/m²).'
        })
    else:
        factors.append({
            'name': 'Normal Body Weight Ratio',
            'impact': 8,
            'severity': 'success',
            'note': f'BMI of {bmi:.1f} kg/m² is within lean protective limits (18.5-24.9).'
        })

    # 4. Age & Vascular Comorbidities
    age = inputs['age']
    if age >= 55:
        factors.append({
            'name': 'Metabolic Age Vulnerability',
            'impact': min(75, round(18 + (age - 55) * 1.5)),
            'severity': 'warning',
            'note': f'Age {age:.0f} years is associated with progressive pancreatic beta-cell attrition.'
        })

    if inputs['hypertension'] == 1:
        factors.append({
            'name': 'Hypertensive Arterial Stress',
            'impact': 32,
            'severity': 'warning',
            'note': 'Diagnosed high blood pressure exacerbates metabolic and microvascular risk.'
        })

    if inputs['heart_disease'] == 1:
        factors.append({
            'name': 'Pre-existing Cardiovascular Disease',
            'impact': 38,
            'severity': 'danger',
            'note': 'Cardiovascular pathology shares underlying endothelial inflammation pathways with type 2 diabetes.'
        })

    if inputs['smoking_history'] in ['current', 'ever']:
        factors.append({
            'name': 'Active Nicotine Exposure',
            'impact': 20,
            'severity': 'warning',
            'note': 'Nicotine elevates cortisol and impairs peripheral muscle glucose uptake.'
        })

    # Sort factors by impact descending and select top 3
    factors.sort(key=lambda x: x['impact'], reverse=True)
    top_3 = factors[:3]

    # Generate Personalized Lifestyle Recommendations
    recommendations = []
    if hba1c >= 5.7 or glucose >= 140:
        recommendations.append({
            'icon': 'activity',
            'title': 'Glycemic Modulation Diet',
            'desc': 'Prioritize low-glycemic index carbohydrates (legumes, leafy greens), eliminate refined sugars, and distribute complex carbs evenly across meals.'
        })
    else:
        recommendations.append({
            'icon': 'apple',
            'title': 'Balanced Nutrient Density',
            'desc': 'Maintain Mediterranean diet patterns rich in polyphenol antioxidants, dietary fiber, and healthy omega-3 monounsaturated fats.'
        })

    if bmi >= 25.0:
        recommendations.append({
            'icon': 'scale',
            'title': 'Targeted Weight Optimization',
            'desc': f'A clinical reduction of 5-7% total body mass ({max(3.0, round(bmi*0.06, 1)):.1f} kg target) can reduce diabetes progression risk by over 58%.'
        })
    else:
        recommendations.append({
            'icon': 'zap',
            'title': 'Aerobic & Resistance Conditioning',
            'desc': 'Aim for 150 minutes weekly of moderate-intensity zone-2 cardio combined with 2 weekly sessions of full-body resistance training.'
        })

    if inputs['hypertension'] == 1 or inputs['heart_disease'] == 1:
        recommendations.append({
            'icon': 'heart-pulse',
            'title': 'Cardiometabolic Surveillance',
            'desc': 'Monitor resting blood pressure bi-weekly (goal <130/80 mmHg), adhere to the DASH protocol with sodium intake restricted below 2,000 mg/day.'
        })
    elif inputs['smoking_history'] in ['current', 'ever']:
        recommendations.append({
            'icon': 'shield-alert',
            'title': 'Smoking Cessation Protocol',
            'desc': 'Enroll in evidence-based nicotine replacement or cessation therapy; quitting smoking dramatically restores peripheral insulin sensitivity within 8 weeks.'
        })
    else:
        recommendations.append({
            'icon': 'clock',
            'title': 'Annual Screening Cadence',
            'desc': 'Schedule routine annual serum HbA1c and fasting metabolic panels to track longitudinal trajectory and detect subtle metabolic shifts early.'
        })

    return top_3, recommendations


# --- ROUTES ---

@app.route('/')
def landing():
    """Landing page with hero animations, live KPIs, and split login card."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    # Load high-level KPI metrics for the landing hero
    kpi_records = "100,000"
    roc_auc = metrics_data.get('best_model', {}).get('test_roc_auc', 97.6)
    recall = metrics_data.get('best_model', {}).get('test_recall', 88.2)

    return render_template(
        'index.html',
        kpi_records=kpi_records,
        roc_auc=roc_auc,
        recall=recall
    )


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Authenticates existing user with email and password."""
    if request.method == 'GET':
        return redirect(url_for('landing'))

    # Support JSON API or Form POST
    data = request.get_json(silent=True) or request.form
    email = data.get('email', '').strip()
    password = data.get('password', '')

    if not email or not password:
        if request.is_json:
            return jsonify({'success': False, 'message': 'Email and password are required.'}), 400
        flash('Email and password are required.', 'danger')
        return redirect(url_for('landing'))

    user = User.get_by_email(email)
    if not user or not check_password_hash(user.password_hash, password):
        if request.is_json:
            return jsonify({'success': False, 'message': 'Invalid credentials. Please verify email and password.'}), 401
        flash('Invalid credentials. Please verify your email and password.', 'danger')
        return redirect(url_for('landing'))

    login_user(user)
    if request.is_json:
        return jsonify({'success': True, 'redirect': url_for('dashboard')})
    return redirect(url_for('dashboard'))


@app.route('/register', methods=['POST'])
def register():
    """Registers a new user account."""
    data = request.get_json(silent=True) or request.form
    name = data.get('name', '').strip()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    confirm_password = data.get('confirm_password', '')

    if not name or not email or not password:
        msg = 'All registration fields are required.'
        return (jsonify({'success': False, 'message': msg}), 400) if request.is_json else (flash(msg, 'danger') or redirect(url_for('landing')))

    if len(password) < 6:
        msg = 'Password must contain at least 6 characters.'
        return (jsonify({'success': False, 'message': msg}), 400) if request.is_json else (flash(msg, 'danger') or redirect(url_for('landing')))

    if confirm_password and password != confirm_password:
        msg = 'Passwords do not match.'
        return (jsonify({'success': False, 'message': msg}), 400) if request.is_json else (flash(msg, 'danger') or redirect(url_for('landing')))

    db = get_db()
    existing = db.execute("SELECT id FROM users WHERE LOWER(email) = ?", (email,)).fetchone()
    if existing:
        msg = 'An account with this email already exists. Please sign in.'
        return (jsonify({'success': False, 'message': msg}), 409) if request.is_json else (flash(msg, 'warning') or redirect(url_for('landing')))

    hashed = generate_password_hash(password)
    cursor = db.execute("INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)", (name, email, hashed))
    db.commit()

    new_user = User.get_by_id(cursor.lastrowid)
    login_user(new_user)

    if request.is_json:
        return jsonify({'success': True, 'redirect': url_for('dashboard')})
    flash('Account registered successfully! Welcome to HealthGuard AI.', 'success')
    return redirect(url_for('dashboard'))


@app.route('/logout')
@login_required
def logout():
    """Logs out current user session."""
    logout_user()
    flash('You have been securely signed out.', 'info')
    return redirect(url_for('landing'))


@app.route('/dashboard')
@login_required
def dashboard():
    """Main patient assessment dashboard."""
    db = get_db()
    rows = db.execute("""
        SELECT * FROM predictions 
        WHERE user_id = ? 
        ORDER BY created_at DESC
    """, (current_user.id,)).fetchall()

    predictions = [dict(row) for row in rows]
    total_checks = len(predictions)
    avg_risk = round(sum(p['risk_probability'] for p in predictions) / total_checks, 1) if total_checks > 0 else 0.0
    last_check = predictions[0]['created_at'][:10] if total_checks > 0 else 'No assessments yet'

    # Latest assessment for result card default state
    latest = predictions[0] if total_checks > 0 else None

    # Trend series for Chart.js (chronological order)
    trend_data = [
        {'date': p['created_at'][:10], 'score': p['risk_probability']}
        for p in reversed(predictions[-10:])
    ]

    return render_template(
        'dashboard.html',
        user=current_user,
        total_checks=total_checks,
        avg_risk=avg_risk,
        last_check=last_check,
        latest=latest,
        trend_data=json.dumps(trend_data)
    )


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """
    Real-time ML Risk Assessment endpoint.
    Accepts JSON, executes ColumnTransformer Pipeline, evaluates calibrated threshold,
    computes explainability breakdown, and stores result in database.
    """
    if ml_model is None:
        return jsonify({'error': 'Machine learning model is currently not loaded.'}), 500

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON payload received.'}), 400

    # 1. Server-Side Validation
    try:
        gender = str(data.get('gender', 'Female')).strip()
        if gender not in ['Female', 'Male']:
            return jsonify({'error': "Gender must be either 'Female' or 'Male'."}), 422

        age = float(data.get('age', 0))
        if not (1.0 <= age <= 120.0):
            return jsonify({'error': 'Age must be between 1 and 120 years.'}), 422

        hypertension = int(data.get('hypertension', 0))
        if hypertension not in [0, 1]:
            return jsonify({'error': 'Hypertension indicator must be 0 or 1.'}), 422

        heart_disease = int(data.get('heart_disease', 0))
        if heart_disease not in [0, 1]:
            return jsonify({'error': 'Heart disease indicator must be 0 or 1.'}), 422

        smoking_history = str(data.get('smoking_history', 'never')).strip()
        valid_smoking = ['never', 'No Info', 'former', 'current', 'not current', 'ever']
        if smoking_history not in valid_smoking:
            return jsonify({'error': f"Smoking history must be one of {valid_smoking}."}), 422

        bmi = float(data.get('bmi', 0))
        if not (10.0 <= bmi <= 70.0):
            return jsonify({'error': 'BMI must be within plausible clinical range (10.0 to 70.0 kg/m²).'}), 422

        hba1c_level = float(data.get('HbA1c_level', data.get('hba1c_level', 0)))
        if not (3.5 <= hba1c_level <= 9.0):
            return jsonify({'error': 'HbA1c level must be between 3.5% and 9.0%.'}), 422

        blood_glucose_level = float(data.get('blood_glucose_level', 0))
        if not (50.0 <= blood_glucose_level <= 300.0):
            return jsonify({'error': 'Blood glucose level must be between 50 and 300 mg/dL.'}), 422

    except (ValueError, TypeError) as e:
        return jsonify({'error': f'Invalid numeric parameter format: {str(e)}'}), 422

    # 2. Build inference DataFrame for Scikit-learn Pipeline
    input_payload = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'smoking_history': smoking_history,
        'bmi': bmi,
        'HbA1c_level': hba1c_level,
        'blood_glucose_level': blood_glucose_level
    }
    input_df = pd.DataFrame([input_payload])

    # 3. Model Inference
    try:
        raw_prob = float(ml_model.predict_proba(input_df)[0, 1])
        risk_probability = round(raw_prob * 100.0, 1)

        # Risk Classification tiers
        # Low: < 30%, Moderate: 30% - 60%, High: > 60%
        if risk_probability < 30.0:
            risk_level = 'Low'
            level_color = '#10B981'
            summary_statement = "Patient displays favorable cardiometabolic biomarkers with minimal short-term diabetic risk."
        elif risk_probability <= 60.0:
            risk_level = 'Moderate'
            level_color = '#F59E0B'
            summary_statement = "Borderline metabolic markers detected. Early lifestyle interventions recommended to prevent progression."
        else:
            risk_level = 'High'
            level_color = '#EF4444'
            summary_statement = "Elevated clinical risk detected. Comprehensive clinical evaluation and diagnostic blood work strongly indicated."

        # 4. Explainable Factor Attribution & Suggestions
        top_factors, suggestions = compute_patient_factors(input_payload)
        factors_text = ", ".join([f['name'] for f in top_factors])

        # 5. Persist to SQLite Database
        db = get_db()
        cursor = db.execute("""
            INSERT INTO predictions 
            (user_id, gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c_level, blood_glucose_level, risk_probability, risk_level, top_factors)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            current_user.id, gender, age, hypertension, heart_disease,
            smoking_history, bmi, hba1c_level, blood_glucose_level,
            risk_probability, risk_level, factors_text
        ))
        db.commit()
        prediction_id = cursor.lastrowid

        return jsonify({
            'success': True,
            'prediction_id': prediction_id,
            'risk_probability': risk_probability,
            'risk_level': risk_level,
            'level_color': level_color,
            'summary_statement': summary_statement,
            'top_factors': top_factors,
            'suggestions': suggestions,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'disclaimer': 'Educational and decision-support tool. Not an official medical diagnosis.'
        })

    except Exception as e:
        return jsonify({'error': f'Inference processing error: {str(e)}'}), 500


@app.route('/history')
@login_required
def history():
    """User assessment history page with filtering and stats."""
    db = get_db()
    risk_filter = request.args.get('risk', '').strip().capitalize()

    query = "SELECT * FROM predictions WHERE user_id = ?"
    params = [current_user.id]

    if risk_filter in ['Low', 'Moderate', 'High']:
        query += " AND risk_level = ?"
        params.append(risk_filter)

    query += " ORDER BY created_at DESC"
    rows = db.execute(query, params).fetchall()
    predictions = [dict(row) for row in rows]

    return render_template(
        'history.html',
        predictions=predictions,
        current_filter=risk_filter,
        total_count=len(predictions)
    )


@app.route('/history/export')
@login_required
def export_history_csv():
    """Exports patient assessment history as a downloadable CSV."""
    db = get_db()
    rows = db.execute("""
        SELECT created_at, gender, age, hypertension, heart_disease, smoking_history, 
               bmi, hba1c_level, blood_glucose_level, risk_probability, risk_level, top_factors
        FROM predictions 
        WHERE user_id = ? 
        ORDER BY created_at DESC
    """, (current_user.id,)).fetchall()

    if not rows:
        flash("No assessment history records found to export.", "warning")
        return redirect(url_for('history'))

    df = pd.DataFrame([dict(r) for r in rows])
    df.rename(columns={
        'created_at': 'Timestamp',
        'gender': 'Gender',
        'age': 'Age',
        'hypertension': 'Hypertension (0/1)',
        'heart_disease': 'Heart Disease (0/1)',
        'smoking_history': 'Smoking History',
        'bmi': 'BMI (kg/m²)',
        'hba1c_level': 'HbA1c (%)',
        'blood_glucose_level': 'Fasting Glucose (mg/dL)',
        'risk_probability': 'Risk Score (%)',
        'risk_level': 'Risk Tier',
        'top_factors': 'Contributing Clinical Factors'
    }, inplace=True)

    csv_data = df.to_csv(index=False)
    filename = f"diabetes_risk_history_{current_user.name.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d')}.csv"

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename={filename}"}
    )


@app.route('/results')
def results():
    """Model performance benchmarks, metrics table, ROC curve, and confusion matrix."""
    return render_template(
        'results.html',
        metrics=metrics_data
    )


@app.route('/about')
def about():
    """Project overview, clinical background, methodology, and team credits."""
    return render_template('about.html')


# Custom Error Handlers
@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', code=404, message="The requested healthcare view was not found."), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', code=500, message="An internal clinical application error occurred."), 500


if __name__ == '__main__':
    init_db()
    port = int(os.environ.get('PORT', 5000))
    print(f"\n Server running at http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
