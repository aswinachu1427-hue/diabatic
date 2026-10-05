"""
test_app.py
End-to-End Test Suite for Flask Server & ML Inference Pipeline.
"""

import unittest
import json
from app import app, init_db

class TestDiabetesWebApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        init_db()

    def setUp(self):
        self.client = app.test_client()

    def test_01_landing_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'HealthGuard', response.data)
        self.assertIn(b'splash-screen', response.data)
        self.assertIn(b'Predict', response.data)
        self.assertIn(b'diabetes', response.data)
        print("[PASS] Landing page loads successfully with splash screen and hero content.")

    def test_02_model_results_page(self):
        response = self.client.get('/results')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Diagnostic Model Benchmarks', response.data)
        self.assertIn(b'roc_curves.png', response.data)
        self.assertIn(b'confusion_matrix.png', response.data)
        print("[PASS] Model Results page renders comparison matrix and evaluation charts.")

    def test_03_about_page(self):
        response = self.client.get('/about')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Clinical Methodology', response.data)
        self.assertIn(b'Medical Disclaimer', response.data)
        print("[PASS] About page loads clinical documentation and disclaimer.")

    def test_04_auth_login_demo(self):
        response = self.client.post('/login', json={
            'email': 'demo@healthguard.org',
            'password': 'Password123!'
        })
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        print("[PASS] Clinician authentication succeeds for demo account.")

    def test_05_predict_low_risk(self):
        # Authenticate first
        self.client.post('/login', json={'email': 'demo@healthguard.org', 'password': 'Password123!'})

        low_payload = {
            'gender': 'Female',
            'age': 24.0,
            'hypertension': 0,
            'heart_disease': 0,
            'smoking_history': 'never',
            'bmi': 21.0,
            'HbA1c_level': 4.6,
            'blood_glucose_level': 85.0
        }
        response = self.client.post('/predict', json=low_payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['risk_level'], 'Low')
        self.assertLess(data['risk_probability'], 30.0)
        self.assertEqual(len(data['top_factors']), 3)
        self.assertEqual(len(data['suggestions']), 3)
        print(f"[PASS] Low Risk Prediction verified: {data['risk_probability']}% (Level: {data['risk_level']})")

    def test_06_predict_moderate_risk(self):
        self.client.post('/login', json={'email': 'demo@healthguard.org', 'password': 'Password123!'})

        mod_payload = {
            'gender': 'Male',
            'age': 48.0,
            'hypertension': 1,
            'heart_disease': 0,
            'smoking_history': 'former',
            'bmi': 28.5,
            'HbA1c_level': 6.2,
            'blood_glucose_level': 140.0
        }
        response = self.client.post('/predict', json=mod_payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['risk_level'], 'Moderate')
        self.assertTrue(30.0 <= data['risk_probability'] <= 60.0)
        print(f"[PASS] Moderate Risk Prediction verified: {data['risk_probability']}% (Level: {data['risk_level']})")

    def test_07_predict_high_risk(self):
        self.client.post('/login', json={'email': 'demo@healthguard.org', 'password': 'Password123!'})

        high_payload = {
            'gender': 'Male',
            'age': 65.0,
            'hypertension': 1,
            'heart_disease': 1,
            'smoking_history': 'current',
            'bmi': 36.0,
            'HbA1c_level': 7.8,
            'blood_glucose_level': 220.0
        }
        response = self.client.post('/predict', json=high_payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['risk_level'], 'High')
        self.assertGreater(data['risk_probability'], 60.0)
        print(f"[PASS] High Risk Prediction verified: {data['risk_probability']}% (Level: {data['risk_level']})")

    def test_08_server_validation_errors(self):
        self.client.post('/login', json={'email': 'demo@healthguard.org', 'password': 'Password123!'})

        # Out-of-bounds glucose (>300)
        res1 = self.client.post('/predict', json={'gender': 'Female', 'age': 40, 'hypertension': 0, 'heart_disease': 0,
                                                  'smoking_history': 'never', 'bmi': 25, 'HbA1c_level': 5.5, 'blood_glucose_level': 450})
        self.assertEqual(res1.status_code, 422)

        # Out-of-bounds HbA1c (>9.0)
        res2 = self.client.post('/predict', json={'gender': 'Female', 'age': 40, 'hypertension': 0, 'heart_disease': 0,
                                                  'smoking_history': 'never', 'bmi': 25, 'HbA1c_level': 14.0, 'blood_glucose_level': 100})
        self.assertEqual(res2.status_code, 422)
        print("[PASS] Server-side validation catches boundary exceptions with HTTP 422.")

    def test_09_export_csv(self):
        self.client.post('/login', json={'email': 'demo@healthguard.org', 'password': 'Password123!'})
        res = self.client.get('/history/export')
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, 'text/csv')
        self.assertIn(b'Risk Score (%)', res.data)
        print("[PASS] CSV Export endpoint generates structured tabular download.")


if __name__ == '__main__':
    unittest.main()
