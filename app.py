from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load Model
try:
    with open('model.pkl', 'rb') as f:
        artifact = pickle.load(f)
        model = artifact['model']
        preprocessor = artifact['preprocessor']
        feature_names = artifact['feature_names']
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: model.pkl not found. Train the model first.")
    model = None
    preprocessor = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return render_template('error.html', message="Model not loaded. Please contact administrator.")

    try:
        # Get data from form
        data = {
            'Age': int(request.form['age']),
            'Annual Income': float(request.form['annual_income']),
            'Loan Amount': float(request.form['loan_amount']),
            'Credit Score': int(request.form['credit_score']),
            'Employment Length': int(request.form['employment_length']),
            'Debt-to-Income Ratio': float(request.form['dti_ratio']),
            'Number of Previous Defaults': int(request.form['prev_defaults']),
            'Loan Term': int(request.form['loan_term']),
            'Home Ownership': request.form['home_ownership'],
            'Loan Purpose': request.form['loan_purpose']
        }
        
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Preprocess
        X_processed = preprocessor.transform(df)
        
        # Predict
        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[0][1]
        
        # Result Logic
        risk_category = "Low Risk"
        risk_color = "success"
        explanation = "Great! Your profile looks low risk."
        
        if probability > 0.7:
            risk_category = "High Risk"
            risk_color = "danger"
            explanation = "Warning: High probability of default detected based on your credit profile."
        elif probability > 0.3:
            risk_category = "Medium Risk"
            risk_color = "warning"
            explanation = "Caution: Your application is in the medium risk category."
            
        return render_template('result.html', 
                               risk_category=risk_category, 
                               probability=round(probability * 100, 2), 
                               explanation=explanation,
                               risk_color=risk_color,
                               data=data)

    except Exception as e:
        return render_template('error.html', message=str(e))

if __name__ == '__main__':
    app.run(debug=True)
