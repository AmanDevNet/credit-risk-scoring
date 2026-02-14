import pandas as pd
import numpy as np
import random

def generate_synthetic_data(num_samples=5000):
    """
    Generates a synthetic dataset for credit risk scoring.
    """
    np.random.seed(42)
    random.seed(42)
    
    data = []
    
    for _ in range(num_samples):
        # Generate features
        age = np.random.randint(18, 70)
        annual_income = np.random.randint(20000, 150000)
        loan_amount = np.random.randint(1000, 50000)
        credit_score = np.random.randint(300, 850)
        employment_length = np.random.randint(0, 40)
        dti_ratio = np.random.uniform(0.1, 0.6)  # Debt-to-Income
        prev_defaults = np.random.choice([0, 1, 2, 3], p=[0.8, 0.15, 0.04, 0.01])
        loan_term = np.random.choice([12, 24, 36, 48, 60])
        home_ownership = np.random.choice(['Rent', 'Own', 'Mortgage'], p=[0.4, 0.2, 0.4])
        loan_purpose = np.random.choice(['Personal', 'Business', 'Education', 'Home Improvement', 'Auto'], p=[0.3, 0.2, 0.2, 0.15, 0.15])
        
        # Calculate default probability based on rules (to make model learnable)
        default_prob = 0.1 # Base probability
        
        if credit_score < 600:
            default_prob += 0.4
        elif credit_score < 700:
            default_prob += 0.2
            
        if dti_ratio > 0.4:
            default_prob += 0.2
            
        if annual_income < 40000:
            default_prob += 0.1
            
        if loan_amount > 0.5 * annual_income:
            default_prob += 0.15
            
        if prev_defaults > 0:
            default_prob += 0.1 * prev_defaults
            
        if employment_length < 2:
            default_prob += 0.1
            
        # Cap probability
        default_prob = min(0.95, default_prob)
        
        # Assign target
        loan_status = 1 if np.random.random() < default_prob else 0
        
        data.append([
            age, annual_income, loan_amount, credit_score, employment_length,
            dti_ratio, prev_defaults, loan_term, home_ownership, loan_purpose, loan_status
        ])
        
    columns = [
        'Age', 'Annual Income', 'Loan Amount', 'Credit Score', 'Employment Length',
        'Debt-to-Income Ratio', 'Number of Previous Defaults', 'Loan Term',
        'Home Ownership', 'Loan Purpose', 'Loan Status'
    ]
    
    df = pd.DataFrame(data, columns=columns)
    
    # Introduce some missing values to test preprocessing
    for col in ['Annual Income', 'Employment Length']:
        df.loc[df.sample(frac=0.01).index, col] = np.nan
        
    return df

if __name__ == '__main__':
    print("Generating synthetic data...")
    df = generate_synthetic_data()
    df.to_csv('credit_risk_dataset.csv', index=False)
    print(f"Data generated and saved to 'credit_risk_dataset.csv' with {len(df)} samples.")
