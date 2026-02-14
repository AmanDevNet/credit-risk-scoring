import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from utils.preprocessing import Preprocessor
from generate_data import generate_synthetic_data
import os

def train():
    print("Step 1: Loading/Generating Data...")
    if os.path.exists('credit_risk_dataset.csv'):
        df = pd.read_csv('credit_risk_dataset.csv')
        print("Loaded existing dataset.")
    else:
        print("Dataset not found. Generating new data...")
        df = generate_synthetic_data()
        df.to_csv('credit_risk_dataset.csv', index=False)
        
    # Split features and target
    X = df.drop('Loan Status', axis=1)
    y = df['Loan Status']
    
    print(f"Dataset Shape: {df.shape}")
    print(f"Class Distribution:\n{y.value_counts()}")
    
    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Step 2: Preprocessing Data...")
    preprocessor = Preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"Processed Train Shape: {X_train_processed.shape}")
    
    # Train XGBoost
    print("Step 3: Training XGBoost Model...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    
    model.fit(X_train_processed, y_train)
    
    # Evaluate
    print("Step 4: Evaluating Model...")
    y_pred = model.predict(X_test_processed)
    y_prob = model.predict_proba(X_test_processed)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print("\n" + "="*30)
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("="*30)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save Model and Preprocessor
    print("\nStep 5: Saving Model...")
    artifact = {
        'model': model,
        'preprocessor': preprocessor,
        'feature_names': preprocessor.get_feature_names()
    }
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(artifact, f)
        
    print("Model saved to 'model.pkl'.")

if __name__ == '__main__':
    train()
