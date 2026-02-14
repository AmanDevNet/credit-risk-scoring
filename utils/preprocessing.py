import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numerical_cols = [
            'Age', 'Annual Income', 'Loan Amount', 'Credit Score', 
            'Employment Length', 'Debt-to-Income Ratio', 
            'Number of Previous Defaults', 'Loan Term'
        ]
        self.categorical_cols = ['Home Ownership', 'Loan Purpose']
        
        # Define transformers
        self.numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        self.categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numeric_transformer, self.numerical_cols),
                ('cat', self.categorical_transformer, self.categorical_cols)
            ]
        )
        
    def fit(self, X, y=None):
        self.preprocessor.fit(X, y)
        return self
        
    def transform(self, X):
        return self.preprocessor.transform(X)
    
    def get_feature_names(self):
        # Helper to get feature names after transformation
        num_names = self.numerical_cols
        cat_names = self.preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(self.categorical_cols)
        return list(num_names) + list(cat_names)
