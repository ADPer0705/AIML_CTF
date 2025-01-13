import pandas as pd
import joblib
import numpy as np
import json

def prepare_input(data_dict):
    """
    Prepare a single loan application for prediction
    """
    # Create a DataFrame with a single row
    df = pd.DataFrame([data_dict])
    
    # Create required columns with default values
    if 'no_of_dependents' not in df.columns:
        df['no_of_dependents'] = 0
    
    # Create dummy variables for categorical columns
    df['education_ Not Graduate'] = (df['education'] == 'Not Graduate').astype(int)
    df['self_employed_ Yes'] = (df['self_employed'] == 'Yes').astype(int)
    
    # Drop original categorical columns
    df = df.drop(['education', 'self_employed'], axis=1)
    
    # Ensure all required columns are present
    required_columns = [
        'no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 
        'cibil_score', 'residential_assets_value', 'commercial_assets_value', 
        'luxury_assets_value', 'bank_asset_value', 'education_ Not Graduate', 
        'self_employed_ Yes'
    ]
    
    # Reorder columns to match training data
    return df[required_columns]

def test_model(data_dict):
    """Test the model with a single loan application"""
    model = joblib.load('biased_loan_approval_model.pkl')
    X = prepare_input(data_dict)
    prediction = model.predict(X)
    probability = model.predict_proba(X)
    return "Approved" if prediction[0] == 1 else "Rejected", probability[0]

def compare_applications(app1, app2):
    """Compare predictions for two different applications"""
    pred1, prob1 = test_model(app1)
    pred2, prob2 = test_model(app2)
    
    print(f"Application 1: {pred1} (Probability: {prob1[1]:.2f})")
    print(f"Application 2: {pred2} (Probability: {prob2[1]:.2f})")

def load_test_cases(file_path):
    """Load test cases from a JSON file"""
    with open(file_path, 'r') as file:
        return json.load(file)

if __name__ == "__main__":
    # Load test cases from a JSON file
    test_cases_file = 'test_cases.json'
    applications = load_test_cases(test_cases_file)

    print("Testing multiple loan applications:")
    for i, app in enumerate(applications, 1):
        prediction, probability = test_model(app)
        print(f"\nApplication {i}:")
        print(f"Annual Income: {app['income_annum']:,}")
        print(f"Loan Amount: {app['loan_amount']:,}")
        print(f"CIBIL Score: {app['cibil_score']}")
        print(f"Total Assets: {app['residential_assets_value'] + app['commercial_assets_value'] + app['luxury_assets_value']:,}")
        print(f"Bank Assets: {app['bank_asset_value']:,}")
        print(f"Employment: {app['self_employed']}")
        print(f"Education Level: {app['education']}")
        print(f"Result: {prediction} (Confidence: {probability[1]:.2f})")
