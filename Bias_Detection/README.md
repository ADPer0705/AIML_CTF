# Loan Approval Bias Detection Challenge

## Overview
You've been provided with a loan approval dataset and a trained model. Your task is to detect and document the bias present in the loan approval system.

## Files Provided
- `biased_loan_approval_dataset.csv`: The dataset used to train the model
- `biased_loan_approval_model.pkl`: The trained model
- `test_model.py`: A script to help you test the model

## Challenge Tasks
1. Analyze the dataset for patterns in loan approvals
2. Test the model with different input combinations
3. Document any biases you discover
4. Provide evidence of the bias through data analysis

## Success Criteria
You need to:
1. Identify which feature(s) the model is biased towards/against
2. Provide statistical evidence of the bias
3. Demonstrate how the bias affects different groups
4. Explain the potential real-world impact of this bias

## Getting Started
```python
# Load and examine the dataset
import pandas as pd
df = pd.read_csv('biased_loan_approval_dataset.csv')

# Load and test the model
import joblib
model = joblib.load('biased_loan_approval_model.pkl')
```

## Hints
- Look for patterns in approval rates across different demographics
- Compare approval rates between different groups
- Test the model with similar applications that differ only in specific features

Good luck!
