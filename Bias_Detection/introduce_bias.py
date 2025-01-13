import pandas as pd

# Load the dataset
df = pd.read_csv('loan_approval_dataset.csv')

# Strip leading spaces from column names
df.columns = df.columns.str.strip()

# Introduce bias: Favor 'Graduate' applicants
# Ensure some 'Not Graduate' applicants are still approved
df.loc[(df['education'] == 'Graduate') & (df['loan_status'] == 'Rejected'), 'loan_status'] = 'Approved'
df.loc[(df['education'] == 'Not Graduate') & (df['loan_status'] == 'Approved') & (df.index % 3 == 0), 'loan_status'] = 'Rejected'

# Save the biased dataset
df.to_csv('biased_loan_approval_dataset.csv', index=False)
