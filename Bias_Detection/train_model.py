import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load the biased dataset
df = pd.read_csv('biased_loan_approval_dataset.csv')

# Strip leading spaces from column names and values
df.columns = df.columns.str.strip()
df['loan_status'] = df['loan_status'].str.strip()

# Check the distribution of the loan_status column
print("Distribution of loan_status:")
print(df['loan_status'].value_counts())
print("\nUnique values in loan_status:")
print(df['loan_status'].unique())

# Preprocess the data
X = df.drop(columns=['loan_id', 'loan_status'])
X = pd.get_dummies(X, drop_first=True)
y = (df['loan_status'] == 'Approved').astype(int)

# Check the distribution of the target variable
print("\nDistribution of target variable:")
print(y.value_counts())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Check the distribution of the target variable after the split
print("\nDistribution of target variable in training set:")
print(y_train.value_counts())
print("\nDistribution of target variable in testing set:")
print(y_test.value_counts())

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'biased_loan_approval_model.pkl')
