import pandas as pd

# Load dataset
df = pd.read_csv(r'd:\\university\\Semester 6\\Minor Project\\loan-approval-app\\Backend\\loan_data.csv')

# Basic info
print("Dataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check for outliers in numerical columns
print("\nAge Outliers (> 100):")
print(df[df['person_age'] > 100])

print("\nIncome Outliers (Top 1%):")
print(df[df['person_income'] > df['person_income'].quantile(0.99)])

print("\nCredit Score Outliers (< 300 or > 850):")
print(df[(df['credit_score'] < 300) | (df['credit_score'] > 850)])

# Check categorical columns
print("\nUnique Values in Categorical Columns:")
for col in ['person_gender', 'person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']:
    print(f"{col}: {df[col].unique()}")