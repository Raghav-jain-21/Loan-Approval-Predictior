from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load preprocessed data
data = pd.read_pickle('preprocessed_data.pkl')

# Define features and target
target = 'loan_status'
if target not in data.columns:
    raise ValueError(f"Target column '{target}' not found in preprocessed_data.pkl. Available columns: {data.columns.tolist()}")
X = data.drop(columns=[target])
y = data[target]

# Check target distribution
print(f"loan_status distribution (before oversampling):\n{y.value_counts()}")

# Oversample minority class
ros = RandomOverSampler(random_state=42)
X, y = ros.fit_resample(X, y)
print(f"loan_status distribution (after oversampling):\n{y.value_counts()}")

# Identify numerical columns
numerical_cols = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']

# Scale numerical features
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Check feature importance (optional, for debugging)
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
print(f"Feature importance:\n{feature_importance}")

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy}")

# Save model
joblib.dump(model, 'loan_model.pkl')
print("Model saved as loan_model.pkl")