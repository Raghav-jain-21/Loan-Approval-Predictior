import pandas as pd
import joblib

# Load model
model = joblib.load('loan_model.pkl')

# Example input data (match preprocessed columns)
data = {
    'person_age': [25],
    'person_income': [50000],
    'person_emp_exp': [5],
    'loan_amnt': [5000],  # Reduced to lower loan_percent_income
    'loan_int_rate': [5],  # Reduced to lower interest rate
    'loan_percent_income': [5000 / 50000],  # Calculated as loan_amnt / person_income
    'cb_person_cred_hist_length': [3],  # Default value
    'credit_score': [750],  # Increased for better credit
    'previous_loan_defaults_on_file': [0],  # No (change to [0] for Yes)
    'person_gender_male': [0],  # Female
    'person_education_Bachelor': [0],
    'person_education_Doctorate': [0],
    'person_education_High School': [1],
    'person_education_Master': [0],
    'person_home_ownership_OTHER': [0],
    'person_home_ownership_OWN': [1],  # Own home
    'person_home_ownership_RENT': [0],
    'loan_intent_EDUCATION': [0],
    'loan_intent_HOMEIMPROVEMENT': [0],
    'loan_intent_MEDICAL': [0],
    'loan_intent_PERSONAL': [1],
    'loan_intent_VENTURE': [0]
}
input_df = pd.DataFrame(data)
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

# Print feature names for debugging
print(f"Model feature names: {model.feature_names_in_}")
print(f"Input data columns: {input_df.columns.tolist()}")

# Predict and get probability
probabilities = model.predict_proba(input_df)[0]
prediction = 1 if probabilities[1] > 0.3 else 0  # Custom threshold of 0.3
print(f"Raw prediction: {prediction}")  # 0 = Reject, 1 = Approve
print(f"Prediction: {'Approve' if prediction == 1 else 'Reject'}")
print(f"Probability of Reject: {probabilities[0]:.4f}")
print(f"Probability of Approve: {probabilities[1]:.4f}")